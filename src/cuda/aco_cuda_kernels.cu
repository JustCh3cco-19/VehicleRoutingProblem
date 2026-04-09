#include "aco_cuda_kernels.h"
#include <device_launch_parameters.h>
#include <math.h>

// --- Helper Functions (Warp Level) ---

__device__ __forceinline__ float fast_rand(uint32_t* state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return (*state & 0x7FFFFFFF) / 2147483648.0f;
}

__device__ __forceinline__ void set_bit(uint32_t* bitmask, int node) {
    atomicOr(&bitmask[node >> 5], (1u << (node & 0x1F)));
}

__device__ __forceinline__ bool get_bit(const uint32_t* bitmask, int node) {
    return (bitmask[node >> 5] >> (node & 0x1F)) & 1u;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_prefix_sum(float val, int lane) {
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane >= offset) val += n;
    }
    return val;
}

// --- Kernels ---

// K1: Setup statico delle liste candidati (N blocchi x 32 thread)
extern __shared__ uint32_t shared_mem[];

__global__ void kernel_setup_aco(const float* d_costs, int* d_cand_list, int n) {
    int i = blockIdx.x; 
    int lane = threadIdx.x % 32;
    if (i > n || threadIdx.x >= 32) return;

    int words_per_mask = (n + 1 + 31) / 32;
    uint32_t* my_bitmask = shared_mem; 

    for (int j = lane; j < words_per_mask; j += 32) my_bitmask[j] = 0;
    __syncwarp();

    for (int k = 0; k < MAX_CANDIDATES; ++k) {
        float local_min_c = 1e30f;
        int local_min_node = -1;
        
        for (int j = 1 + lane; j <= n; j += 32) {
            if (i == j) continue;
            
            if (!((my_bitmask[j >> 5] >> (j & 0x1F)) & 1u)) {
                float c = d_costs[i * (n + 1) + j];
                if (c < local_min_c) {
                    local_min_c = c;
                    local_min_node = j;
                }
            }
        }
        
        // Warp reduction per trovare il minimo (costo, nodo)
        for (int offset = 16; offset > 0; offset /= 2) {
            float remote_c = __shfl_down_sync(0xFFFFFFFF, local_min_c, offset);
            int remote_node = __shfl_down_sync(0xFFFFFFFF, local_min_node, offset);
            if (remote_c < local_min_c) {
                local_min_c = remote_c;
                local_min_node = remote_node;
            }
        }
        
        local_min_node = __shfl_sync(0xFFFFFFFF, local_min_node, 0);
        
        if (local_min_node == -1) break;

        if (lane == 0) {
            my_bitmask[local_min_node >> 5] |= (1u << (local_min_node & 0x1F));
            d_cand_list[i * MAX_CANDIDATES + k] = local_min_node;
        }
        __syncwarp();
    }
}

// K2: Selezione dinamica dei candidati basata sugli score (Warp-Centric)
extern __shared__ uint32_t shared_mem[];

__global__ void kernel_precompute_candidate_scores(
    const float* d_tau, const float* d_costs, int* d_cand_list, 
    float* d_cand_scores, int n, float alpha, float beta) {
    
    int lane = threadIdx.x % 32;
    int i = blockIdx.x; 
    if (i > n || threadIdx.x >= 32) return; 

    int words_per_mask = (n + 1 + 31) / 32;
    uint32_t* my_bitmask = shared_mem; 

    for (int j = lane; j < words_per_mask; j += 32) my_bitmask[j] = 0;
    __syncwarp();

    for (int k = 0; k < MAX_CANDIDATES; ++k) {
        float max_score = -1.0f;
        int best_node = -1;

        for (int j = 1 + lane; j <= n; j += 32) {
            if (i == j) continue;
            if (!((my_bitmask[j >> 5] >> (j & 0x1F)) & 1u)) {
                float tau = d_tau[i * (n + 1) + j];
                float cost = d_costs[i * (n + 1) + j];
                float eta = 1.0f / (cost + 1e-9f);
                float score = powf(tau, alpha) * powf(eta, beta); 
                if (score > max_score) { max_score = score; best_node = j; }
            }
        }

        for (int offset = 16; offset > 0; offset /= 2) {
            float remote_score = __shfl_down_sync(0xFFFFFFFF, max_score, offset);
            int remote_node = __shfl_down_sync(0xFFFFFFFF, best_node, offset);
            if (remote_score > max_score) { max_score = remote_score; best_node = remote_node; }
        }

        best_node = __shfl_sync(0xFFFFFFFF, best_node, 0);
        max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

        if (best_node == -1) break;

        if (lane == 0) {
            my_bitmask[best_node >> 5] |= (1u << (best_node & 0x1F));
            d_cand_list[i * MAX_CANDIDATES + k] = best_node;
            d_cand_scores[i * MAX_CANDIDATES + k] = max_score;
        }
        __syncwarp();
    }
}

// K4: Construct Solutions (Warp-Centric)
__global__ void kernel_construct_solutions(
    int m, int K, int n, float Q, unsigned int seed,
    const float* d_costs, const float* d_tau, float alpha, float beta,
    const int* d_cand_list, const float* d_cand_scores,
    float* d_ant_costs, int* d_ant_routes) {

    int warp_in_block = threadIdx.x / 32;
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (global_warp_id >= m) return;

    uint32_t rng_state = seed + global_warp_id;

    int words_per_mask = (n + 1 + 31) / 32;
    int words_per_vehicle = sizeof(VehicleState) / sizeof(uint32_t);
    uint32_t* my_bitmask = &shared_mem[warp_in_block * (words_per_mask + K * words_per_vehicle)];
    VehicleState* my_fleet = (VehicleState*)&my_bitmask[words_per_mask];

    for (int i = lane; i < words_per_mask; i += 32) my_bitmask[i] = 0;
    
    float initial_capacity = (float)(n - K + 3);
    if (lane == 0) {
        for (int v = 0; v < K; v++) {
            my_fleet[v].curr_node = 0;
            my_fleet[v].capacity = initial_capacity;
            my_fleet[v].cost = 0.0f;
        }
    }
    __syncwarp();

    int unvisited = n;
    // Loop sui veicoli (costruzione sequenziale delle rotte)
    for (int v_idx = 0; v_idx < K && unvisited > 0; v_idx++) {
        int curr = 0; // Parte dal deposito
        int route_len = 0;
        
        while (unvisited > 0 && my_fleet[v_idx].capacity > 0.5f) {
            int next_node = -1;

            // 1. Candidate List
            int cand_id = d_cand_list[curr * MAX_CANDIDATES + lane];
            float score = d_cand_scores[curr * MAX_CANDIDATES + lane];
            if (cand_id <= 0 || get_bit(my_bitmask, cand_id)) score = 0.0f;

            float sum_p = warp_reduce_sum(score);
            float total_p = __shfl_sync(0xFFFFFFFF, sum_p, 0);
            
            if (total_p > 1e-9f) {
                float r_val;
                if (lane == 0) r_val = fast_rand(&rng_state);
                float r = __shfl_sync(0xFFFFFFFF, r_val, 0) * total_p;
                float prefix = warp_prefix_sum(score, lane);
                unsigned int mask = __ballot_sync(0xFFFFFFFF, prefix >= r && score > 0);
                if (mask != 0) next_node = __shfl_sync(0xFFFFFFFF, cand_id, __ffs(mask) - 1);
            }

            // 2. Fallback Globale
            if (next_node == -1) {
                float local_sum = 0.0f;
                for (int j = 1 + lane; j <= n; j += 32) {
                    if (!get_bit(my_bitmask, j)) {
                        float tau = d_tau[curr * (n + 1) + j];
                        float cost = d_costs[curr * (n + 1) + j];
                        float eta = 1.0f / (cost + 1e-9f);
                        local_sum += powf(tau, alpha) * powf(eta, beta);
                    }
                }
                float total_fallback_p = warp_reduce_sum(local_sum);
                total_fallback_p = __shfl_sync(0xFFFFFFFF, total_fallback_p, 0);
                if (total_fallback_p > 1e-9f) {
                    float r_val;
                    if (lane == 0) r_val = fast_rand(&rng_state);
                    float r = __shfl_sync(0xFFFFFFFF, r_val, 0) * total_fallback_p;
                    float running_sum = 0.0f;
                    for (int j = 1 + lane; j <= n; j += 32) {
                        float s = 0.0f;
                        if (!get_bit(my_bitmask, j)) {
                            float tau = d_tau[curr * (n + 1) + j];
                            float cost = d_costs[curr * (n + 1) + j];
                            float eta = 1.0f / (cost + 1e-9f);
                            s = powf(tau, alpha) * powf(eta, beta);
                        }
                        float chunk_sum = warp_reduce_sum(s);
                        float current_total = __shfl_sync(0xFFFFFFFF, running_sum + chunk_sum, 0);
                        if (r <= current_total) {
                            float prefix = warp_prefix_sum(s, lane);
                            unsigned int mask = __ballot_sync(0xFFFFFFFF, (running_sum + prefix) >= r && s > 0);
                            if (mask != 0) next_node = __shfl_sync(0xFFFFFFFF, (int)j, __ffs(mask) - 1);
                            break;
                        }
                        running_sum += chunk_sum;
                    }
                }
            }

            if (next_node != -1) {
                if (lane == 0) {
                    set_bit(my_bitmask, next_node);
                    my_fleet[v_idx].cost += d_costs[curr * (n + 1) + next_node];
                    my_fleet[v_idx].curr_node = next_node;
                    my_fleet[v_idx].capacity -= 1.0f;
                    // Salvataggio contiguo: rotta v parte all'offset v * (n/K + 2) circa
                    // Per semplicità usiamo un layout flat: [R1][R2]...[RK] dove ogni Ri ha spazio per n+2
                    int route_offset = v_idx * (n + 2);
                    d_ant_routes[global_warp_id * (n + 2 * K) + route_offset + route_len] = next_node;
                    route_len++;
                }
                if (__shfl_sync(0xFFFFFFFF, next_node != -1, 0)) unvisited--;
                curr = __shfl_sync(0xFFFFFFFF, next_node, 0);
            } else {
                break; // Veicolo bloccato
            }
            __syncwarp();
        }
        // Il veicolo torna al deposito (già implicito nel calcolo del costo finale)
    }

    if (lane == 0) {
        float total_ant_cost = 0.0f;
        for (int v = 0; v < K; v++) {
            total_ant_cost += my_fleet[v].cost + d_costs[my_fleet[v].curr_node * (n + 1) + 0];
        }
        d_ant_costs[global_warp_id] = total_ant_cost;
    }
}

// K5: Pheromones
__global__ void kernel_evaporate_pheromones(float* d_tau, float rho, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_tau[idx] *= (1.0f - rho);
}

__global__ void kernel_deposit_pheromones(float* d_tau, const int* d_best_route, int n, int K, float deposit) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= K) return;
    
    // Ogni rotta v ha a disposizione (n+2) slot consecutivi
    int route_offset = v * (n + 2);
    int prev = 0;
    for (int s = 0; s < (n + 2); s++) {
        int curr = d_best_route[route_offset + s];
        if (curr > 0 && curr <= n) {
            atomicAdd(&d_tau[prev * (n + 1) + curr], deposit);
            prev = curr;
        } else if (s > 0 && curr == 0) {
            // Fine della rotta (torna al deposito)
            break;
        }
    }
    atomicAdd(&d_tau[prev * (n + 1) + 0], deposit);
}

// --- Wrapper functions ---
void launch_evaporate_pheromones(float* d_tau, float rho, int size, int threads_per_block) {
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    kernel_evaporate_pheromones<<<blocks, threads_per_block>>>(d_tau, rho, size);
}

void launch_deposit_pheromones(float* d_tau, const int* d_best_route, int n, int K, float deposit) {
    int threads = 64;
    int blocks = (K + threads - 1) / threads;
    kernel_deposit_pheromones<<<blocks, threads>>>(d_tau, d_best_route, n, K, deposit);
}

void launch_precompute_candidate_scores(const float* d_tau, const float* d_costs, int* d_cand_list, float* d_cand_scores, int n, float alpha, float beta, int threads_per_block, size_t shared_mem_size) {
    kernel_precompute_candidate_scores<<<n + 1, 32, shared_mem_size>>>(d_tau, d_costs, d_cand_list, d_cand_scores, n, alpha, beta);
}

void launch_construct_solutions(int m, int K, int n, float Q, unsigned int seed, const float* d_costs, const float* d_tau, float alpha, float beta, const int* d_cand_list, const float* d_cand_scores, float* d_ant_costs, int* d_ant_routes, int threads_per_block, size_t shared_mem_size) {
    int blocks = (m * 32 + threads_per_block - 1) / threads_per_block;
    kernel_construct_solutions<<<blocks, threads_per_block, shared_mem_size>>>(m, K, n, Q, seed, d_costs, d_tau, alpha, beta, d_cand_list, d_cand_scores, d_ant_costs, d_ant_routes);
}

void launch_setup_aco(const float* d_costs, int* d_cand_list, int n, int threads_per_block, size_t shared_mem_size) {
    kernel_setup_aco<<<n + 1, 32, shared_mem_size>>>(d_costs, d_cand_list, n);
}
