#include "aco_cuda_kernels.h"
#include <curand_kernel.h>
#include <math.h>

#define EPS 1e-9

__global__ void init_curand_states_kernel(curandState *states, int m, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

void init_curand_states(void *d_curand_states, int m, unsigned int seed, int threads_per_block) {
    int blocks = (m + threads_per_block - 1) / threads_per_block;
    init_curand_states_kernel<<<blocks, threads_per_block>>>((curandState *)d_curand_states, m, seed);
}

__global__ void init_matrices_kernel(double *eta, double *tau, const double *c, int n, double tau0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i == j) {
            eta[idx] = 0.0;
            tau[idx] = 0.0;
        } else {
            eta[idx] = 1.0 / (c[idx] + EPS);
            tau[idx] = tau0;
        }
    }
}

void launch_init_matrices(double *d_eta, double *d_tau, const double *d_c, int n, double tau0, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    init_matrices_kernel<<<blocks, threads_per_block>>>(d_eta, d_tau, d_c, n, tau0);
}

__global__ void precompute_scores_kernel(double *scores, const double *eta, const double *tau, int n, double alpha, double beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i != j) {
            scores[idx] = pow(tau[idx], alpha) * pow(eta[idx], beta);
        } else {
            scores[idx] = 0.0;
        }
    }
}

void launch_precompute_scores(double *d_scores, const double *d_eta, const double *d_tau,
                              int n, double alpha, double beta, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    precompute_scores_kernel<<<blocks, threads_per_block>>>(d_scores, d_eta, d_tau, n, alpha, beta);
}

__device__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ double warp_prefix_sum(double val, int lane_id) {
    for (int offset = 1; offset < warpSize; offset *= 2) {
        double n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) val += n;
    }
    return val;
}

__global__ void construct_solutions_warp_v3_kernel(int m, int K, int n, const double *scores,
                                                   const double *c, double *costs, int *routes,
                                                   int *route_lens, curandState *states, bool *visited) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (global_warp_id >= m) return;

    curandState local_state = states[global_warp_id];
    bool *my_visited = &visited[global_warp_id * (n + 1)];
    
    for (int i = lane_id; i <= n; i += 32) my_visited[i] = false;
    __syncwarp();

    int *my_routes = &routes[global_warp_id * K * (n + 2)];
    int *my_route_lens = &route_lens[global_warp_id * K];
    double total_cost = 0.0;
    int unvisited_count = n;

    for (int vehicle = 0; vehicle < K; ++vehicle) {
        int *r = &my_routes[vehicle * (n + 2)];
        int len = 0;
        if (lane_id == 0) r[len++] = 0;
        int current = 0;

        while (unvisited_count > 0 && unvisited_count > (K - 1 - vehicle)) {
            // 1. CALCOLO DENOMINATORE (Somma Totale)
            double local_sum = 0.0;
            for (int j = 1 + lane_id; j <= n; j += 32) {
                if (!my_visited[j]) local_sum += scores[current * (n + 1) + j];
            }
            double denom = warp_reduce_sum(local_sum);
            denom = __shfl_sync(0xFFFFFFFF, denom, 0);

            int next_node = -1;
            if (denom <= 0.0) {
                // Fallback: primo nodo non visitato
                for (int j = 1; j <= n; ++j) {
                    if (!my_visited[j]) { next_node = j; break; }
                }
            } else {
                double r_val;
                if (lane_id == 0) r_val = curand_uniform_double(&local_state);
                r_val = __shfl_sync(0xFFFFFFFF, r_val, 0);
                double threshold = r_val * denom;
                double cumulative_base = 0.0;

                // 2. SCELTA PARALLELA A BLOCCHI DI 32 (PARALLEL SCAN)
                for (int j_base = 1; j_base <= n; j_base += 32) {
                    int j = j_base + lane_id;
                    double s = (j <= n && !my_visited[j]) ? scores[current * (n + 1) + j] : 0.0;
                    
                    double prefix_s = warp_prefix_sum(s, lane_id);
                    double total_chunk_s = __shfl_sync(0xFFFFFFFF, prefix_s, 31);

                    if (cumulative_base + total_chunk_s >= threshold) {
                        // Il nodo è in questo chunk!
                        unsigned int mask = __ballot_sync(0xFFFFFFFF, cumulative_base + prefix_s >= threshold);
                        int winner_lane = __ffs(mask) - 1;
                        next_node = j_base + winner_lane;
                        break;
                    }
                    cumulative_base += total_chunk_s;
                }
            }

            if (lane_id == 0) {
                r[len++] = next_node;
                my_visited[next_node] = true;
                total_cost += c[current * (n + 1) + next_node];
                unvisited_count--;
            }
            next_node = __shfl_sync(0xFFFFFFFF, next_node, 0);
            unvisited_count = __shfl_sync(0xFFFFFFFF, unvisited_count, 0);
            current = next_node;
        }
        if (lane_id == 0) {
            r[len++] = 0;
            total_cost += c[current * (n + 1) + 0];
            my_route_lens[vehicle] = len;
        }
    }
    if (lane_id == 0) {
        costs[global_warp_id] = total_cost;
        states[global_warp_id] = local_state;
    }
}

void launch_construct_solutions(int m, int K, int n, const double *d_scores,
                                const double *d_c, double *d_costs, int *d_routes,
                                int *d_route_lens, void *d_curand_states, bool *d_visited, int threads_per_block) {
    int total_threads = m * 32;
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    construct_solutions_warp_v3_kernel<<<blocks, threads_per_block>>>(m, K, n, d_scores, d_c, d_costs,
                                                                       d_routes, d_route_lens,
                                                                       (curandState *)d_curand_states, d_visited);
}

__global__ void deposit_pheromones_kernel(double *tau, int n, int K, const int *routes, const int *route_lens, double deposit) {
    int vehicle = blockIdx.x * blockDim.x + threadIdx.x;
    if (vehicle >= K) return;

    const int *r = &routes[vehicle * (n + 2)];
    int len = route_lens[vehicle];
    for (int t = 0; t + 1 < len; ++t) {
        int u = r[t];
        int v = r[t + 1];
        atomicAdd(&tau[u * (n + 1) + v], deposit);
        atomicAdd(&tau[v * (n + 1) + u], deposit);
    }
}

void launch_deposit_pheromones(double *d_tau, int n, int K, const int *d_routes, const int *d_route_lens, double deposit, int threads_per_block) {
    int blocks = (K + threads_per_block - 1) / threads_per_block;
    deposit_pheromones_kernel<<<blocks, threads_per_block>>>(d_tau, n, K, d_routes, d_route_lens, deposit);
}

__global__ void evaporate_pheromones_kernel(double *tau, int n, double rho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i != j) tau[idx] *= (1.0 - rho);
    }
}

void launch_evaporate_pheromones(double *d_tau, int n, double rho, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    evaporate_pheromones_kernel<<<blocks, threads_per_block>>>(d_tau, n, rho);
}
