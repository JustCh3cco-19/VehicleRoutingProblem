#include "aco_cuda_v6_kernels.h"

#include <math.h>
#include <stdio.h>

/* -- Funzioni Helper Dispositivo -- */

__device__ static inline float cuda_v6_rand01(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return (float)(*state) / 4294967295.0f;
}

__device__ static inline float dequantize_tau(uint8_t q, float log_min, float log_step) {
  return expf((float)q * log_step + log_min);
}

__device__ static inline uint8_t quantize_tau(float tau, float log_min, float log_step, uint8_t q_min, uint8_t q_max) {
  if (tau <= 0.0f) return q_min;
  float log_t = logf(tau);
  float q_f = (log_t - log_min) / log_step;
  if (q_f <= (float)q_min) return q_min;
  if (q_f >= (float)q_max) return q_max;
  return (uint8_t)roundf(q_f);
}

__device__ static inline float calc_dist(float2 a, float2 b) {
  float dx = a.x - b.x;
  float dy = a.y - b.y;
  return sqrtf(dx * dx + dy * dy);
}

__device__ static inline float cuda_v6_warp_sum(float value) {
  for (int offset = CUDA_V6_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ static inline float cuda_v6_warp_prefix_sum(float value, int lane) {
  for (int offset = 1; offset < CUDA_V6_WARP_SIZE; offset <<= 1) {
    float other = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  return value;
}

__device__ static inline unsigned int cuda_v6_warp_sum_u32(unsigned int value) {
  for (int offset = CUDA_V6_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ static inline uint64_t cuda_v6_relevant_mask_for_word(int word_idx, int n) {
  int start_node = word_idx << 6;
  int end_node = start_node + 63;
  uint64_t mask = 0xFFFFFFFFFFFFFFFFull;

  if (start_node == 0) {
    mask &= ~1ull; // Node 0 (depot) is handled separately
  }
  if (end_node > n) {
    int valid_bits = n - start_node + 1;
    if (valid_bits <= 0) return 0ull;
    if (valid_bits < 64) {
      mask &= ((1ull << valid_bits) - 1ull);
    }
  }
  return mask;
}

/* -- Kernels -- */

__global__ void kernel_init_tau_v6(uint8_t *d_tau, uint64_t total_elements, uint8_t q_tau0) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    d_tau[idx] = q_tau0;
  }
}

__global__ void kernel_evaporate_tau_v6(uint8_t *d_tau, uint64_t total_elements, uint8_t delta, uint8_t q_min) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total_elements) {
    uint8_t q = d_tau[idx];
    if (q > q_min + delta) {
      d_tau[idx] = q - delta;
    } else {
      d_tau[idx] = q_min;
    }
  }
}

__global__ void kernel_build_candidate_lists_v6(const float2 *d_coords,
                                                int *d_candidate_idx,
                                                float *d_eta_beta,
                                                CudaV6Params params) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int cand_k = params.cand_k;
  
  if (i > params.n) return;

  int best_nodes[CUDA_V6_MAX_CANDIDATES];
  float best_costs[CUDA_V6_MAX_CANDIDATES];
  
  for (int t = 0; t < cand_k; ++t) {
    best_nodes[t] = 0;
    best_costs[t] = INFINITY;
  }

  float2 p_i = d_coords[i];

  for (int j = 1; j <= params.n; ++j) {
    if (i == j) continue;
    float d = calc_dist(p_i, d_coords[j]);
    
    int pos = -1;
    for (int t = 0; t < cand_k; ++t) {
      if (d < best_costs[t]) {
        pos = t;
        break;
      }
    }
    if (pos >= 0) {
      for (int t = cand_k - 1; t > pos; --t) {
        best_costs[t] = best_costs[t - 1];
        best_nodes[t] = best_nodes[t - 1];
      }
      best_costs[pos] = d;
      best_nodes[pos] = j;
    }
  }

  for (int t = 0; t < cand_k; ++t) {
    int out_idx = i * cand_k + t;
    d_candidate_idx[out_idx] = best_nodes[t];
    if (best_nodes[t] > 0) {
      float eta = 1.0f / (best_costs[t] + CUDA_V6_EPS);
      d_eta_beta[out_idx] = powf(eta, params.beta);
    } else {
      d_eta_beta[out_idx] = 0.0f;
    }
  }
}

__global__ void kernel_reset_ant_state_v6(int *d_routes, 
                                          int *d_route_lengths,
                                          uint64_t *d_visited_l1,
                                          uint64_t *d_visited_l2,
                                          unsigned int *d_rng_states,
                                          CudaV6AntSummary *d_ant_summary,
                                          CudaV6Params params,
                                          unsigned int seed) {
  int ant = blockIdx.x * blockDim.x + threadIdx.x;
  if (ant >= params.m) return;

  uint64_t *l1_row = d_visited_l1 + ant * (params.visited_row_stride / 8);
  uint64_t *l2_row = d_visited_l2 + ant * params.visited_l2_words;

  for (int w = 0; w < params.visited_l1_words; ++w) {
    l1_row[w] = ~cuda_v6_relevant_mask_for_word(w, params.n);
  }
  for (int w = 0; w < params.visited_l2_words; ++w) {
    l2_row[w] = 0ull; 
  }

  for (int v = 0; v < params.K; ++v) {
    d_routes[0 * params.m + ant] = 0; // step 0 is depot
    d_route_lengths[ant * params.K + v] = 1;
  }

  d_rng_states[ant] = seed ^ (1664525u * (unsigned int)(ant + 1));
  d_ant_summary[ant].feasible = 0;
  d_ant_summary[ant].unvisited_count = params.n;
  d_ant_summary[ant].cost = 0.0f;
}

__device__ static int select_from_candidates_v6(int current,
                                                const uint8_t *d_tau,
                                                const int *d_candidate_idx,
                                                const float *d_eta_beta,
                                                const uint64_t *visited_l1,
                                                unsigned int *rng_state,
                                                CudaV6Params params,
                                                float *total_score_out) {
  int lane = threadIdx.x & (CUDA_V6_WARP_SIZE - 1);
  int base = current * params.cand_k;
  int candidate = 0;
  float score = 0.0f;

  if (lane < params.cand_k) {
    candidate = d_candidate_idx[base + lane];
    if (candidate > 0) {
      int word_idx = candidate >> 6;
      int bit_idx = candidate & 63;
      uint64_t mask = 1ull << bit_idx;
      if ((visited_l1[word_idx] & mask) == 0) {
        uint64_t tau_idx = (uint64_t)current * (params.n + 1) + candidate;
        uint8_t q_tau = d_tau[tau_idx];
        float tau = dequantize_tau(q_tau, params.log_tau_min, params.log_tau_step);
        float tau_term = powf(tau, params.alpha);
        score = tau_term * d_eta_beta[base + lane];
      }
    }
  }

  float total = cuda_v6_warp_sum(score);
  total = __shfl_sync(0xFFFFFFFFu, total, 0);
  if (total_score_out) *total_score_out = total;

  if (total <= 0.0f) return -1;

  float threshold = 0.0f;
  if (lane == 0) {
    threshold = cuda_v6_rand01(rng_state) * total;
  }
  threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);

  float prefix = cuda_v6_warp_prefix_sum(score, lane);
  int selected = (score > 0.0f && prefix >= threshold && (prefix - score) < threshold);
  unsigned int mask_sel = __ballot_sync(0xFFFFFFFFu, selected);

  if (mask_sel != 0u) {
    int chosen_lane = __ffs((int)mask_sel) - 1;
    return __shfl_sync(0xFFFFFFFFu, candidate, chosen_lane);
  }

  return -1;
}

__device__ static int select_global_fallback_v6(int current,
                                                const float2 *d_coords,
                                                const uint8_t *d_tau,
                                                const uint64_t *visited_l1,
                                                const uint64_t *visited_l2,
                                                unsigned int *rng_state,
                                                CudaV6Params params,
                                                float *total_score_out) {
  int lane = threadIdx.x & (CUDA_V6_WARP_SIZE - 1);
  float local_total = 0.0f;
  float total;
  float threshold = 0.0f;
  float running = 0.0f;
  
  int word_chunks = (params.visited_l1_words + CUDA_V6_WARP_SIZE - 1) / CUDA_V6_WARP_SIZE;
  float2 p_curr = d_coords[current];

  for (int chunk = 0; chunk < word_chunks; ++chunk) {
    int word_idx = chunk * CUDA_V6_WARP_SIZE + lane;
    uint64_t free_mask = 0ull;
    
    if (word_idx < params.visited_l1_words) {
      int l2_word = word_idx >> 6;
      int l2_bit = word_idx & 63;
      uint64_t l2_mask = 1ull << l2_bit;
      
      if ((visited_l2[l2_word] & l2_mask) == 0) {
        uint64_t relevant_mask = cuda_v6_relevant_mask_for_word(word_idx, params.n);
        free_mask = (~visited_l1[word_idx]) & relevant_mask;
      }
    }

    unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, free_mask != 0ull);
    if (active_mask == 0u) continue;

    if (free_mask != 0ull) {
      int base_node = word_idx << 6;
      uint64_t scan_mask = free_mask;
      while (scan_mask != 0ull) {
        int bit = __ffsll((long long)scan_mask) - 1;
        int node = base_node + bit;
        
        float d = calc_dist(p_curr, d_coords[node]);
        float eta = 1.0f / (d + CUDA_V6_EPS);
        
        uint64_t tau_idx = (uint64_t)current * (params.n + 1) + node;
        uint8_t q_tau = d_tau[tau_idx];
        float tau = dequantize_tau(q_tau, params.log_tau_min, params.log_tau_step);
        
        local_total += powf(tau, params.alpha) * powf(eta, params.beta);
        scan_mask &= (scan_mask - 1ull);
      }
    }
  }

  total = cuda_v6_warp_sum(local_total);
  total = __shfl_sync(0xFFFFFFFFu, total, 0);
  if (total_score_out) *total_score_out = total;
  if (total <= 0.0f) return -1;

  if (lane == 0) {
    threshold = cuda_v6_rand01(rng_state) * total;
  }
  threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);

  // Pass 2: find the node
  for (int chunk = 0; chunk < word_chunks; ++chunk) {
    int word_idx = chunk * CUDA_V6_WARP_SIZE + lane;
    uint64_t free_mask = 0ull;
    float lane_total = 0.0f;
    
    if (word_idx < params.visited_l1_words) {
      int l2_word = word_idx >> 6;
      int l2_bit = word_idx & 63;
      if ((visited_l2[l2_word] & (1ull << l2_bit)) == 0) {
        uint64_t relevant_mask = cuda_v6_relevant_mask_for_word(word_idx, params.n);
        free_mask = (~visited_l1[word_idx]) & relevant_mask;
      }
    }

    unsigned int active_mask = __ballot_sync(0xFFFFFFFFu, free_mask != 0ull);
    if (active_mask == 0u) continue;

    if (free_mask != 0ull) {
      int base_node = word_idx << 6;
      uint64_t scan_mask = free_mask;
      while (scan_mask != 0ull) {
        int bit = __ffsll((long long)scan_mask) - 1;
        int node = base_node + bit;
        float d = calc_dist(p_curr, d_coords[node]);
        float eta = 1.0f / (d + CUDA_V6_EPS);
        uint64_t tau_idx = (uint64_t)current * (params.n + 1) + node;
        float tau = dequantize_tau(d_tau[tau_idx], params.log_tau_min, params.log_tau_step);
        lane_total += powf(tau, params.alpha) * powf(eta, params.beta);
        scan_mask &= (scan_mask - 1ull);
      }
    }

    float chunk_sum = cuda_v6_warp_sum(lane_total);
    chunk_sum = __shfl_sync(0xFFFFFFFFu, chunk_sum, 0);

    if (chunk_sum > 0.0f && threshold < (running + chunk_sum)) {
      float lane_prefix = cuda_v6_warp_prefix_sum(lane_total, lane);
      float lane_start = running + lane_prefix - lane_total;
      float lane_end = running + lane_prefix;
      int lane_selected_node = -1;

      if (lane_total > 0.0f && threshold < lane_end && threshold >= lane_start) {
        int base_node = word_idx << 6;
        uint64_t scan_mask = free_mask;
        float word_running = lane_start;
        while (scan_mask != 0ull) {
          int bit = __ffsll((long long)scan_mask) - 1;
          int node = base_node + bit;
          float d = calc_dist(p_curr, d_coords[node]);
          float eta = 1.0f / (d + CUDA_V6_EPS);
          uint64_t tau_idx = (uint64_t)current * (params.n + 1) + node;
          float tau = dequantize_tau(d_tau[tau_idx], params.log_tau_min, params.log_tau_step);
          float score = powf(tau, params.alpha) * powf(eta, params.beta);
          word_running += score;
          if (score > 0.0f && threshold < word_running) {
            lane_selected_node = node;
            break;
          }
          scan_mask &= (scan_mask - 1ull);
        }
      }

      unsigned int lane_selected_mask = __ballot_sync(0xFFFFFFFFu, lane_selected_node > 0);
      if (lane_selected_mask != 0u) {
        int chosen_lane = __ffs((int)lane_selected_mask) - 1;
        return __shfl_sync(0xFFFFFFFFu, lane_selected_node, chosen_lane);
      }
    }
    running += chunk_sum;
    running = __shfl_sync(0xFFFFFFFFu, running, 0);
  }

  return -1;
}

__global__ void kernel_construct_solutions_v6(const float2 *d_coords,
                                              const uint8_t *d_tau,
                                              const int *d_candidate_idx,
                                              const float *d_eta_beta,
                                              int *d_routes,
                                              int *d_route_lengths,
                                              uint64_t *d_visited_l1,
                                              uint64_t *d_visited_l2,
                                              unsigned int *d_rng_states,
                                              CudaV6AntSummary *d_ant_summary,
                                              CudaV6IterStats *d_iter_stats,
                                              CudaV6Params params) {
  int lane = threadIdx.x & (CUDA_V6_WARP_SIZE - 1);
  int ant = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_V6_WARP_SIZE;
  
  if (ant >= params.m) return;

  uint64_t *l1_row = d_visited_l1 + ant * (params.visited_row_stride / 8);
  uint64_t *l2_row = d_visited_l2 + ant * params.visited_l2_words;
  
  unsigned int rng_state = 0u;
  int remaining = 0;
  float total_cost = 0.0f;
  int global_step = 0; 

  if (lane == 0) {
    rng_state = d_rng_states[ant];
    remaining = params.n;
  }

  for (int vehicle = 0; vehicle < params.K; ++vehicle) {
    int current = 0;
    int route_load = 0;
    int route_len = 1;

    if (lane != 0) {
      current = 0;
      route_load = 0;
    }

    while (1) {
      int current_b = __shfl_sync(0xFFFFFFFFu, current, 0);
      int route_load_b = __shfl_sync(0xFFFFFFFFu, route_load, 0);
      int remaining_b = __shfl_sync(0xFFFFFFFFu, remaining, 0);
      
      if (remaining_b <= 0 || route_load_b >= params.cap) break;

      float score_cand = 0.0f;
      int move = select_from_candidates_v6(current_b, d_tau, d_candidate_idx, d_eta_beta, l1_row, &rng_state, params, &score_cand);
      
      if (move <= 0) {
        move = select_global_fallback_v6(current_b, d_coords, d_tau, l1_row, l2_row, &rng_state, params, NULL);
      }

      int close_allowed = (remaining_b <= (params.K - vehicle - 1) * params.cap);
      float depot_score = 0.0f;
      if (close_allowed) {
        float d = calc_dist(d_coords[current_b], d_coords[0]);
        float eta = 1.0f / (d + CUDA_V6_EPS);
        depot_score = params.depot_close_weight * powf(eta, params.beta);
      }

      if (move <= 0 && depot_score <= 0.0f) {
        move = -1;
      } else if (move > 0 && depot_score > 0.0f) {
        float threshold = 0.0f;
        if (lane == 0) threshold = cuda_v6_rand01(&rng_state) * (score_cand + depot_score); 
        threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);
        if (threshold >= score_cand) {
          move = 0; 
        }
      } else if (move <= 0) {
        move = 0;
      }

      if (move <= 0) break;

      if (lane == 0) {
        global_step++;
        d_routes[global_step * params.m + ant] = move;
        route_len++;
        route_load++;
        
        int word_idx = move >> 6;
        int bit_idx = move & 63;
        l1_row[word_idx] |= (1ull << bit_idx);
        
        if (l1_row[word_idx] == ~cuda_v6_relevant_mask_for_word(word_idx, params.n)) {
          int l2_word = word_idx >> 6;
          int l2_bit = word_idx & 63;
          l2_row[l2_word] |= (1ull << l2_bit);
        }

        total_cost += calc_dist(d_coords[current], d_coords[move]);
        current = move;
        --remaining;
      }
      __syncwarp();
    }

    if (lane == 0) {
      global_step++;
      d_routes[global_step * params.m + ant] = 0; 
      total_cost += calc_dist(d_coords[current], d_coords[0]);
      d_route_lengths[ant * params.K + vehicle] = route_len + 1;
    }
    __syncwarp();
  }

  if (lane == 0) {
    d_rng_states[ant] = rng_state;
    d_ant_summary[ant].feasible = (remaining == 0) ? 1 : 0;
    d_ant_summary[ant].unvisited_count = remaining;
    d_ant_summary[ant].cost = total_cost;
  }
}

__device__ void atomicMax_uint8_v6(uint8_t* address, uint8_t val) {
    unsigned int* base_address = (unsigned int*)((size_t)address & ~3);
    unsigned int shift = ((size_t)address & 3) * 8;
    unsigned int mask = 0xFF << shift;
    unsigned int assumed, old = *base_address;
    do {
        assumed = old;
        uint8_t old_val = (assumed >> shift) & 0xFF;
        if (old_val >= val) break;
        unsigned int new_val = (assumed & ~mask) | (val << shift);
        old = atomicCAS(base_address, assumed, new_val);
    } while (assumed != old);
}

__global__ void kernel_deposit_solution_v6(uint8_t *d_tau,
                                           const int *d_routes,
                                           const int *d_route_lengths,
                                           float deposit_amount,
                                           int best_ant,
                                           CudaV6Params params) {
  int veh_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (veh_idx >= params.K) return;

  int route_len = d_route_lengths[best_ant * params.K + veh_idx];
  int prev = 0;
  
  int global_step = 0;
  for (int v = 0; v < veh_idx; ++v) {
      global_step += d_route_lengths[best_ant * params.K + v] - 1; // Since steps return to depot, lengths includes it
  }
  
  for (int pos = 1; pos < route_len; ++pos) {
    global_step++;
    int curr = d_routes[global_step * params.m + best_ant];
    
    uint64_t tau_idx = (uint64_t)prev * (params.n + 1) + curr;
    uint8_t old_q = d_tau[tau_idx];
    float old_tau = dequantize_tau(old_q, params.log_tau_min, params.log_tau_step);
    float new_tau = old_tau + deposit_amount;
    uint8_t new_q = quantize_tau(new_tau, params.log_tau_min, params.log_tau_step, params.q_tau_min, params.q_tau_max);
    
    atomicMax_uint8_v6(&d_tau[tau_idx], new_q);
    
    uint64_t tau_idx_rev = (uint64_t)curr * (params.n + 1) + prev;
    atomicMax_uint8_v6(&d_tau[tau_idx_rev], new_q);
    
    prev = curr;
  }
}

__global__ void kernel_deposit_flat_solution_v6(uint8_t *d_tau,
                                                const int *d_flat_routes,
                                                const int *d_flat_lengths,
                                                float deposit_amount,
                                                CudaV6Params params) {
  int veh_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (veh_idx >= params.K) return;

  int route_len = d_flat_lengths[veh_idx];
  int prev = 0;
  
  // Stride of flat routes is params.n + 1
  int offset = veh_idx * (params.n + 1);
  
  for (int pos = 1; pos < route_len; ++pos) {
    int curr = d_flat_routes[offset + pos];
    
    uint64_t tau_idx = (uint64_t)prev * (params.n + 1) + curr;
    uint8_t old_q = d_tau[tau_idx];
    float old_tau = dequantize_tau(old_q, params.log_tau_min, params.log_tau_step);
    float new_tau = old_tau + deposit_amount;
    uint8_t new_q = quantize_tau(new_tau, params.log_tau_min, params.log_tau_step, params.q_tau_min, params.q_tau_max);
    
    atomicMax_uint8_v6(&d_tau[tau_idx], new_q);
    
    uint64_t tau_idx_rev = (uint64_t)curr * (params.n + 1) + prev;
    atomicMax_uint8_v6(&d_tau[tau_idx_rev], new_q);
    
    prev = curr;
  }
}

/* -- Launchers -- */

void launch_init_tau_v6(uint8_t *d_tau, int n, uint8_t q_tau0) {
  uint64_t total = (uint64_t)(n + 1) * (n + 1);
  int blocks = (total + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_init_tau_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_tau, total, q_tau0);
}

void launch_evaporate_tau_v6(uint8_t *d_tau, int n, uint8_t delta, uint8_t q_min) {
  uint64_t total = (uint64_t)(n + 1) * (n + 1);
  int blocks = (total + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_evaporate_tau_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_tau, total, delta, q_min);
}

void launch_build_candidate_lists_v6(const float2 *d_coords, int *d_candidate_idx, float *d_eta_beta, CudaV6Params params) {
  int total = params.n + 1;
  int blocks = (total + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_build_candidate_lists_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_coords, d_candidate_idx, d_eta_beta, params);
}

void launch_reset_ant_state_v6(int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaV6AntSummary *d_ant_summary, CudaV6Params params, unsigned int seed) {
  int blocks = (params.m + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_reset_ant_state_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_routes, d_route_lengths, d_visited_l1, d_visited_l2, d_rng_states, d_ant_summary, params, seed);
}

void launch_construct_solutions_v6(const float2 *d_coords, const uint8_t *d_tau, const int *d_candidate_idx, const float *d_eta_beta, int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaV6AntSummary *d_ant_summary, CudaV6IterStats *d_iter_stats, CudaV6Params params) {
  int warps_per_block = CUDA_V6_THREADS_PER_BLOCK / CUDA_V6_WARP_SIZE;
  int blocks = (params.m + warps_per_block - 1) / warps_per_block;
  kernel_construct_solutions_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_coords, d_tau, d_candidate_idx, d_eta_beta, d_routes, d_route_lengths, d_visited_l1, d_visited_l2, d_rng_states, d_ant_summary, d_iter_stats, params);
}

void launch_deposit_solution_v6(uint8_t *d_tau, const int *d_routes, const int *d_route_lengths, float deposit_amount, int best_ant, CudaV6Params params) {
  int blocks = (params.K + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_deposit_solution_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_tau, d_routes, d_route_lengths, deposit_amount, best_ant, params);
}

void launch_deposit_flat_solution_v6(uint8_t *d_tau, const int *d_flat_routes, const int *d_flat_lengths, float deposit_amount, CudaV6Params params) {
  int blocks = (params.K + CUDA_V6_THREADS_PER_BLOCK - 1) / CUDA_V6_THREADS_PER_BLOCK;
  kernel_deposit_flat_solution_v6<<<blocks, CUDA_V6_THREADS_PER_BLOCK>>>(d_tau, d_flat_routes, d_flat_lengths, deposit_amount, params);
}
