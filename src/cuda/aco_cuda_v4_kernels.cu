#include "aco_cuda_v4_kernels.h"

#include <math.h>

__device__ static float cuda_v4_rand01(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return (float)(*state) / 4294967295.0f;
}

__device__ static int cuda_v4_visited_get(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >>
                ((unsigned int)node & 63u)) &
               1u);
}

__device__ static void cuda_v4_visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |=
      ((uint64_t)1u << ((unsigned int)node & 63u));
}

__device__ static float cuda_v4_warp_sum(float value) {
  for (int offset = CUDA_V4_WARP_SIZE / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

__device__ static float cuda_v4_warp_prefix_sum(float value, int lane) {
  for (int offset = 1; offset < CUDA_V4_WARP_SIZE; offset <<= 1) {
    float other = __shfl_up_sync(0xFFFFFFFFu, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  return value;
}

__device__ static int cuda_v4_select_from_candidates_warp(
    int current, const float *tau, const int *candidate_idx,
    const float *eta_beta, const uint64_t *visited, unsigned int *rng_state,
    CudaV4Params params, float *total_score_out) {
  int lane = threadIdx.x & (CUDA_V4_WARP_SIZE - 1);
  int side = params.n + 1;
  int base = current * params.cand_k;
  int candidate = 0;
  float score = 0.0f;
  float total;
  float threshold = 0.0f;
  unsigned int mask;

  if (lane < params.cand_k) {
    candidate = candidate_idx[base + lane];
    if (candidate > 0 && !cuda_v4_visited_get(visited, candidate)) {
      float tau_term = powf(tau[current * side + candidate], params.alpha);
      score = tau_term * eta_beta[base + lane];
    }
  }

  total = cuda_v4_warp_sum(score);
  total = __shfl_sync(0xFFFFFFFFu, total, 0);
  if (total_score_out) {
    *total_score_out = total;
  }
  if (total <= 0.0f) {
    return -1;
  }

  if (lane == 0) {
    threshold = cuda_v4_rand01(rng_state) * total;
  }
  threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);

  {
    float prefix = cuda_v4_warp_prefix_sum(score, lane);
    int selected = (score > 0.0f && prefix >= threshold &&
                    (prefix - score) < threshold);
    mask = __ballot_sync(0xFFFFFFFFu, selected);
  }

  if (mask != 0u) {
    int chosen_lane = __ffs((int)mask) - 1;
    return __shfl_sync(0xFFFFFFFFu, candidate, chosen_lane);
  }

  return -1;
}

__device__ static int cuda_v4_select_global_fallback_warp(
    int current, const float *costs, const float *tau, const uint64_t *visited,
    unsigned int *rng_state, CudaV4Params params, float *total_score_out) {
  int lane = threadIdx.x & (CUDA_V4_WARP_SIZE - 1);
  int side = params.n + 1;
  int chunks = (params.n + CUDA_V4_WARP_SIZE - 1) / CUDA_V4_WARP_SIZE;
  float local_total = 0.0f;
  float total;
  float threshold = 0.0f;
  float running = 0.0f;

  for (int chunk = 0; chunk < chunks; ++chunk) {
    int node = 1 + lane + chunk * CUDA_V4_WARP_SIZE;
    if (node <= params.n && !cuda_v4_visited_get(visited, node)) {
      float eta = 1.0f / (costs[current * side + node] + CUDA_V4_EPS);
      local_total +=
          powf(tau[current * side + node], params.alpha) * powf(eta, params.beta);
    }
  }

  total = cuda_v4_warp_sum(local_total);
  total = __shfl_sync(0xFFFFFFFFu, total, 0);
  if (total_score_out) {
    *total_score_out = total;
  }
  if (total <= 0.0f) {
    return -1;
  }

  if (lane == 0) {
    threshold = cuda_v4_rand01(rng_state) * total;
  }
  threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);

  for (int chunk = 0; chunk < chunks; ++chunk) {
    int node = 1 + lane + chunk * CUDA_V4_WARP_SIZE;
    float score = 0.0f;
    float chunk_sum;

    if (node <= params.n && !cuda_v4_visited_get(visited, node)) {
      float eta = 1.0f / (costs[current * side + node] + CUDA_V4_EPS);
      score =
          powf(tau[current * side + node], params.alpha) * powf(eta, params.beta);
    }

    chunk_sum = cuda_v4_warp_sum(score);
    chunk_sum = __shfl_sync(0xFFFFFFFFu, chunk_sum, 0);

    if (chunk_sum > 0.0f && threshold < (running + chunk_sum)) {
      float prefix = cuda_v4_warp_prefix_sum(score, lane);
      int selected = (score > 0.0f && (running + prefix) >= threshold &&
                      (running + prefix - score) < threshold);
      unsigned int mask = __ballot_sync(0xFFFFFFFFu, selected);
      if (mask != 0u) {
        int chosen_lane = __ffs((int)mask) - 1;
        return __shfl_sync(0xFFFFFFFFu, node, chosen_lane);
      }
    }

    running += chunk_sum;
    running = __shfl_sync(0xFFFFFFFFu, running, 0);
  }

  return -1;
}

__device__ static int cuda_v4_select_next_customer_warp(
    int current, const float *costs, const float *tau, const int *candidate_idx,
    const float *eta_beta, const uint64_t *visited, unsigned int *rng_state,
    CudaV4Params params, float *total_score_out, int *used_fallback_out) {
  float candidate_total = 0.0f;
  int next = cuda_v4_select_from_candidates_warp(current, tau, candidate_idx,
                                                 eta_beta, visited, rng_state,
                                                 params, &candidate_total);
  if (next > 0) {
    if (total_score_out) {
      *total_score_out = candidate_total;
    }
    if (used_fallback_out) {
      *used_fallback_out = 0;
    }
    return next;
  }

  if (used_fallback_out) {
    *used_fallback_out = 1;
  }
  return cuda_v4_select_global_fallback_warp(current, costs, tau, visited,
                                             rng_state, params, total_score_out);
}

__device__ static float cuda_v4_score_depot_move(int current, const float *costs,
                                                 CudaV4Params params) {
  int side = params.n + 1;
  float eta = 1.0f / (costs[current * side + 0] + CUDA_V4_EPS);
  return params.depot_close_weight * powf(eta, params.beta);
}

__device__ static int cuda_v4_select_next_move_warp(
    int current, int close_allowed, const float *costs, const float *tau,
    const int *candidate_idx, const float *eta_beta, const uint64_t *visited,
    unsigned int *rng_state, CudaV4Params params, int *move_kind_out) {
  int lane = threadIdx.x & (CUDA_V4_WARP_SIZE - 1);
  float customer_total = 0.0f;
  int used_fallback = 0;
  int customer = cuda_v4_select_next_customer_warp(
      current, costs, tau, candidate_idx, eta_beta, visited, rng_state, params,
      &customer_total, &used_fallback);
  float depot_score = 0.0f;
  float threshold = 0.0f;

  if (close_allowed) {
    depot_score = cuda_v4_score_depot_move(current, costs, params);
  }
  depot_score = __shfl_sync(0xFFFFFFFFu, depot_score, 0);
  customer_total = __shfl_sync(0xFFFFFFFFu, customer_total, 0);
  customer = __shfl_sync(0xFFFFFFFFu, customer, 0);

  if (customer_total <= 0.0f && depot_score <= 0.0f) {
    if (move_kind_out) {
      *move_kind_out = -1;
    }
    return -1;
  }
  if (customer_total <= 0.0f) {
    if (move_kind_out) {
      *move_kind_out = 2;
    }
    return 0;
  }
  if (depot_score <= 0.0f) {
    if (move_kind_out) {
      *move_kind_out = used_fallback ? 1 : 0;
    }
    return customer;
  }

  if (lane == 0) {
    threshold = cuda_v4_rand01(rng_state) * (customer_total + depot_score);
  }
  threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);
  if (threshold < customer_total) {
    if (move_kind_out) {
      *move_kind_out = used_fallback ? 1 : 0;
    }
    return customer;
  }
  if (move_kind_out) {
    *move_kind_out = 2;
  }
  return 0;
}

__global__ void kernel_build_candidate_lists_v4(const float *costs,
                                                int *candidate_idx,
                                                float *eta_beta, int n,
                                                int cand_k, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int best_nodes[CUDA_V4_MAX_CANDIDATES];
  float best_costs[CUDA_V4_MAX_CANDIDATES];
  int t;
  int node;

  if (i > n) {
    return;
  }

  for (t = 0; t < cand_k; ++t) {
    best_nodes[t] = 0;
    best_costs[t] = INFINITY;
  }

  for (node = 1; node <= n; ++node) {
    int pos = -1;
    float d;
    if (node == i) {
      continue;
    }

    d = costs[i * side + node];
    for (t = 0; t < cand_k; ++t) {
      if (d < best_costs[t]) {
        pos = t;
        break;
      }
    }
    if (pos < 0) {
      continue;
    }
    for (t = cand_k - 1; t > pos; --t) {
      best_costs[t] = best_costs[t - 1];
      best_nodes[t] = best_nodes[t - 1];
    }
    best_costs[pos] = d;
    best_nodes[pos] = node;
  }

  for (t = 0; t < cand_k; ++t) {
    int out = i * cand_k + t;
    candidate_idx[out] = best_nodes[t];
    if (best_nodes[t] > 0) {
      float eta = 1.0f / (best_costs[t] + CUDA_V4_EPS);
      eta_beta[out] = powf(eta, beta);
    } else {
      eta_beta[out] = 0.0f;
    }
  }
}

__global__ void kernel_init_tau_v4(float *tau, int n, float tau0) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int total = side * side;

  if (idx >= total) {
    return;
  }

  {
    int i = idx / side;
    int j = idx % side;
    tau[idx] = (i == j) ? 0.0f : tau0;
  }
}

__global__ void kernel_reset_ant_state_v4(int *routes, int *route_lengths,
                                          int *route_loads, int *curr_nodes,
                                          uint64_t *visited,
                                          unsigned int *rng_states,
                                          CudaV4AntSummary *ant_summary,
                                          CudaV4Params params,
                                          unsigned int seed) {
  int ant = blockIdx.x * blockDim.x + threadIdx.x;
  int v;
  int w;

  if (ant >= params.m) {
    return;
  }

  for (w = 0; w < params.visited_words; ++w) {
    visited[ant * params.visited_words + w] = 0u;
  }

  for (v = 0; v < params.K; ++v) {
    int route_base = ((ant * params.K) + v) * params.route_stride;
    routes[route_base] = 0;
    route_lengths[ant * params.K + v] = 1;
    route_loads[ant * params.K + v] = 0;
    curr_nodes[ant * params.K + v] = 0;
  }

  rng_states[ant] = seed ^ (1664525u * (unsigned int)(ant + 1));
  ant_summary[ant].feasible = 0;
  ant_summary[ant].unvisited_count = params.n;
  ant_summary[ant].cost = 0.0f;
}

__global__ void kernel_construct_solutions_v4(
    const float *costs, const float *tau, const int *candidate_idx,
    const float *eta_beta, int *routes, int *route_lengths, int *route_loads,
    int *curr_nodes, uint64_t *visited, unsigned int *rng_states,
    CudaV4AntSummary *ant_summary, CudaV4IterStats *iter_stats,
    CudaV4Params params) {
  int lane = threadIdx.x & (CUDA_V4_WARP_SIZE - 1);
  int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  int ant = global_thread / CUDA_V4_WARP_SIZE;
  int side = params.n + 1;
  uint64_t *visited_row;
  unsigned int rng_state = 0u;
  int remaining = 0;
  float total_cost = 0.0f;

  if (ant >= params.m) {
    return;
  }

  visited_row = visited + (size_t)ant * (size_t)params.visited_words;
  if (lane == 0) {
    rng_state = rng_states[ant];
    remaining = params.n;
    total_cost = 0.0f;
  }

  for (int vehicle = 0; vehicle < params.K; ++vehicle) {
    int route_base = ((ant * params.K) + vehicle) * params.route_stride;
    int current = 0;
    int route_len = 1;
    int route_load = 0;
    int remaining_vehicles = params.K - vehicle - 1;
    int future_capacity = remaining_vehicles * params.cap;

    if (lane != 0) {
      current = 0;
      route_len = 0;
      route_load = 0;
    }

    while (1) {
      int current_b = __shfl_sync(0xFFFFFFFFu, current, 0);
      int route_load_b = __shfl_sync(0xFFFFFFFFu, route_load, 0);
      int remaining_b = __shfl_sync(0xFFFFFFFFu, remaining, 0);
      int close_allowed_b;
      int move_kind = -1;
      int move;

      if (remaining_b <= 0 || route_load_b >= params.cap) {
        break;
      }

      close_allowed_b = (remaining_b <= future_capacity) ? 1 : 0;
      move = cuda_v4_select_next_move_warp(current_b, close_allowed_b, costs,
                                           tau, candidate_idx, eta_beta,
                                           visited_row, &rng_state, params,
                                           &move_kind);
      move = __shfl_sync(0xFFFFFFFFu, move, 0);
      move_kind = __shfl_sync(0xFFFFFFFFu, move_kind, 0);
      if (move <= 0) {
        if (lane == 0 && move == 0 && iter_stats) {
          atomicAdd(&iter_stats->depot_close_moves, 1ULL);
        }
        break;
      }

      if (lane == 0) {
        if (iter_stats) {
          atomicAdd(&iter_stats->customer_moves, 1ULL);
          if (move_kind == 0) {
            atomicAdd(&iter_stats->candidate_moves, 1ULL);
          } else if (move_kind == 1) {
            atomicAdd(&iter_stats->fallback_moves, 1ULL);
          }
        }
        routes[route_base + route_len] = move;
        ++route_len;
        ++route_load;
        cuda_v4_visited_set(visited_row, move);
        total_cost += costs[current * side + move];
        current = move;
        --remaining;
      }
      __syncwarp();
    }

    if (lane == 0) {
      if (route_load > 0 && iter_stats) {
        atomicAdd(&iter_stats->nonempty_routes, 1ULL);
      }
      routes[route_base + route_len] = 0;
      ++route_len;
      total_cost += costs[current * side + 0];
      route_lengths[ant * params.K + vehicle] = route_len;
      route_loads[ant * params.K + vehicle] = route_load;
      curr_nodes[ant * params.K + vehicle] = 0;
    }
    __syncwarp();
  }

  if (lane == 0) {
    rng_states[ant] = rng_state;
    ant_summary[ant].feasible = (remaining == 0) ? 1 : 0;
    ant_summary[ant].unvisited_count = remaining;
    ant_summary[ant].cost = total_cost;
  }
}

__global__ void kernel_evaporate_tau_v4(float *tau, int n, float rho) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int total = side * side;

  if (idx >= total) {
    return;
  }

  {
    int i = idx / side;
    int j = idx % side;
    if (i == j) {
      tau[idx] = 0.0f;
    } else {
      tau[idx] *= (1.0f - rho);
    }
  }
}

__global__ void kernel_deposit_solution_v4(float *tau, const int *routes,
                                           const int *route_lengths, int K,
                                           int route_stride, int n,
                                           float deposit) {
  int vehicle = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int prev = 0;

  if (vehicle >= K) {
    return;
  }

  for (int pos = 1; pos < route_lengths[vehicle]; ++pos) {
    int curr = routes[vehicle * route_stride + pos];
    atomicAdd(&tau[prev * side + curr], deposit);
    atomicAdd(&tau[curr * side + prev], deposit);
    prev = curr;
  }
}

__global__ void kernel_clamp_tau_v4(float *tau, int n, float tau_min,
                                    float tau_max) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int total = side * side;

  if (idx >= total) {
    return;
  }

  {
    int i = idx / side;
    int j = idx % side;
    if (i == j) {
      tau[idx] = 0.0f;
    } else if (tau[idx] < tau_min) {
      tau[idx] = tau_min;
    } else if (tau[idx] > tau_max) {
      tau[idx] = tau_max;
    }
  }
}

__global__ void kernel_recenter_tau_v4(float *tau, int n, float tau0) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int total = side * side;

  if (idx >= total) {
    return;
  }

  {
    int i = idx / side;
    int j = idx % side;
    if (i == j) {
      tau[idx] = 0.0f;
    } else {
      tau[idx] = 0.5f * tau[idx] + 0.5f * tau0;
    }
  }
}

void launch_build_candidate_lists_v4(const float *costs, int *candidate_idx,
                                     float *eta_beta, int n, int cand_k,
                                     float beta) {
  int total = n + 1;
  int blocks = (total + CUDA_V4_THREADS_PER_BLOCK - 1) /
               CUDA_V4_THREADS_PER_BLOCK;
  kernel_build_candidate_lists_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(
      costs, candidate_idx, eta_beta, n, cand_k, beta);
}

void launch_init_tau_v4(float *tau, int n, float tau0) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V4_THREADS_PER_BLOCK - 1) /
               CUDA_V4_THREADS_PER_BLOCK;
  kernel_init_tau_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(tau, n, tau0);
}

void launch_reset_ant_state_v4(int *routes, int *route_lengths,
                               int *route_loads, int *curr_nodes,
                               uint64_t *visited, unsigned int *rng_states,
                               CudaV4AntSummary *ant_summary,
                               CudaV4Params params, unsigned int seed) {
  int blocks =
      (params.m + CUDA_V4_THREADS_PER_BLOCK - 1) / CUDA_V4_THREADS_PER_BLOCK;
  kernel_reset_ant_state_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(
      routes, route_lengths, route_loads, curr_nodes, visited, rng_states,
      ant_summary, params, seed);
}

void launch_construct_solutions_v4(const float *costs, const float *tau,
                                   const int *candidate_idx,
                                   const float *eta_beta, int *routes,
                                   int *route_lengths, int *route_loads,
                                   int *curr_nodes, uint64_t *visited,
                                   unsigned int *rng_states,
                                   CudaV4AntSummary *ant_summary,
                                   CudaV4IterStats *iter_stats,
                                   CudaV4Params params) {
  int warps_per_block = CUDA_V4_THREADS_PER_BLOCK / CUDA_V4_WARP_SIZE;
  int blocks = (params.m + warps_per_block - 1) / warps_per_block;
  kernel_construct_solutions_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(
      costs, tau, candidate_idx, eta_beta, routes, route_lengths, route_loads,
      curr_nodes, visited, rng_states, ant_summary, iter_stats, params);
}

void launch_evaporate_tau_v4(float *tau, int n, float rho) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V4_THREADS_PER_BLOCK - 1) /
               CUDA_V4_THREADS_PER_BLOCK;
  kernel_evaporate_tau_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(tau, n, rho);
}

void launch_deposit_solution_v4(float *tau, const int *routes,
                                const int *route_lengths, int K,
                                int route_stride, int n, float deposit) {
  int blocks =
      (K + CUDA_V4_THREADS_PER_BLOCK - 1) / CUDA_V4_THREADS_PER_BLOCK;
  kernel_deposit_solution_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(
      tau, routes, route_lengths, K, route_stride, n, deposit);
}

void launch_clamp_tau_v4(float *tau, int n, float tau_min, float tau_max) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V4_THREADS_PER_BLOCK - 1) /
               CUDA_V4_THREADS_PER_BLOCK;
  kernel_clamp_tau_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(tau, n, tau_min,
                                                             tau_max);
}

void launch_recenter_tau_v4(float *tau, int n, float tau0) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V4_THREADS_PER_BLOCK - 1) /
               CUDA_V4_THREADS_PER_BLOCK;
  kernel_recenter_tau_v4<<<blocks, CUDA_V4_THREADS_PER_BLOCK>>>(tau, n, tau0);
}
