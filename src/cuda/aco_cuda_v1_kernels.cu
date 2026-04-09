#include "aco_cuda_v1_kernels.h"

#include <math.h>

__device__ static float cuda_v1_rand01(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return (float)(*state) / 4294967295.0f;
}

__device__ static int cuda_v1_visited_get(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >>
                ((unsigned int)node & 63u)) &
               1u);
}

__device__ static void cuda_v1_visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |=
      ((uint64_t)1u << ((unsigned int)node & 63u));
}

__device__ static int cuda_v1_select_from_candidates(
    int current, const float *tau, const int *candidate_idx,
    const float *eta_beta, uint64_t *visited, unsigned int *rng_state,
    CudaV1Params params) {
  int side = params.n + 1;
  int base = current * params.cand_k;
  float total = 0.0f;
  float weights[CUDA_V1_MAX_CANDIDATES];
  int t;

  for (t = 0; t < params.cand_k; ++t) {
    int node = candidate_idx[base + t];
    float weight = 0.0f;
    if (node > 0 && !cuda_v1_visited_get(visited, node)) {
      float tau_term = powf(tau[current * side + node], params.alpha);
      weight = tau_term * eta_beta[base + t];
    }
    weights[t] = weight;
    total += weight;
  }

  if (total <= 0.0f) {
    return -1;
  }

  {
    float threshold = cuda_v1_rand01(rng_state) * total;
    float cumulative = 0.0f;
    for (t = 0; t < params.cand_k; ++t) {
      if (weights[t] <= 0.0f) {
        continue;
      }
      cumulative += weights[t];
      if (cumulative >= threshold) {
        return candidate_idx[base + t];
      }
    }
  }

  for (t = params.cand_k - 1; t >= 0; --t) {
    if (weights[t] > 0.0f) {
      return candidate_idx[base + t];
    }
  }

  return -1;
}

__device__ static int cuda_v1_select_global_fallback(
    int current, const float *costs, const float *tau, uint64_t *visited,
    unsigned int *rng_state, CudaV1Params params) {
  int side = params.n + 1;
  float total = 0.0f;
  int node;

  for (node = 1; node <= params.n; ++node) {
    if (!cuda_v1_visited_get(visited, node)) {
      float eta = 1.0f / (costs[current * side + node] + CUDA_V1_EPS);
      total += powf(tau[current * side + node], params.alpha) *
               powf(eta, params.beta);
    }
  }

  if (total <= 0.0f) {
    return -1;
  }

  {
    float threshold = cuda_v1_rand01(rng_state) * total;
    float cumulative = 0.0f;
    for (node = 1; node <= params.n; ++node) {
      if (!cuda_v1_visited_get(visited, node)) {
        float eta = 1.0f / (costs[current * side + node] + CUDA_V1_EPS);
        cumulative += powf(tau[current * side + node], params.alpha) *
                      powf(eta, params.beta);
        if (cumulative >= threshold) {
          return node;
        }
      }
    }
  }

  return -1;
}

__device__ static int cuda_v1_select_next_customer(
    int current, const float *costs, const float *tau, const int *candidate_idx,
    const float *eta_beta, uint64_t *visited, unsigned int *rng_state,
    CudaV1Params params) {
  int next = cuda_v1_select_from_candidates(current, tau, candidate_idx,
                                            eta_beta, visited, rng_state,
                                            params);
  if (next > 0) {
    return next;
  }

  return cuda_v1_select_global_fallback(current, costs, tau, visited, rng_state,
                                        params);
}

__global__ void kernel_build_candidate_lists_v1(const float *costs,
                                                int *candidate_idx,
                                                float *eta_beta, int n,
                                                int cand_k, float beta) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int best_nodes[CUDA_V1_MAX_CANDIDATES];
  float best_costs[CUDA_V1_MAX_CANDIDATES];
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
      float eta = 1.0f / (best_costs[t] + CUDA_V1_EPS);
      eta_beta[out] = powf(eta, beta);
    } else {
      eta_beta[out] = 0.0f;
    }
  }
}

__global__ void kernel_init_tau_v1(float *tau, int n, float tau0) {
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

__global__ void kernel_reset_ant_state_v1(int *routes, int *route_lengths,
                                          int *route_loads, int *curr_nodes,
                                          uint64_t *visited,
                                          unsigned int *rng_states,
                                          CudaV1AntSummary *ant_summary,
                                          CudaV1Params params,
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

__global__ void kernel_construct_solutions_v1(
    const float *costs, const float *tau, const int *candidate_idx,
    const float *eta_beta, int *routes, int *route_lengths, int *route_loads,
    int *curr_nodes, uint64_t *visited, unsigned int *rng_states,
    CudaV1AntSummary *ant_summary, CudaV1Params params) {
  int ant = blockIdx.x * blockDim.x + threadIdx.x;
  int side = params.n + 1;
  int remaining = params.n;
  float total_cost = 0.0f;
  uint64_t *visited_row;
  unsigned int rng_state;
  int vehicle;

  if (ant >= params.m) {
    return;
  }

  visited_row = visited + (size_t)ant * (size_t)params.visited_words;
  rng_state = rng_states[ant];

  for (vehicle = 0; vehicle < params.K; ++vehicle) {
    int route_base = ((ant * params.K) + vehicle) * params.route_stride;
    int route_len = 1;
    int route_load = 0;
    int current = 0;

    while (remaining > 0 && route_load < params.cap) {
      int next = cuda_v1_select_next_customer(current, costs, tau, candidate_idx,
                                              eta_beta, visited_row, &rng_state,
                                              params);
      if (next <= 0) {
        break;
      }

      routes[route_base + route_len] = next;
      ++route_len;
      ++route_load;
      cuda_v1_visited_set(visited_row, next);
      total_cost += costs[current * side + next];
      current = next;
      --remaining;
    }

    routes[route_base + route_len] = 0;
    ++route_len;
    total_cost += costs[current * side + 0];

    route_lengths[ant * params.K + vehicle] = route_len;
    route_loads[ant * params.K + vehicle] = route_load;
    curr_nodes[ant * params.K + vehicle] = 0;
  }

  rng_states[ant] = rng_state;
  ant_summary[ant].feasible = (remaining == 0) ? 1 : 0;
  ant_summary[ant].unvisited_count = remaining;
  ant_summary[ant].cost = total_cost;
}

__global__ void kernel_evaporate_tau_v1(float *tau, int n, float rho) {
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

__global__ void kernel_deposit_solution_v1(float *tau, const int *routes,
                                           const int *route_lengths, int K,
                                           int route_stride, int n,
                                           float deposit) {
  int vehicle = blockIdx.x * blockDim.x + threadIdx.x;
  int side = n + 1;
  int pos;
  int prev = 0;

  if (vehicle >= K) {
    return;
  }

  for (pos = 1; pos < route_lengths[vehicle]; ++pos) {
    int curr = routes[vehicle * route_stride + pos];
    atomicAdd(&tau[prev * side + curr], deposit);
    atomicAdd(&tau[curr * side + prev], deposit);
    prev = curr;
  }
}

__global__ void kernel_clamp_tau_v1(float *tau, int n, float tau_min,
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

void launch_build_candidate_lists_v1(const float *costs, int *candidate_idx,
                                     float *eta_beta, int n, int cand_k,
                                     float beta) {
  int total = n + 1;
  int blocks = (total + CUDA_V1_THREADS_PER_BLOCK - 1) /
               CUDA_V1_THREADS_PER_BLOCK;
  kernel_build_candidate_lists_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(
      costs, candidate_idx, eta_beta, n, cand_k, beta);
}

void launch_init_tau_v1(float *tau, int n, float tau0) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V1_THREADS_PER_BLOCK - 1) /
               CUDA_V1_THREADS_PER_BLOCK;
  kernel_init_tau_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(tau, n, tau0);
}

void launch_reset_ant_state_v1(int *routes, int *route_lengths,
                               int *route_loads, int *curr_nodes,
                               uint64_t *visited, unsigned int *rng_states,
                               CudaV1AntSummary *ant_summary,
                               CudaV1Params params, unsigned int seed) {
  int blocks =
      (params.m + CUDA_V1_THREADS_PER_BLOCK - 1) / CUDA_V1_THREADS_PER_BLOCK;
  kernel_reset_ant_state_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(
      routes, route_lengths, route_loads, curr_nodes, visited, rng_states,
      ant_summary, params, seed);
}

void launch_construct_solutions_v1(const float *costs, const float *tau,
                                   const int *candidate_idx,
                                   const float *eta_beta, int *routes,
                                   int *route_lengths, int *route_loads,
                                   int *curr_nodes, uint64_t *visited,
                                   unsigned int *rng_states,
                                   CudaV1AntSummary *ant_summary,
                                   CudaV1Params params) {
  int blocks =
      (params.m + CUDA_V1_THREADS_PER_BLOCK - 1) / CUDA_V1_THREADS_PER_BLOCK;
  kernel_construct_solutions_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(
      costs, tau, candidate_idx, eta_beta, routes, route_lengths, route_loads,
      curr_nodes, visited, rng_states, ant_summary, params);
}

void launch_evaporate_tau_v1(float *tau, int n, float rho) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V1_THREADS_PER_BLOCK - 1) /
               CUDA_V1_THREADS_PER_BLOCK;
  kernel_evaporate_tau_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(tau, n, rho);
}

void launch_deposit_solution_v1(float *tau, const int *routes,
                                const int *route_lengths, int K,
                                int route_stride, int n, float deposit) {
  int blocks =
      (K + CUDA_V1_THREADS_PER_BLOCK - 1) / CUDA_V1_THREADS_PER_BLOCK;
  kernel_deposit_solution_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(
      tau, routes, route_lengths, K, route_stride, n, deposit);
}

void launch_clamp_tau_v1(float *tau, int n, float tau_min, float tau_max) {
  int side = n + 1;
  int total = side * side;
  int blocks = (total + CUDA_V1_THREADS_PER_BLOCK - 1) /
               CUDA_V1_THREADS_PER_BLOCK;
  kernel_clamp_tau_v1<<<blocks, CUDA_V1_THREADS_PER_BLOCK>>>(tau, n, tau_min,
                                                             tau_max);
}
