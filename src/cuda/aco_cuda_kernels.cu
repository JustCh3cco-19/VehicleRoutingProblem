#include "aco_cuda_kernels.h"

#include <float.h>
#include <limits.h>
#include <math.h>

/* Device counterpart of host RNG for reproducible ant sampling. */
__device__ static unsigned int xorshift32_dev(unsigned int *state) {
  unsigned int x = *state;
  if (x == 0u) {
    x = 0x9e3779b9u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

__device__ static float rand01_dev(unsigned int *state) {
  return (float)(xorshift32_dev(state) & 0x00FFFFFFu) / 16777216.0f;
}

__device__ static unsigned int ant_seed_dev(unsigned int seed, int iter,
                                            int ant) {
  return seed ^ (0x9e3779b9u * (unsigned int)(iter + 1)) ^
         (0x85ebca6bu * (unsigned int)(ant + 1));
}

__device__ __forceinline__ int pair_is_better(float cost_a, int id_a,
                                              float cost_b, int id_b) {
  if (cost_a < cost_b) {
    return 1;
  }
  if (fabsf(cost_a - cost_b) <= 1e-6f && id_a < id_b) {
    return 1;
  }
  return 0;
}

__device__ __forceinline__ int bit_is_set(const unsigned int *bits, int node) {
  int word = node >> 5;
  int bit = node & 31;
  return (bits[word] >> bit) & 1u;
}

__device__ __forceinline__ void bit_set(unsigned int *bits, int node) {
  int word = node >> 5;
  int bit = node & 31;
  bits[word] |= (1u << bit);
}

__global__ void aco_cuda_init_eta_tau_kernel(float *eta, float *tau,
                                             const float *c,
                                             int n, float tau0) {
  int size = n + 1;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = size * size;
  if (idx >= total) {
    return;
  }

  int i = idx / size;
  int j = idx % size;
  if (i == j) {
    eta[idx] = 0.0f;
    tau[idx] = 0.0f;
  } else {
    eta[idx] = 1.0f / (c[idx] + ACO_CUDA_EPS_F);
    tau[idx] = tau0;
  }
}

__global__ void aco_cuda_evaporate_tau_kernel(float *tau, int n, float rho) {
  int size = n + 1;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = size * size;
  if (idx >= total) {
    return;
  }

  int i = idx / size;
  int j = idx % size;
  if (i != j) {
    tau[idx] *= (1.0f - rho);
  }
}

__global__ void aco_cuda_construct_ants_kernel(int n, int K, int m, int iter,
                                               const float *c,
                                               const float *tau,
                                               const float *eta,
                                               const int *candidates,
                                               int candidate_count,
                                               float alpha, float beta,
                                               unsigned int seed,
                                               int *routes,
                                               int *route_lens,
                                               float *costs) {
  int ant = blockIdx.x * blockDim.x + threadIdx.x;
  if (ant >= m) {
    return;
  }
  if (n > ACO_CUDA_MAX_N) {
    costs[ant] = FLT_MAX;
    return;
  }

  enum { VIS_WORDS = (ACO_CUDA_MAX_N + 32) / 32 };
  unsigned int visited[VIS_WORDS];
  for (int i = 0; i < VIS_WORDS; ++i) {
    visited[i] = 0u;
  }

  int size = n + 1;
  int max_route_len = n + 2;
  int ant_route_base = ant * K * max_route_len;
  int ant_len_base = ant * K;

  int unvisited = n;
  unsigned int rng_state = ant_seed_dev(seed, iter, ant);

  for (int vehicle = 0; vehicle < K; ++vehicle) {
    int route_base = ant_route_base + vehicle * max_route_len;
    int len = 0;
    routes[route_base + len++] = 0;

    int current = 0;
    while (unvisited > 0 && unvisited > (K - (vehicle + 1))) {
      float denom = 0.0f;

      int cand_base = current * candidate_count;
      for (int ci = 0; ci < candidate_count; ++ci) {
        int j = candidates[cand_base + ci];
        if (j < 1) {
          break;
        }
        if (!bit_is_set(visited, j)) {
          int idx = current * size + j;
          float score = powf(tau[idx], alpha) * powf(eta[idx], beta);
          denom += score;
        }
      }

      int next = 0;
      if (denom <= 0.0f) {
        /* Fallback to full scan if candidate list is exhausted. */
        for (int j = 1; j <= n; ++j) {
          if (!bit_is_set(visited, j)) {
            next = j;
            break;
          }
        }
      } else {
        float r = rand01_dev(&rng_state);
        float cumulative = 0.0f;
        int fallback = 0;

        for (int ci = 0; ci < candidate_count; ++ci) {
          int j = candidates[cand_base + ci];
          if (j < 1) {
            break;
          }
          if (!bit_is_set(visited, j)) {
            int idx = current * size + j;
            float score = powf(tau[idx], alpha) * powf(eta[idx], beta);
            cumulative += score / denom;
            fallback = j;
            if (cumulative >= r) {
              next = j;
              break;
            }
          }
        }

        if (next == 0) {
          if (fallback != 0) {
            next = fallback;
          } else {
            for (int j = 1; j <= n; ++j) {
              if (!bit_is_set(visited, j)) {
                next = j;
                break;
              }
            }
          }
        }
      }

      routes[route_base + len++] = next;
      bit_set(visited, next);
      --unvisited;
      current = next;
    }

    routes[route_base + len++] = 0;
    route_lens[ant_len_base + vehicle] = len;
  }

  if (unvisited > 0) {
    int last_vehicle = K - 1;
    int route_base = ant_route_base + last_vehicle * max_route_len;
    int len_idx = ant_len_base + last_vehicle;
    int len = route_lens[len_idx];
    if (len > 0) {
      --len;
    }
    for (int j = 1; j <= n; ++j) {
      if (!bit_is_set(visited, j)) {
        routes[route_base + len++] = j;
        bit_set(visited, j);
      }
    }
    routes[route_base + len++] = 0;
    route_lens[len_idx] = len;
  }

  float total_cost = 0.0f;
  for (int vehicle = 0; vehicle < K; ++vehicle) {
    int route_base = ant_route_base + vehicle * max_route_len;
    int len = route_lens[ant_len_base + vehicle];
    for (int t = 0; t + 1 < len; ++t) {
      int u = routes[route_base + t];
      int v = routes[route_base + t + 1];
      total_cost += c[u * size + v];
    }
  }
  costs[ant] = total_cost;
}

__global__ void aco_cuda_reduce_costs_stage_kernel(const float *costs,
                                                   int m,
                                                   float *out_costs,
                                                   int *out_ids) {
  __shared__ float s_cost[ACO_CUDA_BLOCK_SIZE];
  __shared__ int s_id[ACO_CUDA_BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  float cost = FLT_MAX;
  int id = INT_MAX;
  if (gid < m) {
    cost = costs[gid];
    id = gid;
  }

  s_cost[tid] = cost;
  s_id[tid] = id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float c1 = s_cost[tid + stride];
      int i1 = s_id[tid + stride];
      float c0 = s_cost[tid];
      int i0 = s_id[tid];
      if (pair_is_better(c1, i1, c0, i0)) {
        s_cost[tid] = c1;
        s_id[tid] = i1;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_costs[blockIdx.x] = s_cost[0];
    out_ids[blockIdx.x] = s_id[0];
  }
}

__global__ void aco_cuda_reduce_pairs_stage_kernel(const float *in_costs,
                                                   const int *in_ids,
                                                   int n_items,
                                                   float *out_costs,
                                                   int *out_ids) {
  __shared__ float s_cost[ACO_CUDA_BLOCK_SIZE];
  __shared__ int s_id[ACO_CUDA_BLOCK_SIZE];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  float cost = FLT_MAX;
  int id = INT_MAX;
  if (gid < n_items) {
    cost = in_costs[gid];
    id = in_ids[gid];
  }

  s_cost[tid] = cost;
  s_id[tid] = id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float c1 = s_cost[tid + stride];
      int i1 = s_id[tid + stride];
      float c0 = s_cost[tid];
      int i0 = s_id[tid];
      if (pair_is_better(c1, i1, c0, i0)) {
        s_cost[tid] = c1;
        s_id[tid] = i1;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_costs[blockIdx.x] = s_cost[0];
    out_ids[blockIdx.x] = s_id[0];
  }
}

__global__ void aco_cuda_deposit_best_tau_kernel(float *tau, int n,
                                                 int K, int max_route_len,
                                                 const int *routes,
                                                 const int *route_lens,
                                                 int best_ant,
                                                 float deposit) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (best_ant < 0 || deposit <= 0.0f) {
    return;
  }

  int size = n + 1;
  int ant_route_base = best_ant * K * max_route_len;
  int ant_len_base = best_ant * K;

  for (int vehicle = 0; vehicle < K; ++vehicle) {
    int route_base = ant_route_base + vehicle * max_route_len;
    int len = route_lens[ant_len_base + vehicle];
    for (int t = 0; t + 1 < len; ++t) {
      int u = routes[route_base + t];
      int v = routes[route_base + t + 1];
      tau[u * size + v] += deposit;
      tau[v * size + u] += deposit;
    }
  }
}

__global__ void aco_cuda_extract_best_pair_kernel(const float *best_cost_in,
                                                  const int *best_id_in,
                                                  int *best_ant_out,
                                                  float *best_cost_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *best_ant_out = best_id_in[0];
  *best_cost_out = best_cost_in[0];
}

__global__ void aco_cuda_deposit_best_tau_from_device_kernel(
    float *tau, int n, int K, int max_route_len,
    const int *routes, const int *route_lens,
    const int *best_ant_dev, const float *best_cost_dev,
    float Q) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int best_ant = best_ant_dev[0];
  float best_cost = best_cost_dev[0];
  if (best_ant < 0 || best_cost <= 0.0f || !isfinite(best_cost)) {
    return;
  }
  float deposit = Q / best_cost;
  if (deposit <= 0.0f || !isfinite(deposit)) {
    return;
  }

  int size = n + 1;
  int ant_route_base = best_ant * K * max_route_len;
  int ant_len_base = best_ant * K;

  for (int vehicle = 0; vehicle < K; ++vehicle) {
    int route_base = ant_route_base + vehicle * max_route_len;
    int len = route_lens[ant_len_base + vehicle];
    for (int t = 0; t + 1 < len; ++t) {
      int u = routes[route_base + t];
      int v = routes[route_base + t + 1];
      tau[u * size + v] += deposit;
      tau[v * size + u] += deposit;
    }
  }
}
