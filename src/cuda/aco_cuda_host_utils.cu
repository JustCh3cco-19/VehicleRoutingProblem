#include "aco_cuda_host_utils.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void aco_cuda_buffers_init(AcoCudaBuffers *b) {
  /* Zero-init enables safe partial cleanup on allocation failures. */
  memset(b, 0, sizeof(*b));
}

cudaError_t aco_cuda_buffers_alloc(AcoCudaBuffers *b, int n, int K, int m,
                                   int candidate_count) {
  b->n = n;
  b->K = K;
  b->m = m;
  b->size = n + 1;
  b->matrix_elems = b->size * b->size;
  b->max_route_len = n + 2;

  if (candidate_count < 1) {
    candidate_count = 1;
  }
  if (candidate_count > ACO_CUDA_MAX_CANDIDATES) {
    candidate_count = ACO_CUDA_MAX_CANDIDATES;
  }
  if (candidate_count > n) {
    candidate_count = n;
  }
  b->candidate_count = candidate_count;

  b->reduce_capacity = (m + ACO_CUDA_BLOCK_SIZE - 1) / ACO_CUDA_BLOCK_SIZE;
  if (b->reduce_capacity < 1) {
    b->reduce_capacity = 1;
  }

  b->matrix_bytes = (size_t)b->matrix_elems * sizeof(float);
  b->routes_bytes = (size_t)m * (size_t)K * (size_t)b->max_route_len * sizeof(int);
  b->lens_bytes = (size_t)m * (size_t)K * sizeof(int);
  b->costs_bytes = (size_t)m * sizeof(float);
  b->candidates_bytes = (size_t)b->size * (size_t)b->candidate_count * sizeof(int);
  b->best_routes_bytes = (size_t)K * (size_t)b->max_route_len * sizeof(int);
  b->best_lens_bytes = (size_t)K * sizeof(int);
  b->reduce_bytes = (size_t)b->reduce_capacity * sizeof(float);

  cudaError_t err;

  err = cudaHostAlloc((void **)&b->h_c, b->matrix_bytes, cudaHostAllocDefault);
  if (err != cudaSuccess) return err;

  b->h_candidates = (int *)malloc(b->candidates_bytes);
  if (!b->h_candidates) return cudaErrorMemoryAllocation;

  b->h_best_routes = (int *)malloc(b->best_routes_bytes);
  if (!b->h_best_routes) return cudaErrorMemoryAllocation;
  b->h_best_lens = (int *)malloc(b->best_lens_bytes);
  if (!b->h_best_lens) return cudaErrorMemoryAllocation;

  err = cudaMalloc((void **)&b->d_c, b->matrix_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_tau, b->matrix_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_eta, b->matrix_bytes);
  if (err != cudaSuccess) return err;

  err = cudaMalloc((void **)&b->d_routes, b->routes_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_lens, b->lens_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_costs, b->costs_bytes);
  if (err != cudaSuccess) return err;

  err = cudaMalloc((void **)&b->d_candidates, b->candidates_bytes);
  if (err != cudaSuccess) return err;

  err = cudaMalloc((void **)&b->d_reduce_costs_a, b->reduce_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_reduce_costs_b, b->reduce_bytes);
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_reduce_ids_a,
                   (size_t)b->reduce_capacity * sizeof(int));
  if (err != cudaSuccess) return err;
  err = cudaMalloc((void **)&b->d_reduce_ids_b,
                   (size_t)b->reduce_capacity * sizeof(int));
  if (err != cudaSuccess) return err;

  err = cudaStreamCreateWithFlags(&b->stream_compute, cudaStreamNonBlocking);
  if (err != cudaSuccess) return err;

  return cudaSuccess;
}

void aco_cuda_buffers_free(AcoCudaBuffers *b) {
  if (b->stream_compute) {
    cudaStreamDestroy(b->stream_compute);
  }

  if (b->d_reduce_ids_b) {
    cudaFree(b->d_reduce_ids_b);
  }
  if (b->d_reduce_ids_a) {
    cudaFree(b->d_reduce_ids_a);
  }
  if (b->d_reduce_costs_b) {
    cudaFree(b->d_reduce_costs_b);
  }
  if (b->d_reduce_costs_a) {
    cudaFree(b->d_reduce_costs_a);
  }

  if (b->d_candidates) {
    cudaFree(b->d_candidates);
  }

  if (b->d_costs) {
    cudaFree(b->d_costs);
  }
  if (b->d_lens) {
    cudaFree(b->d_lens);
  }
  if (b->d_routes) {
    cudaFree(b->d_routes);
  }

  if (b->d_eta) {
    cudaFree(b->d_eta);
  }
  if (b->d_tau) {
    cudaFree(b->d_tau);
  }
  if (b->d_c) {
    cudaFree(b->d_c);
  }

  free(b->h_best_lens);
  free(b->h_best_routes);
  free(b->h_candidates);

  if (b->h_c) {
    cudaFreeHost(b->h_c);
  }

  aco_cuda_buffers_init(b);
}

void aco_cuda_copy_cost_to_flat(double **c, int n, float *flat) {
  int size = n + 1;
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      flat[i * size + j] = (float)c[i][j];
    }
  }
}

void aco_cuda_build_candidates(double **c, int n, int candidate_count,
                               int *flat_candidates) {
  int size = n + 1;
  memset(flat_candidates, 0xFF,
         (size_t)size * (size_t)candidate_count * sizeof(int));

  int *chosen = (int *)calloc((size_t)(n + 1), sizeof(int));
  if (!chosen) {
    return;
  }

  for (int i = 0; i < size; ++i) {
    memset(chosen, 0, (size_t)(n + 1) * sizeof(int));
    int base = i * candidate_count;

    for (int slot = 0; slot < candidate_count; ++slot) {
      int best_j = -1;
      double best_cost = DBL_MAX;

      for (int j = 1; j <= n; ++j) {
        if (j == i || chosen[j]) {
          continue;
        }
        double cost = c[i][j];
        if (cost < best_cost) {
          best_cost = cost;
          best_j = j;
        }
      }

      if (best_j < 0) {
        break;
      }

      flat_candidates[base + slot] = best_j;
      chosen[best_j] = 1;
    }
  }

  free(chosen);
}

int aco_cuda_better_candidate(float cost_a, int ant_a,
                              float cost_b, int ant_b) {
  if (cost_a < cost_b) {
    return 1;
  }
  if (fabsf(cost_a - cost_b) <= 1e-6f && ant_a < ant_b) {
    return 1;
  }
  return 0;
}

cudaError_t aco_cuda_copy_ant_from_device(AcoCudaBuffers *b,
                                          int ant,
                                          Solution *dst) {
  if (ant < 0 || ant >= b->m) {
    return cudaErrorInvalidValue;
  }

  size_t route_offset = (size_t)ant * (size_t)b->K * (size_t)b->max_route_len;
  size_t len_offset = (size_t)ant * (size_t)b->K;

  cudaError_t err = cudaMemcpyAsync(b->h_best_lens,
                                    b->d_lens + len_offset,
                                    b->best_lens_bytes,
                                    cudaMemcpyDeviceToHost,
                                    b->stream_compute);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaMemcpyAsync(b->h_best_routes,
                        b->d_routes + route_offset,
                        b->best_routes_bytes,
                        cudaMemcpyDeviceToHost,
                        b->stream_compute);
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaStreamSynchronize(b->stream_compute);
  if (err != cudaSuccess) {
    return err;
  }

  solution_reset(dst);
  for (int vehicle = 0; vehicle < b->K; ++vehicle) {
    int len = b->h_best_lens[vehicle];
    if (len < 0) {
      len = 0;
    }
    if (len > b->max_route_len) {
      len = b->max_route_len;
    }

    const int *src = &b->h_best_routes[vehicle * b->max_route_len];
    Route *r = &dst->routes[vehicle];
    for (int t = 0; t < len; ++t) {
      route_append(r, src[t]);
    }
  }

  return cudaSuccess;
}
