#ifndef ACO_CUDA_HOST_UTILS_H
#define ACO_CUDA_HOST_UTILS_H

#include "aco_cuda_kernels.h"
#include "solution.h"

#include <cuda_runtime.h>
#include <stddef.h>

typedef struct {
  int n;
  int K;
  int m;
  int size;
  int matrix_elems;
  int max_route_len;
  int candidate_count;
  int reduce_capacity;

  size_t matrix_bytes;
  size_t routes_bytes;
  size_t lens_bytes;
  size_t costs_bytes;
  size_t candidates_bytes;
  size_t best_routes_bytes;
  size_t best_lens_bytes;
  size_t reduce_bytes;

  float *h_c;
  int *h_candidates;
  int *h_best_routes;
  int *h_best_lens;

  float *d_c;
  float *d_tau;
  float *d_eta;
  int *d_routes;
  int *d_lens;
  float *d_costs;
  int *d_candidates;

  float *d_reduce_costs_a;
  float *d_reduce_costs_b;
  int *d_reduce_ids_a;
  int *d_reduce_ids_b;
  int *d_iter_best_ant;
  float *d_iter_best_cost;

  cudaStream_t stream_compute;
} AcoCudaBuffers;

/* Init/alloc/free helpers for host+device resources used by CUDA backend. */
void aco_cuda_buffers_init(AcoCudaBuffers *b);
cudaError_t aco_cuda_buffers_alloc(AcoCudaBuffers *b, int n, int K, int m,
                                   int candidate_count);
void aco_cuda_buffers_free(AcoCudaBuffers *b);

/* Utility helpers for host-side flattening/comparison/copy-back. */
void aco_cuda_copy_cost_to_flat(double **c, int n, float *flat);
void aco_cuda_build_candidates(double **c, int n, int candidate_count,
                               int *flat_candidates);
int aco_cuda_better_candidate(float cost_a, int ant_a,
                              float cost_b, int ant_b);
cudaError_t aco_cuda_copy_ant_from_device(AcoCudaBuffers *b,
                                          int ant,
                                          Solution *dst);

#endif
