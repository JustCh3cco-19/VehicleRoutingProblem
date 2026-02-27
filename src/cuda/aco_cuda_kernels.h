#ifndef ACO_CUDA_KERNELS_H
#define ACO_CUDA_KERNELS_H

#include <cuda_runtime.h>

#define ACO_CUDA_EPS_F 1e-6f
#define ACO_CUDA_MAX_N 512
#define ACO_CUDA_MAX_CANDIDATES 32
#define ACO_CUDA_BLOCK_SIZE 256

/* Initialize eta/tau from flattened cost matrix. */
__global__ void aco_cuda_init_eta_tau_kernel(float *eta, float *tau,
                                             const float *c,
                                             int n, float tau0);

/* Apply evaporation: tau = (1-rho)*tau for off-diagonal entries. */
__global__ void aco_cuda_evaporate_tau_kernel(float *tau, int n, float rho);

/* Build all ants in parallel and return routes + costs. */
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
                                               float *costs);

/* Stage 1 reduction from raw ant costs to per-block best pairs. */
__global__ void aco_cuda_reduce_costs_stage_kernel(const float *costs,
                                                   int m,
                                                   float *out_costs,
                                                   int *out_ids);

/* Generic pair reduction stage (cost,id) -> per-block best pairs. */
__global__ void aco_cuda_reduce_pairs_stage_kernel(const float *in_costs,
                                                   const int *in_ids,
                                                   int n_items,
                                                   float *out_costs,
                                                   int *out_ids);

/* Deposit pheromone from one best ant route. */
__global__ void aco_cuda_deposit_best_tau_kernel(float *tau, int n,
                                                 int K, int max_route_len,
                                                 const int *routes,
                                                 const int *route_lens,
                                                 int best_ant,
                                                 float deposit);

/* Read reduced best pair and write device scalars (best ant + best cost). */
__global__ void aco_cuda_extract_best_pair_kernel(const float *best_cost_in,
                                                  const int *best_id_in,
                                                  int *best_ant_out,
                                                  float *best_cost_out);

/* Deposit pheromone using device-side best pair directly. */
__global__ void aco_cuda_deposit_best_tau_from_device_kernel(
    float *tau, int n, int K, int max_route_len,
    const int *routes, const int *route_lens,
    const int *best_ant_dev, const float *best_cost_dev,
    float Q);

#endif
