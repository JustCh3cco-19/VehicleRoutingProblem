#ifndef ACO_CUDA_V3_KERNELS_H
#define ACO_CUDA_V3_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#define CUDA_V3_THREADS_PER_BLOCK 128
#define CUDA_V3_MAX_CANDIDATES 32
#define CUDA_V3_EPS 1e-9f
#define CUDA_V3_WARP_SIZE 32

typedef struct {
  int n;
  int K;
  int m;
  int cap;
  int cand_k;
  int visited_words;
  int route_stride;
  float alpha;
  float beta;
  float rho;
  float tau0;
  float Q;
  float tau_min;
  float tau_max;
  float depot_close_weight;
} CudaV3Params;

typedef struct {
  int feasible;
  int unvisited_count;
  float cost;
} CudaV3AntSummary;

__global__ void kernel_build_candidate_lists_v3(const float *costs,
                                                int *candidate_idx,
                                                float *eta_beta, int n,
                                                int cand_k, float beta);

__global__ void kernel_init_tau_v3(float *tau, int n, float tau0);

__global__ void kernel_reset_ant_state_v3(int *routes, int *route_lengths,
                                          int *route_loads, int *curr_nodes,
                                          uint64_t *visited,
                                          unsigned int *rng_states,
                                          CudaV3AntSummary *ant_summary,
                                          CudaV3Params params,
                                          unsigned int seed);

__global__ void kernel_construct_solutions_v3(
    const float *costs, const float *tau, const int *candidate_idx,
    const float *eta_beta, int *routes, int *route_lengths, int *route_loads,
    int *curr_nodes, uint64_t *visited, unsigned int *rng_states,
    CudaV3AntSummary *ant_summary, CudaV3Params params);

__global__ void kernel_evaporate_tau_v3(float *tau, int n, float rho);

__global__ void kernel_deposit_solution_v3(float *tau, const int *routes,
                                           const int *route_lengths, int K,
                                           int route_stride, int n,
                                           float deposit);

__global__ void kernel_clamp_tau_v3(float *tau, int n, float tau_min,
                                    float tau_max);

__global__ void kernel_recenter_tau_v3(float *tau, int n, float tau0);

void launch_build_candidate_lists_v3(const float *costs, int *candidate_idx,
                                     float *eta_beta, int n, int cand_k,
                                     float beta);
void launch_init_tau_v3(float *tau, int n, float tau0);
void launch_reset_ant_state_v3(int *routes, int *route_lengths,
                               int *route_loads, int *curr_nodes,
                               uint64_t *visited, unsigned int *rng_states,
                               CudaV3AntSummary *ant_summary,
                               CudaV3Params params, unsigned int seed);
void launch_construct_solutions_v3(const float *costs, const float *tau,
                                   const int *candidate_idx,
                                   const float *eta_beta, int *routes,
                                   int *route_lengths, int *route_loads,
                                   int *curr_nodes, uint64_t *visited,
                                   unsigned int *rng_states,
                                   CudaV3AntSummary *ant_summary,
                                   CudaV3Params params);
void launch_evaporate_tau_v3(float *tau, int n, float rho);
void launch_deposit_solution_v3(float *tau, const int *routes,
                                const int *route_lengths, int K,
                                int route_stride, int n, float deposit);
void launch_clamp_tau_v3(float *tau, int n, float tau_min, float tau_max);
void launch_recenter_tau_v3(float *tau, int n, float tau0);

#endif
