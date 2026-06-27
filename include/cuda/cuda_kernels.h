#ifndef CUDA_KERNELS_H
# define CUDA_KERNELS_H

#include "cuda_types.h"
#include <cuda_runtime.h>
#include <stdint.h>

# define CUDA_THREADS_PER_BLOCK 128
# define CUDA_WARP_SIZE 32
# define CUDA_CACHE_LINE_SIZE 128
# define CUDA_MAX_CANDIDATES 32
# define CUDA_EPS 1e-9f

__global__ void	kern_init_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t q_tau0);
__global__ void	kern_build_candidates(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
__global__ void	kern_reset_ants(t_cuda_ant_state ants,
					t_cuda_params params, unsigned int seed);
__global__ void	kern_construct_solutions(t_cuda_problem_data prob,
					t_cuda_ant_state ants, t_cuda_iter_stats *d_iter_stats,
					t_cuda_params params);
__global__ void	kern_evaporate_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t delta, uint8_t q_min);
__global__ void	kern_deposit_best(uint8_t *d_tau, t_cuda_ant_state ants,
					t_cuda_deposit_params dep, t_cuda_params params);
__global__ void	kern_deposit_flat(uint8_t *d_tau, t_cuda_flat_routes flat,
					float deposit_amount, t_cuda_params params);

void			launch_init_tau(uint8_t *d_tau, int n, uint8_t q_tau0);
void			launch_build_candidates(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
void			launch_reset_ants(t_cuda_ant_state ants,
					t_cuda_params params, unsigned int seed);
void			launch_construct_solutions(t_cuda_problem_data prob,
					t_cuda_ant_state ants, t_cuda_iter_stats *d_iter_stats,
					t_cuda_params params);
void			launch_evaporate_tau(uint8_t *d_tau, int n, uint8_t delta,
					uint8_t q_min);
void			launch_deposit_best(uint8_t *d_tau, t_cuda_ant_state ants,
					t_cuda_deposit_params dep, t_cuda_params params);
void			launch_deposit_flat(uint8_t *d_tau, t_cuda_flat_routes flat,
					float deposit_amount, t_cuda_params params);

#endif
