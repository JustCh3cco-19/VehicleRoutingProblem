#include "cuda/cuda_kernels.h"

void	launch_evaporate_tau(uint8_t *d_tau, int n, uint8_t delta,
			uint8_t q_min)
{
	uint64_t	total;
	int			blocks;

	total = (uint64_t)(n + 1) * (n + 1);
	blocks = (total + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_evaporate_tau<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_tau, total,
		delta, q_min);
}

void	launch_build_candidates(const float2 *d_coords, int *d_candidate_idx,
			float *d_eta_beta, t_cuda_params params)
{
	int	total;
	int	blocks;

	total = params.n + 1;
	blocks = (total + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_build_candidates<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_coords,
		d_candidate_idx, d_eta_beta, params);
}

void	launch_reset_ants(t_cuda_ant_state ants, t_cuda_params params,
			unsigned int seed)
{
	int	blocks;

	blocks = (params.m + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_reset_ants<<<blocks, CUDA_THREADS_PER_BLOCK>>>(ants, params, seed);
}

void	launch_construct_solutions(t_cuda_problem_data prob,
			t_cuda_ant_state ants, t_cuda_iter_stats *d_iter_stats,
			t_cuda_params params)
{
	int	warps_per_block;
	int	blocks;

	warps_per_block = CUDA_THREADS_PER_BLOCK / CUDA_WARP_SIZE;
	blocks = (params.m + warps_per_block - 1) / warps_per_block;
	kern_construct_solutions<<<blocks, CUDA_THREADS_PER_BLOCK>>>(prob, ants,
		d_iter_stats, params);
}

void	launch_deposit_best(uint8_t *d_tau, t_cuda_ant_state ants,
			t_cuda_deposit_params dep, t_cuda_params params)
{
	int	blocks;

	blocks = (params.k + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_deposit_best<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_tau, ants, dep,
		params);
}

void	launch_deposit_flat(uint8_t *d_tau, t_cuda_flat_routes flat,
			float deposit_amount, t_cuda_params params)
{
	int	blocks;

	blocks = (params.k + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_deposit_flat<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_tau, flat,
		deposit_amount, params);
}
