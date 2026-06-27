extern "C" {
#include "solver.h"
}

#include "cuda/cuda_internal.h"
#include <stdlib.h>

static t_status	cuda_alloc_device_ants(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMalloc(&ctx->ants.routes,
			ctx->max_steps * ctx->m * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.route_lengths,
			ctx->m * ctx->params->k * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.visited_l1,
			ctx->m * ctx->cuda_params.visited_row_stride));
	CHECK_CUDA(cudaMalloc(&ctx->ants.visited_l2,
			ctx->m * ctx->cuda_params.visited_l2_words * sizeof(uint64_t)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.rng_states,
			ctx->m * sizeof(unsigned int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.ant_summary,
			ctx->m * sizeof(t_cuda_ant_summary)));
	return (SOLVER_OK);
cleanup:
	return (status);
}

t_status	cuda_alloc_device(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMalloc(&ctx->d_coords,
			(ctx->params->n + 1) * sizeof(float2)));
	CHECK_CUDA(cudaMalloc(&ctx->d_tau,
			ctx->tau_elements * sizeof(uint8_t)));
	CHECK_CUDA(cudaMalloc(&ctx->d_candidate_idx,
			(ctx->params->n + 1) * ctx->cand_k * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->d_eta_beta,
			(ctx->params->n + 1) * ctx->cand_k * sizeof(float)));
	status = cuda_alloc_device_ants(ctx);
	if (status != SOLVER_OK)
		return (status);
	CHECK_CUDA(cudaMalloc(&ctx->d_iter_stats, sizeof(t_cuda_iter_stats)));
	CHECK_CUDA(cudaMalloc(&ctx->flat.routes,
			ctx->params->k * (ctx->params->n + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->flat.lengths,
			ctx->params->k * sizeof(int)));
	return (SOLVER_OK);
cleanup:
	return (status);
}

t_status	cuda_init_device(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMemcpy(ctx->d_coords, ctx->h_coords,
			(ctx->params->n + 1) * sizeof(float2), cudaMemcpyHostToDevice));
	launch_init_tau(ctx->d_tau, ctx->params->n, ctx->q_tau0);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_build_candidates(ctx->d_coords, ctx->d_candidate_idx,
		ctx->d_eta_beta, ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	return (SOLVER_OK);
cleanup:
	return (status);
}

t_status	cuda_iter_step(t_cuda_solver_ctx *ctx, t_cuda_problem_data prob)
{
	t_status	status;

	status = SOLVER_OK;
	launch_reset_ants(ctx->ants, ctx->cuda_params,
		ctx->params->seed + ctx->iter);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemset(ctx->d_iter_stats, 0, sizeof(t_cuda_iter_stats)));
	launch_construct_solutions(prob, ctx->ants, ctx->d_iter_stats,
		ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(ctx->h_ant_summary, ctx->ants.ant_summary,
			ctx->m * sizeof(t_cuda_ant_summary), cudaMemcpyDeviceToHost));
	return (SOLVER_OK);
cleanup:
	return (status);
}

t_status	cuda_update_pheromones(t_cuda_solver_ctx *ctx, int best_idx,
		float best_iter_cost)
{
	t_status				status;
	t_cuda_deposit_params	dep;

	status = SOLVER_OK;
	launch_evaporate_tau(ctx->d_tau, ctx->params->n,
		ctx->cuda_params.q_evap_delta, ctx->cuda_params.q_tau_min);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	dep.amount = (0.3f * ctx->cuda_params.q) / best_iter_cost;
	dep.best_ant = best_idx;
	launch_deposit_best(ctx->d_tau, ctx->ants, dep, ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	launch_deposit_flat(ctx->d_tau, ctx->flat,
		(0.7f * ctx->cuda_params.q) / (float)ctx->global_best,
		ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	return (SOLVER_OK);
cleanup:
	return (status);
}

void	cuda_cleanup(t_cuda_solver_ctx *ctx)
{
	cudaFree(ctx->d_coords);
	cudaFree(ctx->d_tau);
	cudaFree(ctx->d_candidate_idx);
	cudaFree(ctx->d_eta_beta);
	cudaFree(ctx->ants.routes);
	cudaFree(ctx->ants.route_lengths);
	cudaFree(ctx->ants.visited_l1);
	cudaFree(ctx->ants.visited_l2);
	cudaFree(ctx->ants.rng_states);
	cudaFree(ctx->ants.ant_summary);
	cudaFree(ctx->d_iter_stats);
	cudaFree(ctx->flat.routes);
	cudaFree(ctx->flat.lengths);
	free(ctx->h_coords);
	free(ctx->h_ant_summary);
	free(ctx->h_routes);
	free(ctx->h_route_lengths);
	free(ctx->h_flat_routes);
	free(ctx->h_flat_lengths);
}
