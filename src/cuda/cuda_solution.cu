extern "C" {
#include "solver.h"
#include "solution.h"
}

#include "cuda/cuda_internal.h"

static int	cuda_copy_ant_to_solution(t_cuda_solver_ctx *ctx, int ant_idx,
				t_solution *dst)
{
	int		v;
	int		global_step;
	int		len;
	t_route	*r;
	int		t;

	solution_reset(dst);
	global_step = 0;
	v = 0;
	for (; v < ctx->params->k; ++v)
	{
		len = ctx->h_route_lengths[ant_idx * ctx->params->k + v];
		r = &dst->routes[v];
		if (len < 2 || len > r->cap)
			return (0);
		r->len = len;
		t = 0;
		for (; t < len; ++t)
			r->nodes[t] = ctx->h_routes[(global_step + t) * ctx->m + ant_idx];
		global_step += len - 1;
	}
	return (1);
}

static void	cuda_update_flat_solution(t_cuda_solver_ctx *ctx)
{
	int	v;
	int	i;

	v = 0;
	for (; v < ctx->params->k; v++)
	{
		ctx->h_flat_lengths[v] = ctx->best_sol->routes[v].len;
		i = 0;
		for (; i < ctx->best_sol->routes[v].len; i++)
			ctx->h_flat_routes[v * (ctx->params->n + 1) + i]
				= ctx->best_sol->routes[v].nodes[i];
	}
	cudaMemcpy(ctx->flat.routes, ctx->h_flat_routes,
		ctx->params->k * (ctx->params->n + 1) * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(ctx->flat.lengths, ctx->h_flat_lengths,
		ctx->params->k * sizeof(int), cudaMemcpyHostToDevice);
}

t_status	cuda_copy_best_data(t_cuda_solver_ctx *ctx, int best_idx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMemcpy(ctx->h_routes, ctx->ants.routes,
			ctx->max_steps * ctx->m * sizeof(int),
			cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(ctx->h_route_lengths, ctx->ants.route_lengths,
			ctx->m * ctx->params->k * sizeof(int),
			cudaMemcpyDeviceToHost));
	cuda_copy_ant_to_solution(ctx, best_idx, ctx->best_sol);
	cuda_update_flat_solution(ctx);
	return (SOLVER_OK);
cleanup:
	return (status);
}

t_status	cuda_update_global_best(t_cuda_solver_ctx *ctx, int best_idx,
		float best_iter_cost)
{
	double		new_best;

	new_best = (double)best_iter_cost;
	if (new_best < ctx->global_best)
	{
		if (cuda_is_significant_improvement(ctx->global_best, new_best,
				ctx->min_rel_imp))
			ctx->iter_since_best = 0;
		else
			ctx->iter_since_best++;
		ctx->global_best = new_best;
		*(ctx->best_cost) = ctx->global_best;
		return (cuda_copy_best_data(ctx, best_idx));
	}
	ctx->iter_since_best++;
	return (SOLVER_OK);
}
