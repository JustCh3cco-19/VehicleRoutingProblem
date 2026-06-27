extern "C" {
#include "solver.h"
#include "solution.h"
}

#include "cuda/cuda_internal.h"
#include <float.h>
#include <string.h>

static t_status	run_solver_loop(t_cuda_solver_ctx *ctx)
{
	t_status			status;
	t_cuda_problem_data	prob;
	int					best_idx;
	float				best_iter_cost;

	prob.coords = ctx->d_coords;
	prob.tau = ctx->d_tau;
	prob.candidate_idx = ctx->d_candidate_idx;
	prob.eta_beta = ctx->d_eta_beta;
	while (!cuda_should_stop(ctx))
	{
		status = cuda_iter_step(ctx, prob);
		if (status != SOLVER_OK)
			return (status);
		cuda_update_accum_stats(ctx);
		if (cuda_select_iter_best_host(ctx->h_ant_summary, ctx->m,
				&best_idx, &best_iter_cost))
		{
			status = cuda_update_global_best(ctx, best_idx, best_iter_cost);
			if (status != SOLVER_OK)
				return (status);
			status = cuda_update_pheromones(ctx, best_idx, best_iter_cost);
			if (status != SOLVER_OK)
				return (status);
		}
		else
			ctx->iter_since_best++;
		ctx->iter++;
	}
	return (SOLVER_OK);
}

static void	init_loop_timers(t_cuda_solver_ctx *ctx)
{
	ctx->max_runtime = ctx->config.timeout_seconds;
	ctx->max_stagnation = ctx->config.stagnation_epochs;
	ctx->min_rel_imp = ctx->config.min_rel_improvement;
	ctx->progress_int = ctx->config.progress_interval_seconds;
	ctx->start_time = cuda_wall_time_seconds();
	if (ctx->progress_int > 0.0)
		ctx->next_progress = ctx->start_time + ctx->progress_int;
	else
		ctx->next_progress = 0.0;
}

static t_status	run_solver_orchestrator(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	init_loop_timers(ctx);
	status = run_solver_loop(ctx);
	if (ctx->config.log_level > LOG_SILENT)
		cuda_print_iteration_stats(ctx);
	return (status);
}

static void	init_solver_ctx(t_cuda_solver_ctx *ctx)
{
	ctx->iter = 0;
	ctx->iter_since_best = 0;
	ctx->global_best = DBL_MAX;
	memset(&ctx->accum_stats, 0, sizeof(t_cuda_iter_stats));
}

static t_status	setup_and_run(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = cuda_init_params(ctx);
	if (status == SOLVER_OK)
		status = cuda_alloc_host(ctx);
	if (status == SOLVER_OK)
		status = cuda_alloc_device(ctx);
	if (status == SOLVER_OK)
		status = cuda_init_device(ctx);
	if (status == SOLVER_OK)
		status = run_solver_orchestrator(ctx);
	cuda_cleanup(ctx);
	return (status);
}

t_status	vrp_solve_cuda(t_solver_params *params,
				t_cuda_coords coords,
				t_solution *best_solution,
				double *best_cost)
{
	t_cuda_solver_ctx	ctx;
	t_status			status;

	if (params->n <= 0 || params->k <= 0 || !coords.x || !coords.y
		|| !best_solution || !best_cost)
		return (SOLVER_ERR_INVALID_INPUT);
	ctx.params = params;
	ctx.coords = coords;
	ctx.best_sol = best_solution;
	ctx.best_cost = best_cost;
	init_solver_ctx(&ctx);
	status = setup_and_run(&ctx);
	if (status == SOLVER_OK)
		status = (*best_cost < DBL_MAX) ? SOLVER_OK
			: SOLVER_ERR_NO_SOLUTION;
	return (status);
}
