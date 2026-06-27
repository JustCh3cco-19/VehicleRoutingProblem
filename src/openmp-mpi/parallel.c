# include "solver.h"
#include "config.h"
#include "openmp-mpi/mpi_internal.h"
#include "solution.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
# include <mpi.h>
#endif

static void	par_solver_epoch_loop(t_par_solver_ctx *ctx, t_par_workspace *ws)
{
	int	iter;

	iter = 0;
	while (1)
	{
		if (ctx->fixed_epochs > 0 && iter >= ctx->fixed_epochs)
			break ;
		if (ctx->fixed_epochs <= 0 && ctx->iter_since_best >= ctx->max_stagnation_epochs)
			break ;
		if (ctx->max_runtime_sec > 0.0 && (par_wall_time() - ctx->start_time)
			>= ctx->max_runtime_sec)
			break ;
#ifdef USE_MPI
#pragma omp master
		par_async_wait_and_apply(&ctx->async_ctx, ctx->tau_mat, ctx->mpi_rank,
			ctx->mpi_size);
#pragma omp barrier
#endif
		par_solver_scores(ctx);
		par_solver_ants(ctx, ws, iter);
		par_solver_reduce_best(ctx, iter);
		par_solver_evaporate(ctx);
		par_solver_deposit(ctx);
		iter++;
	}
}

void	par_solver_thread_run(t_par_solver_ctx *ctx)
{
	t_par_workspace	ws;

	if (!par_ws_init(&ws, ctx->k, ctx->n, ctx->shared.visited_words,
			ctx->shared.meta_words))
	{
#pragma omp atomic write
		ctx->workspace_failed = 1;
	}
#pragma omp barrier
	if (ctx->workspace_failed)
		return ;
	par_solver_log_start(ctx);
	par_solver_epoch_loop(ctx, &ws);
	par_ws_free(&ws);
}

static t_status	par_vrp_run(t_par_solver_ctx *ctx)
{
	if (!par_solver_alloc(ctx))
		return (SOLVER_ERR_ALLOCATION);
	if (!par_solver_init(ctx))
	{
		par_solver_free(ctx);
		return (SOLVER_ERR_ALLOCATION);
	}
#pragma omp parallel default(shared) proc_bind(close)
	par_solver_thread_run(ctx);
	if (ctx->workspace_failed)
	{
		par_solver_free(ctx);
		return (SOLVER_ERR_ALLOCATION);
	}
	if (ctx->log_level > LOG_SILENT && ctx->mpi_rank == 0)
	{
		fprintf(stderr, "ACO Parallel Ultimate Completion. Best: %.3f. Time: %.3fs\n",
			*ctx->best_cost, par_wall_time() - ctx->start_time);
	}
	par_solver_free(ctx);
	if (*ctx->best_cost < DBL_MAX)
		return (SOLVER_OK);
	return (SOLVER_ERR_NO_SOLUTION);
}

t_status	vrp_solve(t_solver_params *params, t_solution *best_solution,
		double *best_cost)
{
	int	cap;

	if (!params)
		return (SOLVER_ERR_INVALID_INPUT);
	if (params->k > 0)
		cap = (int)(((long long)120 * params->n + 100 * params->k - 1)
				/ (100 * params->k));
	else
		cap = params->n;
	params->vehicle_capacity_customers = cap;
	return (vrp_solve_with_capacity(params, best_solution, best_cost));
}

t_status	vrp_solve_with_capacity(t_solver_params *params,
		t_solution *best_solution, double *best_cost)
{
	t_par_solver_ctx	ctx;
	t_config		config;

	if (!params || params->n <= 0 || params->k <= 0 || !params->c
		|| !best_solution || !best_cost)
		return (SOLVER_ERR_INVALID_INPUT);
	ctx.n = params->n;
	ctx.k = params->k;
	ctx.cap = params->vehicle_capacity_customers;
	ctx.m = params->m;
	ctx.c = params->c;
	ctx.alpha = params->alpha;
	ctx.beta = params->beta;
	ctx.rho = params->rho;
	ctx.tau0 = params->tau0;
	ctx.q = params->q;
	ctx.seed = params->seed;
	ctx.best_sol = best_solution;
	ctx.best_cost = best_cost;
	ctx.mpi_rank = 0;
	ctx.mpi_size = 1;
#ifdef USE_MPI
	int mpi_init = 0;
	MPI_Initialized(&mpi_init);
	if (mpi_init)
	{
		MPI_Comm_rank(MPI_COMM_WORLD, &ctx.mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &ctx.mpi_size);
	}
#endif
	runtime_config_load_env(&config);
	ctx.config = &config;
	ctx.cand_k = par_choose_candidate_count(params->n, config.candidate_k);
	ctx.total_m = (params->m <= 0) ? ((params->n / 2) * ctx.mpi_size)
		: params->m;
	if (params->m <= 0 && config.ants > 0)
		ctx.total_m = config.ants;
	int rank_extra = (ctx.mpi_rank < (ctx.total_m % ctx.mpi_size)) ? ctx.mpi_rank
		: (ctx.total_m % ctx.mpi_size);
	ctx.ant_off = ctx.mpi_rank * (ctx.total_m / ctx.mpi_size) + rank_extra;
	ctx.local_m = ctx.total_m / ctx.mpi_size + (ctx.mpi_rank < (ctx.total_m % ctx.mpi_size));
	ctx.max_runtime_sec = config.timeout_seconds;
	ctx.max_stagnation_epochs = config.stagnation_epochs;
	ctx.min_rel_improvement = config.min_rel_improvement;
	ctx.progress_interval_sec = config.progress_interval_seconds;
	ctx.log_level = config.log_level;
	ctx.fixed_epochs = config.fixed_epochs;
	if (ctx.max_stagnation_epochs <= 0)
		ctx.max_stagnation_epochs = 100;
	return (par_vrp_run(&ctx));
}
