#include "solver.h"
#include "config.h"
#include "seq/internal.h"
#include "solution.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

static int		seq_should_stop(t_seq_ctx *ctx);
static void		seq_after_epoch(t_seq_ctx *ctx);

static t_status	aco_vrp_run_with_config(struct s_seq_ctx *ctx)
{
	if (!allocate_ctx(ctx))
		return (SOLVER_ERR_ALLOCATION);
	init_ctx(ctx);
	ctx->iter = 0;
	while (!seq_should_stop(ctx))
	{
		run_epoch(ctx);
		seq_after_epoch(ctx);
		ctx->iter++;
	}
	cleanup_ctx(ctx);
	if (*ctx->best_cost < DBL_MAX)
		return (SOLVER_OK);
	return (SOLVER_ERR_NO_SOLUTION);
}

static int	seq_timeout_reached(t_seq_ctx *ctx)
{
	if (ctx->params.timeout_seconds <= 0.0)
		return (0);
	return ((seq_wall_time() - ctx->start_wall)
		>= ctx->params.timeout_seconds);
}

static int	seq_stagnation_reached(t_seq_ctx *ctx)
{
	if (ctx->params.stagnation_epochs <= 0)
		return (0);
	return (ctx->no_improve_epochs >= ctx->params.stagnation_epochs);
}

static int	seq_should_stop(t_seq_ctx *ctx)
{
	if (seq_timeout_reached(ctx))
		return (1);
	return (seq_stagnation_reached(ctx));
}

static void	seq_apply_pheromones(t_seq_ctx *ctx)
{
	evaporate_tau(ctx);
	deposit_iter_best(ctx);
	if (*ctx->best_cost < DBL_MAX)
		deposit_global_best(ctx);
	clamp_pheromones(ctx);
}

static void	seq_after_epoch(t_seq_ctx *ctx)
{
	if (ctx->iter_best_cost >= DBL_MAX)
		return ;
	seq_apply_pheromones(ctx);
	if (ctx->stagnation_iters >= ctx->stagnation_trigger)
	{
		reset_pheromones(ctx);
		ctx->stagnation_iters = 0;
	}
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

static void	init_seq_ctx(t_seq_ctx *ctx, t_solver_params *params,
				t_solution *best_solution, double *best_cost)
{
	ctx->n = params->n;
	ctx->k = params->k;
	ctx->cap = params->vehicle_capacity_customers;
	ctx->m = params->m;
	ctx->c = params->c;
	ctx->alpha = params->alpha;
	ctx->beta = params->beta;
	ctx->rho = params->rho;
	ctx->tau0 = params->tau0;
	ctx->q = params->q;
	ctx->seed = params->seed;
	ctx->best_sol = best_solution;
	ctx->best_cost = best_cost;
}

static t_status	setup_and_run_seq(t_seq_ctx *ctx, t_solver_params *params)
{
	runtime_config_load_env(&ctx->params);
	ctx->params.ants = params->m;
	ctx->params.seed = params->seed;
	if (ctx->m <= 0)
	{
		if (ctx->params.ants > 0)
			ctx->m = ctx->params.ants;
		else
			ctx->m = seq_choose_auto_ants(ctx->n);
	}
	ctx->stagnation_trigger = 32;
	if (ctx->params.stagnation_epochs > 0)
		ctx->stagnation_trigger = ctx->params.stagnation_epochs / 2;
	if (ctx->stagnation_trigger < 4)
		ctx->stagnation_trigger = 4;
	return (aco_vrp_run_with_config(ctx));
}

t_status	vrp_solve_with_capacity(t_solver_params *params,
		t_solution *best_solution, double *best_cost)
{
	struct s_seq_ctx	ctx;

	if (!params)
		return (SOLVER_ERR_INVALID_INPUT);
	init_seq_ctx(&ctx, params, best_solution, best_cost);
	return (setup_and_run_seq(&ctx, params));
}
