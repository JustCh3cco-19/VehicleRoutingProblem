#include "aco.h"
#include "config.h"
#include "seq/internal.h"
#include "solution.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

static AcoStatus	aco_vrp_run_with_config(struct s_seq_ctx *ctx)
{
	if (!allocate_ctx(ctx))
		return (ACO_ERR_ALLOCATION);
	init_ctx(ctx);
	ctx->iter = 0;
	while (1)
	{
		if (ctx->params.timeout_seconds > 0.0 && (seq_wall_time()
				- ctx->start_wall) >= ctx->params.timeout_seconds)
			break ;
		run_epoch(ctx);
		if (ctx->iter_best_cost < DBL_MAX)
		{
			evaporate_tau(ctx);
			deposit_iter_best(ctx);
			if (*ctx->best_cost < DBL_MAX)
				deposit_global_best(ctx);
			clamp_pheromones(ctx);
			if (ctx->stagnation_iters >= ctx->stagnation_trigger)
			{
				reset_pheromones(ctx);
				ctx->stagnation_iters = 0;
			}
		}
		if (ctx->params.stagnation_epochs > 0 && ctx->no_improve_epochs
				>= ctx->params.stagnation_epochs)
			break ;
		ctx->iter++;
	}
	cleanup_ctx(ctx);
	if (*ctx->best_cost < DBL_MAX)
		return (ACO_OK);
	return (ACO_ERR_NO_SOLUTION);
}

AcoStatus	aco_vrp(int n, int k, int m, double **c, double alpha, double beta,
		double rho, double tau0, double q, unsigned int seed,
		Solution *best_solution, double *best_cost)
{
	int	cap;

	if (k > 0)
		cap = (int)(((long long)120 * n + 100 * k - 1) / (100 * k));
	else
		cap = n;
	return (aco_vrp_with_capacity(n, k, cap, m, c, alpha, beta, rho, tau0,
			q, seed, best_solution, best_cost));
}

AcoStatus	aco_vrp_with_capacity(int n, int k, int cap, int m, double **c,
		double alpha, double beta, double rho, double tau0, double q,
		unsigned int seed, Solution *best_solution, double *best_cost)
{
	struct s_seq_ctx	ctx;

	ctx.n = n;
	ctx.k = k;
	ctx.cap = cap;
	ctx.m = m;
	ctx.c = c;
	ctx.alpha = alpha;
	ctx.beta = beta;
	ctx.rho = rho;
	ctx.tau0 = tau0;
	ctx.q = q;
	ctx.seed = seed;
	ctx.best_sol = best_solution;
	ctx.best_cost = best_cost;
	aco_runtime_config_load_env(&ctx.params);
	ctx.params.ants = m;
	ctx.params.seed = seed;
	if (ctx.m <= 0)
	{
		if (ctx.params.ants > 0)
			ctx.m = ctx.params.ants;
		else
			ctx.m = seq_choose_auto_ants(ctx.n);
	}
	ctx.stagnation_trigger = 32;
	if (ctx.params.stagnation_epochs > 0)
		ctx.stagnation_trigger = ctx.params.stagnation_epochs / 2;
	if (ctx.stagnation_trigger < 4)
		ctx.stagnation_trigger = 4;
	return (aco_vrp_run_with_config(&ctx));
}
