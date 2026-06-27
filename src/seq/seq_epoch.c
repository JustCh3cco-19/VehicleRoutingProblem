#include "aco.h"
#include "seq/internal.h"
#include "solution.h"
#include <float.h>
#include <limits.h>
#include <math.h>

void	run_ant(t_seq_ctx *ctx, int ant)
{
	double	cost;

	ctx->ws.rng_state = aco_make_ant_seed(ctx->seed, ctx->iter, ant);
	if (!build_ant_solution(&ctx->ws, &ctx->shared, ctx->k, ctx->cap, ctx->c))
		return ;
	cost = solution_cost(ctx->ws.sol, ctx->c);
	if (cost < ctx->iter_best_cost || (fabs(cost - ctx->iter_best_cost)
			<= ACO_EPS && ant < ctx->iter_best_ant))
	{
		ctx->iter_best_cost = cost;
		ctx->iter_best_ant = ant;
		solution_copy(ctx->iter_best, ctx->ws.sol);
	}
}

void	run_epoch(t_seq_ctx *ctx)
{
	int	ant;

	seq_shared_update_scores(&ctx->shared, ctx->tau, ctx->alpha);
	ctx->iter_best_cost = DBL_MAX;
	ctx->iter_best_ant = INT_MAX;
	ant = 0;
	while (ant < ctx->m)
	{
		run_ant(ctx, ant);
		ant++;
	}
	ctx->improved_global = 0;
	if (seq_is_improvement(*ctx->best_cost, ctx->iter_best_cost,
			ctx->params.min_rel_improvement))
	{
		*ctx->best_cost = ctx->iter_best_cost;
		solution_copy(ctx->best_sol, ctx->iter_best);
		ctx->stagnation_iters = 0;
		ctx->no_improve_epochs = 0;
		ctx->improved_global = 1;
		if (*ctx->best_cost > ACO_EPS)
		{
			ctx->tau_max = 1.0 / ((1.0 - ctx->rho) * (*ctx->best_cost));
			ctx->tau_min = ctx->tau_max * 0.05;
		}
	}
	else
	{
		ctx->stagnation_iters++;
		ctx->no_improve_epochs++;
	}
}
