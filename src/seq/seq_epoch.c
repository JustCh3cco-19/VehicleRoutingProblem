#include "seq/internal.h"
#include <float.h>
#include <limits.h>

static void	reset_epoch_best(t_seq_ctx *ctx)
{
	ctx->iter_best_cost = DBL_MAX;
	ctx->iter_best_ant = INT_MAX;
}

static void	run_epoch_ants(t_seq_ctx *ctx)
{
	int	ant;

	ant = 0;
	while (ant < ctx->m)
	{
		run_ant(ctx, ant);
		ant++;
	}
}

static void	update_tau_bounds(t_seq_ctx *ctx)
{
	if (*ctx->best_cost > SOLVER_EPS)
	{
		ctx->tau_max = 1.0 / ((1.0 - ctx->rho) * (*ctx->best_cost));
		ctx->tau_min = ctx->tau_max * 0.05;
	}
}

static void	apply_global_improvement(t_seq_ctx *ctx)
{
	*ctx->best_cost = ctx->iter_best_cost;
	solution_copy(ctx->best_sol, ctx->iter_best);
	ctx->stagnation_iters = 0;
	ctx->no_improve_epochs = 0;
	ctx->improved_global = 1;
	update_tau_bounds(ctx);
}

static void	update_global_best(t_seq_ctx *ctx)
{
	if (seq_is_improvement(*ctx->best_cost, ctx->iter_best_cost,
			ctx->params.min_rel_improvement))
		apply_global_improvement(ctx);
	else
	{
		ctx->stagnation_iters++;
		ctx->no_improve_epochs++;
	}
}

void	run_epoch(t_seq_ctx *ctx)
{
	seq_shared_update_scores(&ctx->shared, ctx->tau, ctx->alpha);
	reset_epoch_best(ctx);
	run_epoch_ants(ctx);
	ctx->improved_global = 0;
	update_global_best(ctx);
}
