# include "solver.h"
#include "seq/internal.h"
#include "matrix.h"
#include "solution.h"
#include <float.h>

int	allocate_ctx(t_seq_ctx *ctx)
{
	ctx->tau = matrix_alloc(ctx->n);
	ctx->iter_best = solution_create(ctx->k, ctx->n);
	if (!seq_shared_init(&ctx->shared, ctx->n, ctx->params.candidate_k))
	{
		matrix_free(ctx->tau);
		solution_free(ctx->iter_best);
		return (0);
	}
	if (!seq_workspace_init(&ctx->ws, ctx->k, ctx->n,
			ctx->shared.visited_words))
	{
		matrix_free(ctx->tau);
		solution_free(ctx->iter_best);
		seq_shared_free(&ctx->shared);
		return (0);
	}
	if (!ctx->tau || !ctx->iter_best)
	{
		seq_workspace_free(&ctx->ws);
		matrix_free(ctx->tau);
		solution_free(ctx->iter_best);
		seq_shared_free(&ctx->shared);
		return (0);
	}
	return (1);
}

static void	init_tau_matrix(t_seq_ctx *ctx)
{
	int	i;
	int	j;

	i = 0;
	while (i <= ctx->n)
	{
		j = 0;
		while (j <= ctx->n)
		{
			if (i == j)
				ctx->tau[i][j] = 0.0;
			else
				ctx->tau[i][j] = ctx->tau0;
			j++;
		}
		i++;
	}
}

static void	init_runtime_state(t_seq_ctx *ctx)
{
	solution_reset(ctx->best_sol);
	*ctx->best_cost = DBL_MAX;
	ctx->stagnation_iters = 0;
	ctx->no_improve_epochs = 0;
	ctx->tau_max = ctx->tau0;
	ctx->tau_min = ctx->tau0 * 0.05;
	ctx->start_wall = seq_wall_time();
}

void	init_ctx(t_seq_ctx *ctx)
{
	init_tau_matrix(ctx);
	seq_shared_build_candidates(&ctx->shared, ctx->c, ctx->beta);
	seq_shared_update_scores(&ctx->shared, ctx->tau, ctx->alpha);
	init_runtime_state(ctx);
}

void	cleanup_ctx(t_seq_ctx *ctx)
{
	seq_workspace_free(&ctx->ws);
	seq_shared_free(&ctx->shared);
	solution_free(ctx->iter_best);
	matrix_free(ctx->tau);
}
