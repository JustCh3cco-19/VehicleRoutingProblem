# include "solver.h"
#include "seq/internal.h"
#include <stdint.h>

static void	calc_denom_large(const t_seq_shared *shared, int current,
		const uint64_t *visited, struct s_select_ctx *ctx)
{
	ctx->cand_row = shared->candidate_idx + (size_t)current * (size_t)shared->stride;
	ctx->score_row = shared->score + (size_t)current * (size_t)shared->stride;
	ctx->k = shared->candidate_k;
	ctx->denom = 0.0;
	ctx->t = 0;
	while (ctx->t < ctx->k)
	{
		ctx->node = ctx->cand_row[ctx->t];
		if (ctx->node > 0 && !visited_is_set(visited, ctx->node))
		{
			ctx->w = (double)ctx->score_row[ctx->t];
			if (ctx->w > 0.0)
				ctx->denom += ctx->w;
		}
		ctx->t++;
	}
}

static int	choose_large(unsigned int *rng_state, struct s_select_ctx *ctx,
		const uint64_t *visited)
{
	if (ctx->denom > 0.0)
	{
		ctx->threshold = seq_rand01(rng_state) * ctx->denom;
		ctx->cumulative = 0.0;
		ctx->last_valid = 0;
		ctx->t = 0;
		while (ctx->t < ctx->k)
		{
			ctx->node = ctx->cand_row[ctx->t];
			if (ctx->node > 0 && !visited_is_set(visited, ctx->node))
			{
				ctx->w = (double)ctx->score_row[ctx->t];
				if (ctx->w > 0.0)
				{
					ctx->cumulative += ctx->w;
					ctx->last_valid = ctx->node;
					if (ctx->cumulative >= ctx->threshold)
						return (ctx->node);
				}
			}
			ctx->t++;
		}
		if (ctx->last_valid > 0)
			return (ctx->last_valid);
	}
	return (0);
}

int	select_large(const t_seq_shared *shared, int current,
		const uint64_t *visited, unsigned int *rng_state)
{
	struct s_select_ctx	ctx;
	int					res;

	calc_denom_large(shared, current, visited, &ctx);
	res = choose_large(rng_state, &ctx, visited);
	return (res);
}

int	select_next_customer(const t_seq_shared *shared, int current,
		const uint64_t *visited, double **c, unsigned int *rng_state)
{
	int	res;

	if (shared->candidate_k <= 1024)
		res = select_small(shared, current, visited, rng_state);
	else
		res = select_large(shared, current, visited, rng_state);
	if (res <= 0)
		return (find_nearest_unvisited(shared, current, visited, c));
	return (res);
}
