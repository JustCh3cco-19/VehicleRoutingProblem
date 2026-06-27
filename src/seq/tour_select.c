# include "solver.h"
#include "seq/internal.h"
#include <float.h>
#include <stdint.h>

double	seq_rand01(unsigned int *state)
{
	unsigned int	x;

	x = *state;
	if (x == 0u)
		x = 1u;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return ((double)x / 4294967295.0);
}

int	find_nearest_unvisited(const t_seq_shared *shared, int current,
		const uint64_t *restrict visited, double **restrict c)
{
	int					best;
	double				best_dist;
	int					n;
	const double		*c_row;
	int					node;

	best = 0;
	best_dist = DBL_MAX;
	n = shared->n;
	c_row = c[current];
	node = 1;
	while (node <= n)
	{
		if (!visited_is_set(visited, node) && c_row[node] < best_dist)
		{
			best_dist = c_row[node];
			best = node;
		}
		node++;
	}
	return (best);
}

static void	populate_small(const t_seq_shared *shared, int current,
		const uint64_t *visited, struct s_select_ctx *ctx)
{
	ctx->cand_row = shared->candidate_idx + (size_t)current * (size_t)shared->stride;
	ctx->score_row = shared->score + (size_t)current * (size_t)shared->stride;
	ctx->k = shared->candidate_k;
	ctx->count = 0;
	ctx->denom = 0.0;
	ctx->t = 0;
	while (ctx->t < ctx->k)
	{
		ctx->node = ctx->cand_row[ctx->t];
		if (ctx->node > 0 && !visited_is_set(visited, ctx->node))
		{
			ctx->w = (double)ctx->score_row[ctx->t];
			if (ctx->w > 0.0)
			{
				ctx->denom += ctx->w;
				ctx->nodes[ctx->count] = ctx->node;
				ctx->scores[ctx->count] = ctx->w;
				ctx->count++;
			}
		}
		ctx->t++;
	}
}

static int	choose_small(unsigned int *rng_state, struct s_select_ctx *ctx)
{
	int		i;
	double	w;

	if (ctx->denom > 0.0)
	{
		ctx->threshold = seq_rand01(rng_state) * ctx->denom;
		ctx->cumulative = 0.0;
		ctx->last_valid = 0;
		i = 0;
		while (i < ctx->count)
		{
			w = ctx->scores[i];
			ctx->cumulative += w;
			ctx->last_valid = ctx->nodes[i];
			if (ctx->cumulative >= ctx->threshold)
				return (ctx->nodes[i]);
			i++;
		}
		if (ctx->last_valid > 0)
			return (ctx->last_valid);
	}
	return (0);
}

int	select_small(const t_seq_shared *shared, int current,
		const uint64_t *visited, unsigned int *rng_state)
{
	struct s_select_ctx	ctx;
	int					res;

	populate_small(shared, current, visited, &ctx);
	res = choose_small(rng_state, &ctx);
	return (res);
}
