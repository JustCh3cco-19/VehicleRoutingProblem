#include "internal.h"

static int	roulette_select(t_shared_select_ctx *ctx, double denom)
{
	double	threshold;
	double	cumulative;
	int		chosen_idx;
	int		idx;

	threshold = ctx->roulette_r * denom;
	cumulative = 0.0;
	chosen_idx = ctx->unvisited_count - 1;
	idx = 0;
	while (idx < ctx->unvisited_count)
	{
		cumulative += ctx->candidate_scores[idx];
		if (cumulative >= threshold)
		{
			chosen_idx = idx;
			break ;
		}
		idx++;
	}
	if (ctx->selected_index)
		*ctx->selected_index = chosen_idx;
	return (ctx->unvisited_nodes[chosen_idx]);
}

static double	get_node_score(t_shared_select_ctx *ctx,
				const double *score_row, int node)
{
	if (score_row)
		return (score_row[node]);
	return (fast_pow_nonneg(ctx->score_params->tau[ctx->current][node],
			ctx->score_params->alpha)
		* fast_pow_nonneg(ctx->score_params->eta[ctx->current][node],
			ctx->score_params->beta));
}

int	shared_select_next(t_shared_select_ctx *ctx)
{
	const double	*score_row;
	double			denom;
	int				idx;
	int				node;

	if (ctx->unvisited_count <= 0)
	{
		if (ctx->selected_index)
			*ctx->selected_index = -1;
		return (0);
	}
	score_row = score_cache_get_row(ctx->score_cache, ctx->current,
			ctx->score_params);
	denom = 0.0;
	idx = 0;
	while (idx < ctx->unvisited_count)
	{
		node = ctx->unvisited_nodes[idx];
		ctx->candidate_scores[idx] = get_node_score(ctx, score_row, node);
		denom += ctx->candidate_scores[idx];
		idx++;
	}
	return (roulette_select(ctx, denom));
}
