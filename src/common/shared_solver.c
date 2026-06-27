#include "internal.h"
#include "solution.h"

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

int	shared_select_next(t_shared_select_ctx *ctx)
{
	const double	*score_row;
	double			denom;
	int				idx;
	int				node;
	double			score;

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
		score = 0.0;
		if (score_row)
			score = score_row[node];
		else
		{
			score = fast_pow_nonneg(ctx->score_params->tau[ctx->current][node],
						ctx->score_params->alpha)
				* fast_pow_nonneg(ctx->score_params->eta[ctx->current][node],
						ctx->score_params->beta);
		}
		ctx->candidate_scores[idx] = score;
		denom += score;
		idx++;
	}
	return (roulette_select(ctx, denom));
}

static void	init_build_data(t_ant_build_ctx *ctx, int *unvisited_count,
				int *draw_index)
{
	int	i;

	solution_reset(ctx->sol);
	i = 0;
	while (i < ctx->n)
	{
		ctx->unvisited_nodes[i] = i + 1;
		ctx->random_draws[i] = rand01_state(ctx->rng_state);
		i++;
	}
	*unvisited_count = ctx->n;
	*draw_index = 0;
}

static void	init_select_ctx(t_shared_select_ctx *sel, t_ant_build_ctx *ctx)
{
	sel->unvisited_nodes = ctx->unvisited_nodes;
	sel->score_params = ctx->score_params;
	sel->candidate_scores = ctx->candidate_scores;
	sel->score_cache = ctx->score_cache;
}

static bool	build_single_route(t_ant_build_ctx *ctx, int vehicle,
				int *unvisited_count, int *draw_index)
{
	int					current;
	int					assigned;
	t_shared_select_ctx	sel;
	int					next;

	if (!route_append(&ctx->sol->routes[vehicle - 1], 0))
		return (false);
	current = 0;
	assigned = 0;
	init_select_ctx(&sel, ctx);
	while (*unvisited_count > 0 && assigned < ctx->vehicle_capacity_customers
		&& *unvisited_count > (ctx->k - vehicle)
		* ctx->vehicle_capacity_customers)
	{
		sel.current = current;
		sel.unvisited_count = *unvisited_count;
		sel.roulette_r = ctx->random_draws[(*draw_index)++];
		sel.selected_index = &current;
		next = shared_select_next(&sel);
		if (next <= 0)
			break ;
		if (!route_append(&ctx->sol->routes[vehicle - 1], next))
			return (false);
		assigned++;
		ctx->unvisited_nodes[current] = ctx->unvisited_nodes[
			--(*unvisited_count)];
		current = next;
	}
	return (route_append(&ctx->sol->routes[vehicle - 1], 0));
}

static bool	append_remaining_to_last(t_ant_build_ctx *ctx,
				int *unvisited_count)
{
	t_route	*last;
	int		last_customers;

	last = &ctx->sol->routes[ctx->k - 1];
	if (last->len > 0 && last->nodes[last->len - 1] == 0)
		last->len--;
	last_customers = last->len > 0 ? (last->len - 1) : 0;
	while (*unvisited_count > 0
		&& last_customers < ctx->vehicle_capacity_customers)
	{
		(*unvisited_count)--;
		if (!route_append(last, ctx->unvisited_nodes[*unvisited_count]))
			return (false);
		last_customers++;
	}
	return (route_append(last, 0));
}

bool	shared_build_ant_solution(t_ant_build_ctx *ctx)
{
	int		unvisited_count;
	int		draw_index;
	int		vehicle;

	if (ctx->vehicle_capacity_customers <= 0)
		ctx->vehicle_capacity_customers = ctx->n;
	init_build_data(ctx, &unvisited_count, &draw_index);
	vehicle = 1;
	while (vehicle <= ctx->k)
	{
		if (!build_single_route(ctx, vehicle, &unvisited_count, &draw_index))
			return (false);
		vehicle++;
	}
	if (unvisited_count > 0)
	{
		if (!append_remaining_to_last(ctx, &unvisited_count))
			return (false);
	}
	return (true);
}
