# include "solver.h"
#include "seq/internal.h"
#include "solution.h"
#include <stdbool.h>
#include <string.h>

struct s_tour_ctx
{
	t_seq_workspace		*ws;
	const t_seq_shared	*shared;
	int					k;
	int					cap;
	int					remaining;
	double				**c;
};

static bool	build_vehicle_route(struct s_tour_ctx *ctx, int vehicle)
{
	t_route	*r;
	int		current;
	int		next;
	int		fut_cap;

	r = &ctx->ws->sol->routes[vehicle];
	if (!route_append(r, 0))
		return (false);
	current = 0;
	fut_cap = (ctx->k - vehicle - 1) * ctx->cap;
	while (ctx->remaining > 0 && ctx->remaining > fut_cap
		&& ctx->ws->route_loads[vehicle] < ctx->cap)
	{
		next = select_next_customer(ctx->shared, current, ctx->ws->visited,
				ctx->c, &ctx->ws->rng_state);
		if (next <= 0)
			break ;
		if (!route_append(r, next))
			return (false);
		visited_set(ctx->ws->visited, next);
		ctx->ws->route_loads[vehicle]++;
		ctx->remaining--;
		current = next;
	}
	if (!route_append(r, 0))
		return (false);
	return (true);
}

static bool	fill_remaining(struct s_tour_ctx *ctx)
{
	int		v;
	int		current;
	int		next;
	t_route	*r;

	v = ctx->k - 1;
	while (v >= 0 && ctx->remaining > 0)
	{
		r = &ctx->ws->sol->routes[v];
		if (r->len > 0 && r->nodes[r->len - 1] == 0)
			r->len--;
		while (ctx->remaining > 0 && ctx->ws->route_loads[v] < ctx->cap)
		{
			current = 0;
			if (r->len > 0)
				current = r->nodes[r->len - 1];
			next = find_nearest_unvisited(ctx->shared, current,
					ctx->ws->visited, ctx->c);
			if (next <= 0)
				break ;
			if (!route_append(r, next))
				return (false);
			visited_set(ctx->ws->visited, next);
			ctx->ws->route_loads[v]++;
			ctx->remaining--;
		}
		if (!route_append(r, 0))
			return (false);
		v--;
	}
	return (true);
}

bool	build_ant_solution(t_seq_workspace *ws, const t_seq_shared *shared,
		int k, int vehicle_capacity_customers, double **restrict c)
{
	struct s_tour_ctx	ctx;
	int					vehicle;

	solution_reset(ws->sol);
	memset(ws->visited, 0, (size_t)shared->visited_words * sizeof(uint64_t));
	memset(ws->route_loads, 0, (size_t)k * sizeof(int));
	ctx.ws = ws;
	ctx.shared = shared;
	ctx.k = k;
	ctx.cap = vehicle_capacity_customers;
	if (ctx.cap <= 0)
		ctx.cap = shared->n;
	ctx.remaining = shared->n;
	ctx.c = c;
	vehicle = 0;
	while (vehicle < k)
	{
		if (!build_vehicle_route(&ctx, vehicle))
			return (false);
		vehicle++;
	}
	if (ctx.remaining > 0)
	{
		if (!fill_remaining(&ctx))
			return (false);
	}
	return (true);
}
