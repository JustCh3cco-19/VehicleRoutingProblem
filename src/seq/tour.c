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

static bool	can_extend_route(struct s_tour_ctx *ctx, int vehicle, int fut_cap);
static bool	append_selected_customer(struct s_tour_ctx *ctx, int vehicle,
		int *current);
static bool	fill_route_remainder(struct s_tour_ctx *ctx, int v, t_route *r);

static bool	build_vehicle_route(struct s_tour_ctx *ctx, int vehicle)
{
	t_route	*r;
	int		current;
	int		fut_cap;

	r = &ctx->ws->sol->routes[vehicle];
	if (!route_append(r, 0))
		return (false);
	current = 0;
	fut_cap = (ctx->k - vehicle - 1) * ctx->cap;
	while (can_extend_route(ctx, vehicle, fut_cap))
	{
		if (!append_selected_customer(ctx, vehicle, &current))
			break ;
	}
	if (!route_append(r, 0))
		return (false);
	return (true);
}

static bool	can_extend_route(struct s_tour_ctx *ctx, int vehicle, int fut_cap)
{
	if (ctx->remaining <= 0)
		return (false);
	if (ctx->remaining <= fut_cap)
		return (false);
	return (ctx->ws->route_loads[vehicle] < ctx->cap);
}

static bool	append_customer(struct s_tour_ctx *ctx, int vehicle, t_route *r,
		int next)
{
	if (!route_append(r, next))
		return (false);
	visited_set(ctx->ws->visited, next);
	ctx->ws->route_loads[vehicle]++;
	ctx->remaining--;
	return (true);
}

static bool	append_selected_customer(struct s_tour_ctx *ctx, int vehicle,
		int *current)
{
	t_route	*r;
	int		next;

	r = &ctx->ws->sol->routes[vehicle];
	next = select_next_customer(ctx->shared, *current, ctx->ws->visited,
			ctx->c, &ctx->ws->rng_state);
	if (next <= 0)
		return (false);
	if (!append_customer(ctx, vehicle, r, next))
		return (false);
	*current = next;
	return (true);
}

static bool	fill_remaining(struct s_tour_ctx *ctx)
{
	int		v;
	t_route	*r;

	v = ctx->k - 1;
	while (v >= 0 && ctx->remaining > 0)
	{
		r = &ctx->ws->sol->routes[v];
		if (r->len > 0 && r->nodes[r->len - 1] == 0)
			r->len--;
		if (!fill_route_remainder(ctx, v, r))
			return (false);
		if (!route_append(r, 0))
			return (false);
		v--;
	}
	return (true);
}

static int	last_route_node(t_route *r)
{
	if (r->len > 0)
		return (r->nodes[r->len - 1]);
	return (0);
}

static bool	fill_route_remainder(struct s_tour_ctx *ctx, int v, t_route *r)
{
	int	current;
	int	next;

	while (ctx->remaining > 0 && ctx->ws->route_loads[v] < ctx->cap)
	{
		current = last_route_node(r);
		next = find_nearest_unvisited(ctx->shared, current,
				ctx->ws->visited, ctx->c);
		if (next <= 0)
			break ;
		if (!append_customer(ctx, v, r, next))
			return (false);
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
