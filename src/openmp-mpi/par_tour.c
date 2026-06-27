#include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "solution.h"
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

static bool	par_can_extend_route(t_par_tour_ctx *ctx, int v,
				int fut_cap);
static bool	par_append_selected(t_par_tour_ctx *ctx, int v, int *curr);

static void	check_visited_word(const aco_mpi_rank_shared_t *s,
				const aco_mpi_workspace_t *ws, int w,
				struct s_nearest_state *state)
{
	uint64_t	mask;
	int			base;
	int			bits;
	int			bit;
	int			node;

	mask = ~ws->visited[w];
	base = w << 6;
	if (w == s->visited_words - 1)
	{
		bits = (s->n % 64) + 1;
		if (bits < 64)
			mask &= (1ull << bits) - 1;
	}
	if (w == 0)
		mask &= ~1ull;
	while (mask != 0)
	{
		bit = __builtin_ctzll(mask);
		node = base + bit;
		if (state->row[node] < state->best_d)
		{
			state->best_d = state->row[node];
			state->best = node;
		}
		mask &= mask - 1;
	}
}

int	par_nearest_unvisited(const aco_mpi_rank_shared_t *s, int curr,
		const aco_mpi_workspace_t *ws, const aco_mpi_matrix_float_t *c)
{
	struct s_nearest_state	state;
	int						mw;
	uint64_t				m_mask;
	int						w_off;
	int						w;

	state.best = 0;
	state.best_d = FLT_MAX;
	state.row = c->rows[curr];
	mw = 0;
	while (mw < s->meta_words)
	{
		m_mask = ws->meta_active[mw];
		while (m_mask != 0)
		{
			w_off = __builtin_ctzll(m_mask);
			w = (mw << 6) + w_off;
			if (w >= s->visited_words)
				break ;
			check_visited_word(s, ws, w, &state);
			m_mask &= m_mask - 1;
		}
		mw++;
	}
	return (state.best);
}

static void	par_populate_small(struct s_par_tour_ctx *tour, int curr,
		struct s_par_select_ctx *sel)
{
	sel->cands = tour->s->cand_idx + (size_t)curr * (size_t)tour->s->stride;
	sel->sc = tour->scores + (size_t)curr * (size_t)tour->s->stride;
	sel->count = 0;
	sel->denom = 0.0f;
	sel->t = 0;
	while (sel->t < tour->s->cand_k)
	{
		sel->node = sel->cands[sel->t];
		if (sel->node > 0 && !((tour->ws->visited[(unsigned)sel->node >> 6]
				>> ((unsigned)sel->node & 63u)) & 1u))
		{
			sel->w = sel->sc[sel->t];
			if (sel->w > 0.0f)
			{
				sel->denom += sel->w;
				sel->nodes[sel->count] = sel->node;
				sel->scores[sel->count] = sel->w;
				sel->count++;
			}
		}
		sel->t++;
	}
}

static int	par_select_small(struct s_par_tour_ctx *tour, int curr)
{
	struct s_par_select_ctx	sel;
	int						i;
	float					w;

	par_populate_small(tour, curr, &sel);
	if (sel.denom > 0.0f)
	{
		sel.thres = (float)par_rand01(&tour->ws->rng_state) * sel.denom;
		sel.cum = 0.0f;
		i = 0;
		while (i < sel.count)
		{
			w = sel.scores[i];
			sel.cum += w;
			if (sel.cum >= sel.thres)
				return (sel.nodes[i]);
			i++;
		}
		if (sel.count > 0)
			return (sel.nodes[sel.count - 1]);
	}
	return (0);
}

static void	par_calc_denom_large(struct s_par_tour_ctx *tour, int curr,
		struct s_par_select_ctx *sel)
{
	sel->cands = tour->s->cand_idx + (size_t)curr * (size_t)tour->s->stride;
	sel->sc = tour->scores + (size_t)curr * (size_t)tour->s->stride;
	sel->denom = 0.0f;
	sel->t = 0;
	while (sel->t < tour->s->cand_k)
	{
		sel->node = sel->cands[sel->t];
		if (sel->node > 0 && !((tour->ws->visited[(unsigned)sel->node >> 6]
				>> ((unsigned)sel->node & 63u)) & 1u))
		{
			sel->denom += sel->sc[sel->t];
		}
		sel->t++;
	}
}

static int	par_choose_large(struct s_par_tour_ctx *tour,
		struct s_par_select_ctx *sel)
{
	if (sel->denom > 0.0f)
	{
		sel->thres = (float)par_rand01(&tour->ws->rng_state) * sel->denom;
		sel->cum = 0.0f;
		sel->t = 0;
		while (sel->t < tour->s->cand_k)
		{
			sel->node = sel->cands[sel->t];
			if (sel->node > 0 && !((tour->ws->visited[(unsigned)sel->node >> 6]
					>> ((unsigned)sel->node & 63u)) & 1u))
			{
				sel->cum += sel->sc[sel->t];
				if (sel->cum >= sel->thres)
					return (sel->node);
			}
			sel->t++;
		}
	}
	return (0);
}

static int	par_select_large(struct s_par_tour_ctx *tour, int curr)
{
	struct s_par_select_ctx	sel;
	int						res;

	par_calc_denom_large(tour, curr, &sel);
	res = par_choose_large(tour, &sel);
	return (res);
}

static int	par_select_next(struct s_par_tour_ctx *ctx, int curr)
{
	int	next;

	next = 0;
	if (ctx->s->cand_k <= 1024)
		next = par_select_small(ctx, curr);
	else
		next = par_select_large(ctx, curr);
	if (next <= 0)
		next = par_nearest_unvisited(ctx->s, curr, ctx->ws, ctx->c);
	return (next);
}

static bool	par_build_vehicle_route(struct s_par_tour_ctx *ctx, int v)
{
	int	curr;
	int	rem_v;
	int	fut_cap;

	if (!par_route_append(&ctx->ws->sol->routes[v], 0))
		return (false);
	curr = 0;
	while (1)
	{
		rem_v = ctx->k - v - 1;
		fut_cap = rem_v * ctx->cap;
		if (!par_can_extend_route(ctx, v, fut_cap))
			break ;
		if (!par_append_selected(ctx, v, &curr))
			break ;
	}
	if (!par_route_append(&ctx->ws->sol->routes[v], 0))
		return (false);
	return (true);
}

static bool	par_can_extend_route(struct s_par_tour_ctx *ctx, int v,
		int fut_cap)
{
	if (ctx->remaining <= 0)
		return (false);
	if (ctx->remaining <= fut_cap)
		return (false);
	return (ctx->ws->route_loads[v] < ctx->cap);
}

static void	par_mark_visited(struct s_par_tour_ctx *ctx, int next)
{
	ctx->ws->visited[(unsigned)next >> 6] |= (1ull
			<< ((unsigned)next & 63u));
	if (ctx->ws->visited[(unsigned)next >> 6] == 0xFFFFFFFFFFFFFFFFull)
		ctx->ws->meta_active[(unsigned)next >> 12] &= ~(1ull
				<< (((unsigned)next >> 6) & 63u));
}

static bool	par_append_selected(struct s_par_tour_ctx *ctx, int v, int *curr)
{
	int	next;

	next = par_select_next(ctx, *curr);
	if (next <= 0)
		return (false);
	if (!par_route_append(&ctx->ws->sol->routes[v], next))
		return (false);
	par_mark_visited(ctx, next);
	ctx->ws->route_loads[v]++;
	ctx->remaining--;
	*curr = next;
	return (true);
}

bool	par_build_ant(t_par_tour_ctx *ctx)
{
	int	v;

	solution_reset(ctx->ws->sol);
	memset(ctx->ws->visited, 0, (size_t)ctx->s->visited_words
		* sizeof(uint64_t));
	memset(ctx->ws->meta_active, 0xFF, (size_t)ctx->s->meta_words
		* sizeof(uint64_t));
	memset(ctx->ws->route_loads, 0, (size_t)ctx->k * sizeof(int));
	v = 0;
	while (v < ctx->k)
	{
		if (!par_build_vehicle_route(ctx, v))
			return (false);
		v++;
	}
	return (true);
}

double	par_solution_cost(const t_solution *s, float **c)
{
	double			total;
	int				i;
	int				t;
	const t_route		*r;

	total = 0.0;
	i = 0;
	while (i < s->k)
	{
		r = &s->routes[i];
		if (r->len > 2)
		{
			t = 0;
			while (t + 1 < r->len)
			{
				total += (double)c[r->nodes[t]][r->nodes[t + 1]];
				t++;
			}
		}
		i++;
	}
	return (total);
}
