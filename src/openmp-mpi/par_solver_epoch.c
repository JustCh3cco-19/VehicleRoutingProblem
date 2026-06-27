#include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "solution.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

static void	par_solver_score_row(t_par_solver_ctx *ctx, int i);
static void	par_reduce_best_master(t_par_solver_ctx *ctx, int iter);
static void	par_reduce_min(t_par_solver_ctx *ctx, double *g_min);

void	par_solver_scores(t_par_solver_ctx *ctx)
{
	int			i;

#pragma omp for schedule(static) nowait
	for (i = 0; i <= ctx->n; i++)
		par_solver_score_row(ctx, i);
#pragma omp barrier
}

static void	par_solver_score_row(t_par_solver_ctx *ctx, int i)
{
	int			*cands;
	float		*etas;
	float		*sc;
	const float	*tau_row;
	int			t;
	int			node;

	cands = ctx->shared.cand_idx + (size_t)i * (size_t)ctx->shared.stride;
	etas = ctx->shared.eta_beta + (size_t)i * (size_t)ctx->shared.stride;
	sc = ctx->score_mat + (size_t)i * (size_t)ctx->shared.stride;
	tau_row = ctx->tau_mat->rows[i];
	t = 0;
	while (t < ctx->shared.cand_k)
	{
		node = cands[t];
		if (node > 0)
			sc[t] = par_fast_powf(tau_row[node], (float)ctx->alpha)
				* etas[t];
		else
			sc[t] = 0.0f;
		t++;
	}
}

void	par_solver_ants(t_par_solver_ctx *ctx, t_par_workspace *ws, int iter)
{
	double	t_best_c;
	int		a;
	double	cost;

	t_best_c = DBL_MAX;
#pragma omp for schedule(runtime) nowait
	for (a = 0; a < ctx->local_m; a++)
	{
		t_par_tour_ctx	tour_ctx;

		ws->rng_state = make_ant_seed(ctx->seed, iter, ctx->ant_off + a);
		tour_ctx.ws = ws;
		tour_ctx.s = &ctx->shared;
		tour_ctx.k = ctx->k;
		tour_ctx.cap = ctx->cap;
		tour_ctx.c = ctx->c_mat;
		tour_ctx.scores = ctx->score_mat;
		tour_ctx.remaining = ctx->shared.n;
		if (par_build_ant(&tour_ctx))
		{
			cost = par_solution_cost(ws->sol, ctx->c_mat->rows);
			if (cost < t_best_c)
			{
				t_best_c = cost;
				solution_copy(ws->thread_best, ws->sol);
			}
		}
	}
#pragma omp critical
	{
		if (t_best_c < ctx->iter_best_cost_g)
		{
			ctx->iter_best_cost_g = t_best_c;
			solution_copy(ctx->iter_best_sol_rank, ws->thread_best);
		}
	}
#pragma omp barrier
}

static void	par_log_iter(t_par_solver_ctx *ctx, int iter)
{
	double	now;
	double	elapsed;
	double	remaining;

	now = par_wall_time();
	if (ctx->log_level > LOG_SILENT && ctx->mpi_rank == 0
		&& ctx->progress_interval_sec > 0.0 && now >= ctx->next_progress_time)
	{
		elapsed = now - ctx->start_time;
		if (ctx->max_runtime_sec > 0.0)
		{
			remaining = ctx->max_runtime_sec - elapsed;
			if (remaining < 0.0)
				remaining = 0.0;
			fprintf(stderr,
				"[mpi] elapsed %.1fs, remaining %.1fs, iter %d, best %.3f\n",
				elapsed, remaining, iter + 1, *ctx->best_cost);
		}
		else
		{
			fprintf(stderr, "[mpi] elapsed %.1fs, iter %d, best %.3f\n",
				elapsed, iter + 1, *ctx->best_cost);
		}
		ctx->next_progress_time = now + ctx->progress_interval_sec;
	}
}

void	par_solver_reduce_best(t_par_solver_ctx *ctx, int iter)
{
#pragma omp master
	par_reduce_best_master(ctx, iter);
#pragma omp barrier
}

static void	par_reduce_best_master(t_par_solver_ctx *ctx, int iter)
{
	double	g_min;

	g_min = ctx->iter_best_cost_g;
	par_reduce_min(ctx, &g_min);
	if (par_is_improvement(*ctx->best_cost, g_min,
			ctx->min_rel_improvement))
		ctx->iter_since_best = 0;
	else
		ctx->iter_since_best++;
	if (g_min < *ctx->best_cost)
	{
		*ctx->best_cost = g_min;
		solution_copy(ctx->best_sol, ctx->iter_best_sol_rank);
	}
	par_log_iter(ctx, iter);
	ctx->iter_best_cost_g = DBL_MAX;
	ctx->rank_delta_count = 0;
}

static void	par_reduce_min(t_par_solver_ctx *ctx, double *g_min)
{
#ifdef USE_MPI
	if (ctx->mpi_size > 1)
		MPI_Allreduce(MPI_IN_PLACE, g_min, 1, MPI_DOUBLE, MPI_MIN,
			MPI_COMM_WORLD);
#else
	(void)ctx;
	(void)g_min;
#endif
}
