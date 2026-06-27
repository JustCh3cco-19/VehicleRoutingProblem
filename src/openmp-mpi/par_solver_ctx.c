# include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include "solution.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
# include <omp.h>
#endif

static void	par_populate_mats(t_par_solver_ctx *ctx)
{
	int	i;
	int	j;

	i = 0;
	while (i <= ctx->n)
	{
		j = 0;
		while (j <= ctx->n)
		{
			ctx->c_mat->rows[i][j] = (float)ctx->c[i][j];
			if (i == j)
				ctx->tau_mat->rows[i][j] = 0.0f;
			else
				ctx->tau_mat->rows[i][j] = (float)ctx->tau0;
			j++;
		}
		i++;
	}
}

int	par_solver_alloc(t_par_solver_ctx *ctx)
{
	ctx->tau_mat = par_matrix_create(ctx->n);
	ctx->c_mat = par_matrix_create(ctx->n);
	if (!ctx->tau_mat || !ctx->c_mat)
	{
		par_matrix_free(ctx->tau_mat);
		par_matrix_free(ctx->c_mat);
		return (0);
	}
	par_populate_mats(ctx);
	if (!par_shared_init(&ctx->shared, ctx->n, ctx->cand_k, ctx->c_mat,
			ctx->beta))
	{
		par_matrix_free(ctx->tau_mat);
		par_matrix_free(ctx->c_mat);
		return (0);
	}
	return (1);
}

int	par_solver_init(t_par_solver_ctx *ctx)
{
	size_t	score_count;
	size_t	score_bytes;
	size_t	rank_delta_capacity;
	size_t	rank_delta_bytes;

	score_count = 0;
	score_bytes = 0;
	if (!matrix_mul_size((size_t)(ctx->n + 1), (size_t)ctx->shared.stride,
			&score_count) || !matrix_mul_size(score_count, sizeof(float),
			&score_bytes))
		return (0);
	ctx->score_mat = par_aligned_calloc(score_bytes);
	ctx->iter_best_sol_rank = solution_create(ctx->k, ctx->n);
	ctx->iter_best_cost_g = DBL_MAX;
	*ctx->best_cost = DBL_MAX;
	ctx->start_time = par_wall_time();
	if (ctx->progress_interval_sec > 0.0)
		ctx->next_progress_time = ctx->start_time + ctx->progress_interval_sec;
	else
		ctx->next_progress_time = 0.0;
	if (!ctx->score_mat || !ctx->iter_best_sol_rank)
		return (0);
#ifdef USE_MPI
	par_async_init(&ctx->async_ctx, ctx->mpi_size);
#endif
	ctx->iter_since_best = 0;
	rank_delta_capacity = 0;
	rank_delta_bytes = 0;
	if (!matrix_mul_size((size_t)ctx->n + (size_t)ctx->k + 500u, 2u,
			&rank_delta_capacity) || !matrix_mul_size(rank_delta_capacity,
			sizeof(t_par_sparse_delta), &rank_delta_bytes))
		return (0);
	ctx->rank_deltas = malloc(rank_delta_bytes);
	if (!ctx->rank_deltas)
		return (0);
	ctx->workspace_failed = 0;
	return (1);
}

void	par_solver_free(t_par_solver_ctx *ctx)
{
	free(ctx->rank_deltas);
#ifdef USE_MPI
	par_async_cleanup(&ctx->async_ctx);
#endif
	free(ctx->score_mat);
	solution_free(ctx->iter_best_sol_rank);
	par_shared_free(&ctx->shared);
	par_matrix_free(ctx->tau_mat);
	par_matrix_free(ctx->c_mat);
}

void	par_solver_log_start(t_par_solver_ctx *ctx)
{
#ifdef _OPENMP
	if (ctx->log_level > LOG_SILENT && ctx->mpi_rank == 0
		&& omp_get_thread_num() == 0 && !ctx->workspace_failed)
	{
		fprintf(stderr,
			"ACO Parallel Starting with %d threads. N=%d k=%d Cap=%d candidate_k=%d seed=%u\n",
			omp_get_num_threads(), ctx->n, ctx->k, ctx->cap, ctx->cand_k, ctx->seed);
	}
#else
	if (ctx->log_level > LOG_SILENT && ctx->mpi_rank == 0
		&& !ctx->workspace_failed)
	{
		fprintf(stderr,
			"ACO Parallel Starting. N=%d k=%d Cap=%d candidate_k=%d seed=%u\n",
			ctx->n, ctx->k, ctx->cap, ctx->cand_k, ctx->seed);
	}
#endif
}
