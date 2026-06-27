# include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "solution.h"
#include <math.h>

void	par_solver_evaporate(t_par_solver_ctx *ctx)
{
	float	rho_f;
	size_t	total_size;
	size_t	i;

	rho_f = (float)ctx->rho;
	total_size = (size_t)(ctx->n + 1) * ctx->tau_mat->stride;
#pragma omp for schedule(static)
	for (i = 0; i < total_size; i++)
	{
		ctx->tau_mat->data[i] *= (1.0f - rho_f);
	}
}

static void	par_deposit_route(t_par_solver_ctx *ctx, t_route *r,
		float weighted_dep)
{
	int	t;
	int	d_idx;

	t = 0;
	while (t + 1 < r->len)
	{
#pragma omp atomic
		ctx->tau_mat->data[(size_t)r->nodes[t] * ctx->tau_mat->stride + r->nodes[t + 1]] += weighted_dep;
#pragma omp atomic
		ctx->tau_mat->data[(size_t)r->nodes[t + 1] * ctx->tau_mat->stride + r->nodes[t]] += weighted_dep;
#pragma omp atomic capture
		d_idx = ctx->rank_delta_count++;
		ctx->rank_deltas[d_idx] = (t_par_sparse_delta){
			.edge_idx = (uint32_t)((size_t)r->nodes[t] * ctx->tau_mat->stride + r->nodes[t + 1]),
			.increment = weighted_dep
		};
#pragma omp atomic capture
		d_idx = ctx->rank_delta_count++;
		ctx->rank_deltas[d_idx] = (t_par_sparse_delta){
			.edge_idx = (uint32_t)((size_t)r->nodes[t + 1] * ctx->tau_mat->stride + r->nodes[t]),
			.increment = weighted_dep
		};
		t++;
	}
}

void	par_solver_deposit(t_par_solver_ctx *ctx)
{
	float	dep;
	float	weighted_dep;
	int		v;

	dep = (float)(ctx->q / fmax(*ctx->best_cost, 1e-9));
	weighted_dep = dep / (float)ctx->mpi_size;
#pragma omp for schedule(static)
	for (v = 0; v < ctx->k; v++)
	{
		if (ctx->best_sol->routes[v].len > 2)
			par_deposit_route(ctx, &ctx->best_sol->routes[v], weighted_dep);
	}
#pragma omp barrier
#pragma omp master
	{
#ifdef USE_MPI
		if (ctx->mpi_size > 1)
			par_async_start(&ctx->async_ctx, ctx->rank_deltas,
				ctx->rank_delta_count, ctx->mpi_size);
#endif
	}
#pragma omp barrier
}
