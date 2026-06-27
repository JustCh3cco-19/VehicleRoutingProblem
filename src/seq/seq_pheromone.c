#include "solver.h"
#include "seq/internal.h"
#include "solution.h"

void	evaporate_tau(t_seq_ctx *ctx)
{
	int	i;
	int	j;

	i = 0;
	while (i <= ctx->n)
	{
		j = 0;
		while (j <= ctx->n)
		{
			if (i != j)
				ctx->tau[i][j] *= (1.0 - ctx->rho);
			j++;
		}
		i++;
	}
}

void	deposit_iter_best(t_seq_ctx *ctx)
{
	int		i;
	int		j;
	double	dep;
	t_route	*r;

	dep = (0.3 * ctx->q) / ctx->iter_best_cost;
	i = 0;
	while (i < ctx->k)
	{
		r = &ctx->iter_best->routes[i];
		j = 0;
		while (j + 1 < r->len)
		{
			ctx->tau[r->nodes[j]][r->nodes[j + 1]] += dep;
			ctx->tau[r->nodes[j + 1]][r->nodes[j]] += dep;
			j++;
		}
		i++;
	}
}

void	deposit_global_best(t_seq_ctx *ctx)
{
	int		i;
	int		j;
	double	dep;
	t_route	*r;

	dep = (0.7 * ctx->q) / (*ctx->best_cost);
	i = 0;
	while (i < ctx->k)
	{
		r = &ctx->best_sol->routes[i];
		j = 0;
		while (j + 1 < r->len)
		{
			ctx->tau[r->nodes[j]][r->nodes[j + 1]] += dep;
			ctx->tau[r->nodes[j + 1]][r->nodes[j]] += dep;
			j++;
		}
		i++;
	}
}

void	clamp_pheromones(t_seq_ctx *ctx)
{
	int	i;
	int	j;

	i = 0;
	while (i <= ctx->n)
	{
		j = 0;
		while (j <= ctx->n)
		{
			if (i != j)
			{
				if (ctx->tau[i][j] < ctx->tau_min)
					ctx->tau[i][j] = ctx->tau_min;
				else if (ctx->tau[i][j] > ctx->tau_max)
					ctx->tau[i][j] = ctx->tau_max;
			}
			j++;
		}
		i++;
	}
}

void	reset_pheromones(t_seq_ctx *ctx)
{
	int	i;
	int	j;

	i = 0;
	while (i <= ctx->n)
	{
		j = 0;
		while (j <= ctx->n)
		{
			if (i != j)
				ctx->tau[i][j] = 0.5 * ctx->tau[i][j] + 0.5 * ctx->tau0;
			j++;
		}
		i++;
	}
}
