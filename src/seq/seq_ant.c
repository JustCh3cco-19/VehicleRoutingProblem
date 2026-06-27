#include "solver.h"
#include "seq/internal.h"
#include "solution.h"
#include <math.h>

void	run_ant(t_seq_ctx *ctx, int ant)
{
	double	cost;

	ctx->ws.rng_state = make_ant_seed(ctx->seed, ctx->iter, ant);
	if (!build_ant_solution(ctx))
		return ;
	cost = solution_cost(ctx->ws.sol, ctx->c);
	if (cost < ctx->iter_best_cost || (fabs(cost - ctx->iter_best_cost)
			<= SOLVER_EPS && ant < ctx->iter_best_ant))
	{
		ctx->iter_best_cost = cost;
		ctx->iter_best_ant = ant;
		solution_copy(ctx->iter_best, ctx->ws.sol);
	}
}
