extern "C" {
#include "solver.h"
#include "cli_common.h"
#include "instance_parser.h"
#include "solution.h"
}

#include "cuda/cuda_context.h"
#include <stdio.h>
#include <stdlib.h>

t_status	vrp_solve_cuda(t_solver_params *params,
				t_cuda_coords coords,
				t_solution *best_solution,
				double *best_cost);



static int	prepare_cuda_run(t_cuda_main_ctx *ctx)
{
	if (vrp_load_tsplib_instance(ctx->options.instance_path,
			&ctx->instance) != 0)
		return (0);
	if (vrp_instance_create_float_coords(&ctx->instance, &ctx->coords_x,
			&ctx->coords_y) != 0)
		return (0);
	ctx->options.n = ctx->instance.n;
	if (ctx->instance.vehicles > 0 && ctx->instance.vehicles != ctx->options.k)
	{
		fprintf(stderr,
			"instance VEHICLES mismatch: CLI k=%d, file VEHICLES=%d\n",
			ctx->options.k, ctx->instance.vehicles);
		return (0);
	}
	ctx->best = solution_create(ctx->options.k, ctx->options.n);
	if (!ctx->best)
	{
		fprintf(stderr, "failed to allocate solution\n");
		return (0);
	}
	return (1);
}

static void	fill_solver_params(t_solver_params *params, t_cuda_main_ctx *ctx)
{
	params->n = ctx->options.n;
	params->k = ctx->options.k;
	params->m = ctx->options.m;
	params->vehicle_capacity_customers = ctx->instance.capacity;
	params->c = NULL;
	params->alpha = ctx->options.alpha;
	params->beta = ctx->options.beta;
	params->rho = ctx->options.rho;
	params->tau0 = ctx->options.tau0;
	params->q = ctx->options.q;
	params->seed = ctx->options.seed;
}

static int	validate_and_print(t_cuda_main_ctx *ctx)
{
	t_cli_validation	val;

	val.best = ctx->best;
	val.n = ctx->options.n;
	val.k = ctx->options.k;
	val.demands = ctx->instance.demands;
	val.vehicle_capacity = ctx->instance.capacity;
	val.best_cost = ctx->best_cost;
	if (!cli_validate_solution_or_report(&val))
		return (0);
	cli_print_solution_routes(ctx->best, ctx->options.k);
	cli_print_solution_cost(ctx->best_cost);
	return (1);
}

static int	solve_and_validate(t_cuda_main_ctx *ctx, t_solver_params *params,
				t_cuda_coords coords)
{
	if (vrp_solve_cuda(params, coords, ctx->best, &ctx->best_cost)
		!= SOLVER_OK)
		return (0);
	return (validate_and_print(ctx));
}

static int	parse_options(int argc, char **argv, t_cuda_main_ctx *ctx)
{
	cli_options_defaults(&ctx->options);
	if (!cli_parse_solver_options(argc, argv, &ctx->options)
		|| ctx->options.mode != CLI_MODE_INSTANCE)
	{
		cli_print_usage(argv[0]);
		return (0);
	}
	return (1);
}

int	main(int argc, char **argv)
{
	t_cuda_main_ctx	ctx;
	t_solver_params	params;
	t_cuda_coords	coords;
	int				status;

	status = 0;
	ctx.coords_x = NULL;
	ctx.coords_y = NULL;
	ctx.best = NULL;
	ctx.best_cost = 0.0;
	vrp_instance_init(&ctx.instance);
	if (!parse_options(argc, argv, &ctx))
		status = 1;
	else if (!prepare_cuda_run(&ctx))
		status = 1;
	else
	{
		fill_solver_params(&params, &ctx);
		coords.x = ctx.coords_x;
		coords.y = ctx.coords_y;
		if (!solve_and_validate(&ctx, &params, coords))
			status = 1;
	}
	solution_free(ctx.best);
	vrp_instance_free(&ctx.instance);
	free(ctx.coords_x);
	free(ctx.coords_y);
	return (status);
}
