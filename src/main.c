#include "solver.h"
#include "cli_common.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"

#include <math.h>
#include <stdio.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

static void	fill_example_costs(double **c, int n)
{
	int	i;
	int	j;

	i = 0;
	for (; i <= n; i++)
	{
		j = 0;
		for (; j <= n; j++)
		{
			if (i == j)
				c[i][j] = 0.0;
			else
				c[i][j] = 1.0 + fabs((double)i - (double)j);
		}
	}
}

static int	load_costs(t_cli_options *options, t_vrp_instance *instance,
		double ***c)
{
	if (options->mode == CLI_MODE_INSTANCE)
	{
		if (vrp_load_tsplib_instance(options->instance_path, instance) != 0)
			return (0);
		if (vrp_instance_create_euc2d_matrix(instance, c) != 0)
			return (0);
		options->n = instance->n;
		return (1);
	}
	*c = matrix_alloc(options->n);
	if (!*c)
		return (0);
	fill_example_costs(*c, options->n);
	return (1);
}

static void	fill_solver_params(t_solver_params *params,
		t_cli_options *options, double **c)
{
	params->n = options->n;
	params->k = options->k;
	params->m = options->m;
	params->vehicle_capacity_customers = 0;
	params->c = c;
	params->alpha = options->alpha;
	params->beta = options->beta;
	params->rho = options->rho;
	params->tau0 = options->tau0;
	params->q = options->q;
	params->seed = options->seed;
}

static t_status	run_solver(t_solver_params *params, t_cli_options *options,
		t_vrp_instance *instance, t_solution *best)
{
	if (options->mode == CLI_MODE_INSTANCE)
	{
		params->vehicle_capacity_customers = instance->capacity;
		return (vrp_solve_with_capacity(params, best, &options->best_cost));
	}
	return (vrp_solve(params, best, &options->best_cost));
}

static int	print_result(t_status solver_status, t_cli_options *options,
		t_vrp_instance *instance, t_solution *best)
{
	t_cli_validation	validation;

	if (solver_status != SOLVER_OK)
	{
		fprintf(stderr, "solver failed: %s\n", status_string(solver_status));
		return (0);
	}
	validation.best = best;
	validation.n = options->n;
	validation.k = options->k;
	validation.demands = instance->demands;
	validation.vehicle_capacity = instance->capacity;
	validation.best_cost = options->best_cost;
	if (!cli_validate_solution_or_report(&validation))
		return (0);
	cli_print_solution_routes(best, options->k);
	cli_print_solution_cost(options->best_cost);
	return (1);
}

#ifdef USE_MPI
static int	init_mpi(int *mpi_rank)
{
	int	provided;

	provided = 0;
	if (MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided)
		!= MPI_SUCCESS)
	{
		fprintf(stderr, "MPI_Init_thread failed\n");
		return (0);
	}
	MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank);
	return (1);
}
#endif

static int	prepare_run(t_cli_options *options, t_vrp_instance *instance,
		double ***c, t_solution **best)
{
	if (!load_costs(options, instance, c))
		return (0);
	*best = solution_create(options->k, options->n);
	if (!*best)
		return (0);
	if (options->mode == CLI_MODE_INSTANCE && instance->vehicles > 0
		&& instance->vehicles != options->k)
	{
		fprintf(stderr, "instance VEHICLES mismatch: CLI k=%d, file %d\n",
			options->k, instance->vehicles);
		return (0);
	}
	return (1);
}

int	main(int argc, char **argv)
{
	t_cli_options	options;
	t_vrp_instance	instance;
	t_solver_params	params;
	t_solution		*best;
	double			**c;
	int				status;
#ifdef USE_MPI
	int				mpi_rank;
#endif

	status = 0;
	best = NULL;
	c = NULL;
#ifdef USE_MPI
	mpi_rank = 0;
	if (!init_mpi(&mpi_rank))
		return (1);
#endif
	vrp_instance_init(&instance);
	cli_options_defaults(&options);
	if (!cli_parse_solver_options(argc, argv, &options))
		status = 1;
	else if (!prepare_run(&options, &instance, &c, &best))
		status = 1;
	else
	{
		fill_solver_params(&params, &options, c);
		if (!print_result(run_solver(&params, &options, &instance, best),
				&options, &instance, best))
			status = 1;
	}
	if (status == 1)
		cli_print_usage(argv[0]);
	solution_free(best);
	matrix_free(c);
	vrp_instance_free(&instance);
#ifdef USE_MPI
	MPI_Finalize();
#endif
	return (status);
}
