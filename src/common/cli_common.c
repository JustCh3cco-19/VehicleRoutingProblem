#include "cli_common.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int	cli_parse_double_arg(const char *s, double *out)
{
	char	*end;
	double	v;

	end = NULL;
	if (!s || !out)
		return (0);
	errno = 0;
	v = strtod(s, &end);
	if (errno != 0 || end == s || *end != '\0' || !isfinite(v))
		return (0);
	*out = v;
	return (1);
}

int	cli_parse_int_arg(const char *s, int *out)
{
	char	*end;
	long	v;

	end = NULL;
	if (!s || !out)
		return (0);
	errno = 0;
	v = strtol(s, &end, 10);
	if (errno != 0 || end == s || *end != '\0')
		return (0);
	if (v < 0 || v > 100000000L)
		return (0);
	*out = (int)v;
	return (1);
}

unsigned int	cli_parse_uint_arg(const char *s, int *ok)
{
	char			*end;
	unsigned long	v;

	end = NULL;
	if (!s || !ok)
	{
		if (ok)
			*ok = 0;
		return (0u);
	}
	errno = 0;
	v = strtoul(s, &end, 10);
	if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFUL)
	{
		*ok = 0;
		return (0u);
	}
	*ok = 1;
	return ((unsigned int)v);
}

void	cli_options_defaults(t_cli_options *options)
{
	if (!options)
		return ;
	options->mode = CLI_MODE_DEMO;
	options->instance_path = NULL;
	options->n = 5;
	options->k = 2;
	options->m = 10;
	options->seed = 1234u;
	options->alpha = 1.0;
	options->beta = 2.0;
	options->rho = 0.5;
	options->tau0 = 1.0;
	options->q = 1.0;
	options->best_cost = 0.0;
}

static int	cli_set_numeric_option(char *name, double value,
		t_cli_options *options)
{
	if (strcmp(name, "--alpha") == 0)
		options->alpha = value;
	else if (strcmp(name, "--beta") == 0)
		options->beta = value;
	else if (strcmp(name, "--rho") == 0)
		options->rho = value;
	else if (strcmp(name, "--tau0") == 0)
		options->tau0 = value;
	else if (strcmp(name, "--q") == 0)
		options->q = value;
	else
		return (0);
	return (1);
}

static int	cli_numeric_values_ok(t_cli_options *options)
{
	return (options->alpha > 0.0 && options->beta > 0.0
		&& options->rho > 0.0 && options->rho < 1.0
		&& options->tau0 > 0.0 && options->q > 0.0);
}

static int	cli_parse_numeric_options(int argc, char **argv, int start,
		t_cli_options *options)
{
	double	value;
	int		i;

	i = start;
	for (; i < argc; i += 2)
	{
		if (i + 1 >= argc)
			return (0);
		value = 0.0;
		if (!cli_parse_double_arg(argv[i + 1], &value))
			return (0);
		if (!cli_set_numeric_option(argv[i], value, options))
			return (0);
	}
	return (cli_numeric_values_ok(options));
}

static int	cli_parse_demo(int argc, char **argv, t_cli_options *options,
		int *first_value)
{
	int	ok;

	ok = 1;
	options->mode = CLI_MODE_DEMO;
	if (argc < 5)
		return (0);
	ok = ok && cli_parse_int_arg(argv[2], &options->k);
	ok = ok && cli_parse_int_arg(argv[3], &options->m);
	options->seed = cli_parse_uint_arg(argv[4], &ok);
	*first_value = 5;
	if (!ok || options->n <= 0 || options->k <= 0 || options->m < 0)
		return (0);
	return (1);
}

static int	cli_parse_instance(int argc, char **argv, t_cli_options *options,
		int *first_value)
{
	int	ok;

	ok = 1;
	options->mode = CLI_MODE_INSTANCE;
	if (argc < 4)
		return (0);
	options->instance_path = argv[1];
	ok = ok && cli_parse_int_arg(argv[2], &options->k);
	ok = ok && cli_parse_int_arg(argv[3], &options->m);
	*first_value = 4;
	if (argc > 4 && argv[4][0] != '-')
	{
		options->seed = cli_parse_uint_arg(argv[4], &ok);
		*first_value = 5;
	}
	if (!ok || options->k <= 0 || options->m < 0)
		return (0);
	return (1);
}

int	cli_parse_solver_options(int argc, char **argv, t_cli_options *options)
{
	int	first_value;

	first_value = 1;
	if (!options)
		return (0);
	cli_options_defaults(options);
	if (argc == 1)
		return (1);
	if (cli_parse_int_arg(argv[1], &options->n))
	{
		if (!cli_parse_demo(argc, argv, options, &first_value))
			return (0);
	}
	else if (!cli_parse_instance(argc, argv, options, &first_value))
		return (0);
	return (cli_parse_numeric_options(argc, argv, first_value, options));
}

void	cli_print_usage(const char *program_name)
{
	fprintf(stderr, "usage: %s [n k m seed] [--alpha v --beta v --rho v "
		"--tau0 v --q v]\n", program_name);
	fprintf(stderr, "usage: %s <instance.vrp> <k> <m> [seed] "
		"[--alpha v --beta v --rho v --tau0 v --q v]\n", program_name);
}

void	cli_print_solution_routes(const t_solution *best, int k)
{
	const t_route	*r;
	int				printed;
	int				node;
	int				i;
	int				t;

	i = 0;
	for (; i < k; i++)
	{
		r = &best->routes[i];
		printed = 0;
		printf("Route %d:", i + 1);
		t = 0;
		for (; t < r->len; t++)
		{
			node = r->nodes[t];
			if (node != 0)
			{
				printf("%s%d", printed ? " " : " ", node);
				printed = 1;
			}
		}
		printf("\n");
	}
}

void	cli_print_solution_cost(double best_cost)
{
	printf("Cost: %.3f\n", best_cost);
	printf("best cost: %.3f\n", best_cost);
}

static void	fill_solution_validation(t_solution_validation *dst,
		t_cli_validation *src, char *err)
{
	dst->s = src->best;
	dst->n = src->n;
	dst->k = src->k;
	dst->demands = src->demands;
	dst->vehicle_capacity = src->vehicle_capacity;
	dst->err = err;
	dst->err_len = 160;
}

static int	cli_solution_valid(t_cli_validation *validation)
{
	t_solution_validation	sol_validation;
	char					err[160];
	int						valid;

	if (validation->demands)
	{
		fill_solution_validation(&sol_validation, validation, err);
		valid = solution_validate_cvrp(&sol_validation);
	}
	else
	{
		sol_validation.s = validation->best;
		sol_validation.n = validation->n;
		sol_validation.k = validation->k;
		sol_validation.demands = NULL;
		sol_validation.vehicle_capacity = 0;
		sol_validation.err = err;
		sol_validation.err_len = sizeof(err);
		valid = solution_validate(&sol_validation);
	}
	if (!valid)
		fprintf(stderr, "invalid solution: %s\n", err);
	return (valid);
}

int	cli_validate_solution_or_report(t_cli_validation *validation)
{
	if (!validation)
		return (0);
	if (!isfinite(validation->best_cost) || validation->best_cost <= 0.0)
	{
		fprintf(stderr, "invalid solution cost: %.12g\n",
			validation->best_cost);
		return (0);
	}
	return (cli_solution_valid(validation));
}
