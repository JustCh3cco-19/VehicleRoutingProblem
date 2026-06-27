#include "instance_parser.h"
#include "matrix.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double	euc_2d(double x1, double y1, double x2, double y2)
{
	double	dx;
	double	dy;

	dx = x1 - x2;
	dy = y1 - y2;
	return (sqrt(dx * dx + dy * dy));
}

void	vrp_instance_init(t_vrp_instance *instance)
{
	if (!instance)
		return ;
	memset(instance, 0, sizeof(*instance));
}

void	vrp_instance_free(t_vrp_instance *instance)
{
	if (!instance)
		return ;
	free(instance->coords_x);
	free(instance->coords_y);
	free(instance->demands);
	vrp_instance_init(instance);
}

static int	parse_int_field(char *line, const char *name)
{
	char	*colon;
	int		value;

	value = 0;
	colon = strchr(line, ':');
	if (colon)
		return (atoi(colon + 1));
	if (sscanf(line, "%*s %d", &value) != 1)
		return (0);
	(void)name;
	return (value);
}

static int	read_coords(t_vrp_parse_state *state)
{
	int	id;
	int	i;

	if (state->n <= 0)
		return (0);
	state->instance->coords_x = malloc((size_t)(state->n + 1)
			* sizeof(double));
	state->instance->coords_y = malloc((size_t)(state->n + 1)
			* sizeof(double));
	if (!state->instance->coords_x || !state->instance->coords_y)
		return (0);
	i = 0;
	for (; i <= state->n; i++)
	{
		if (fscanf(state->f, "%d %lf %lf", &id,
				&state->instance->coords_x[i],
				&state->instance->coords_y[i]) != 3)
			return (0);
	}
	state->have_coords = 1;
	return (1);
}

static int	read_demands(t_vrp_parse_state *state)
{
	int	id;
	int	i;

	if (state->n <= 0)
		return (0);
	state->instance->demands = malloc((size_t)(state->n + 1) * sizeof(int));
	if (!state->instance->demands)
		return (0);
	i = 0;
	for (; i <= state->n; i++)
	{
		if (fscanf(state->f, "%d %d", &id,
				&state->instance->demands[i]) != 2)
			return (0);
	}
	state->have_demands = 1;
	return (1);
}

static int	parse_line(t_vrp_parse_state *state, char *line)
{
	if (strncmp(line, "DIMENSION", 9) == 0)
		state->n = parse_int_field(line, "DIMENSION") - 1;
	else if (strncmp(line, "VEHICLES", 8) == 0)
	{
		state->vehicles = parse_int_field(line, "VEHICLES");
		state->have_vehicles = (state->vehicles > 0);
	}
	else if (strncmp(line, "CAPACITY", 8) == 0)
	{
		state->capacity = parse_int_field(line, "CAPACITY");
		state->have_capacity = (state->capacity > 0);
	}
	else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0)
		return (read_coords(state));
	else if (strncmp(line, "DEMAND_SECTION", 14) == 0)
		return (read_demands(state));
	return (1);
}

static int	validate_demands(t_vrp_parse_state *state)
{
	int	i;

	if (state->instance->demands[0] != 0)
	{
		fprintf(stderr, "Depot demand must be 0 in %s\n", state->path);
		return (0);
	}
	i = 1;
	for (; i <= state->n; i++)
	{
		if (state->instance->demands[i] != 1)
		{
			fprintf(stderr, "Customer demand must be 1 in %s\n",
				state->path);
			return (0);
		}
	}
	return (1);
}

static int	validate_instance(t_vrp_parse_state *state)
{
	if (state->n <= 0 || !state->have_coords || !state->instance->coords_x
		|| !state->instance->coords_y)
	{
		fprintf(stderr, "Invalid or missing DIMENSION/NODE_COORD_SECTION\n");
		return (0);
	}
	if (!state->have_vehicles || !state->have_capacity
		|| !state->have_demands)
	{
		fprintf(stderr, "Missing VEHICLES/CAPACITY/DEMAND_SECTION in %s\n",
			state->path);
		return (0);
	}
	if (state->capacity <= 0)
	{
		fprintf(stderr, "CAPACITY must be positive in %s\n", state->path);
		return (0);
	}
	return (validate_demands(state));
}

static void	commit_instance(t_vrp_parse_state *state)
{
	state->instance->n = state->n;
	state->instance->vehicles = state->vehicles;
	state->instance->capacity = state->capacity;
}

static int	finish_load(t_vrp_parse_state *state, int ok)
{
	if (ok)
		ok = validate_instance(state);
	if (ok)
		commit_instance(state);
	fclose(state->f);
	if (!ok)
		vrp_instance_free(state->instance);
	return (!ok);
}

int	vrp_load_tsplib_instance(const char *path, t_vrp_instance *instance)
{
	t_vrp_parse_state	state;
	char				line[256];
	int					ok;

	if (!path || !instance)
		return (1);
	vrp_instance_init(instance);
	memset(&state, 0, sizeof(state));
	state.path = path;
	state.instance = instance;
	state.f = fopen(path, "r");
	if (!state.f)
	{
		perror("fopen");
		return (1);
	}
	ok = 1;
	while (ok && fgets(line, sizeof(line), state.f))
		ok = parse_line(&state, line);
	return (finish_load(&state, ok));
}

int	vrp_instance_create_euc2d_matrix(const t_vrp_instance *instance,
		double ***c_out)
{
	double	**c;
	int		n;
	int		i;
	int		j;

	if (!instance || !c_out || instance->n <= 0 || !instance->coords_x
		|| !instance->coords_y)
		return (1);
	n = instance->n;
	c = matrix_alloc(n);
	if (!c)
		return (1);
	i = 0;
	for (; i <= n; i++)
	{
		j = 0;
		for (; j <= n; j++)
		{
			c[i][j] = euc_2d(instance->coords_x[i], instance->coords_y[i],
					instance->coords_x[j], instance->coords_y[j]);
		}
	}
	*c_out = c;
	return (0);
}

int	vrp_instance_create_float_coords(const t_vrp_instance *instance,
		float **coords_x_out, float **coords_y_out)
{
	float	*coords_x;
	float	*coords_y;
	int		i;

	if (!instance || !coords_x_out || !coords_y_out || instance->n <= 0
		|| !instance->coords_x || !instance->coords_y)
		return (1);
	coords_x = malloc((size_t)(instance->n + 1) * sizeof(float));
	coords_y = malloc((size_t)(instance->n + 1) * sizeof(float));
	if (!coords_x || !coords_y)
	{
		free(coords_x);
		free(coords_y);
		return (1);
	}
	i = 0;
	for (; i <= instance->n; i++)
	{
		coords_x[i] = (float)instance->coords_x[i];
		coords_y[i] = (float)instance->coords_y[i];
	}
	*coords_x_out = coords_x;
	*coords_y_out = coords_y;
	return (0);
}
