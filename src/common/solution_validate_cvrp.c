#include "solution.h"

#include <stdbool.h>
#include <stdio.h>

static void	set_err(char *err, size_t err_len, const char *msg)
{
	if (!err || err_len == 0)
		return ;
	snprintf(err, err_len, "%s", msg);
}

static bool	validate_capacity_route(t_solution_validation *val,
		const t_route *r)
{
	int	load;
	int	node;
	int	t;

	load = 0;
	t = 1;
	for (; t + 1 < r->len; t++)
	{
		node = r->nodes[t];
		if (val->demands[node] < 0)
		{
			set_err(val->err, val->err_len, "negative route demand");
			return (false);
		}
		load += val->demands[node];
		if (load > val->vehicle_capacity)
		{
			set_err(val->err, val->err_len, "route exceeds capacity");
			return (false);
		}
	}
	return (true);
}

bool	solution_validate_cvrp(t_solution_validation *validation)
{
	int	i;

	if (!validation)
		return (false);
	if (!solution_validate(validation))
		return (false);
	if (!validation->demands || validation->vehicle_capacity <= 0
		|| validation->demands[0] != 0)
	{
		set_err(validation->err, validation->err_len,
			"invalid capacity validation input");
		return (false);
	}
	i = 0;
	for (; i < validation->s->k; i++)
	{
		if (!validate_capacity_route(validation, &validation->s->routes[i]))
			return (false);
	}
	set_err(validation->err, validation->err_len, "");
	return (true);
}
