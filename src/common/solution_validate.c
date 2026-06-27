#include "solution.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void	set_err(char *err, size_t err_len, const char *msg)
{
	if (!err || err_len == 0)
		return ;
	snprintf(err, err_len, "%s", msg);
}

static bool	validate_header(t_solution_validation *val)
{
	if (!val->s)
	{
		set_err(val->err, val->err_len, "solution is NULL");
		return (false);
	}
	if (val->s->k != val->k || !val->s->routes || val->n < 0 || val->k < 0)
	{
		set_err(val->err, val->err_len, "invalid solution metadata");
		return (false);
	}
	return (true);
}

static bool	validate_node(int node, int t, const t_route *r, bool *seen)
{
	if (node == 0 && t != 0 && t != r->len - 1)
		return (false);
	if (node != 0 && seen[node])
		return (false);
	if (node != 0)
		seen[node] = true;
	return (true);
}

static bool	validate_route(t_solution_validation *val, const t_route *r,
		bool *seen)
{
	int	node;
	int	t;

	if (r->len < 2 || r->nodes[0] != 0 || r->nodes[r->len - 1] != 0)
	{
		set_err(val->err, val->err_len, "route must start/end at depot 0");
		return (false);
	}
	t = 0;
	for (; t < r->len; t++)
	{
		node = r->nodes[t];
		if (node < 0 || node > val->n)
		{
			set_err(val->err, val->err_len, "route node out of range");
			return (false);
		}
		if (!validate_node(node, t, r, seen))
		{
			set_err(val->err, val->err_len, "invalid or repeated route node");
			return (false);
		}
	}
	return (true);
}

static bool	validate_all_seen(bool *seen, int n, char *err, size_t err_len)
{
	int	node;

	node = 1;
	for (; node <= n; node++)
	{
		if (!seen[node])
		{
			set_err(err, err_len, "customer not visited");
			return (false);
		}
	}
	return (true);
}

bool	solution_validate(t_solution_validation *val)
{
	bool	*seen;
	int		i;
	bool	valid;

	set_err(val->err, val->err_len, "");
	if (!validate_header(val))
		return (false);
	seen = (bool *)calloc((size_t)(val->n + 1), sizeof(bool));
	if (!seen)
	{
		set_err(val->err, val->err_len, "solution validation allocation");
		return (false);
	}
	valid = true;
	i = 0;
	for (; valid && i < val->s->k; i++)
		valid = validate_route(val, &val->s->routes[i], seen);
	if (valid)
		valid = validate_all_seen(seen, val->n, val->err, val->err_len);
	free(seen);
	return (valid);
}
