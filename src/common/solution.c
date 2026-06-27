#include "solution.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOLUTION_ALIGNMENT 64u

static size_t	align_up_size(size_t value, size_t alignment)
{
	size_t	rem;

	rem = value % alignment;
	if (rem == 0u)
		return (value);
	return (value + (alignment - rem));
}

bool	route_append(t_route *r, int node)
{
	if (r->len >= r->cap)
		return (false);
	r->nodes[r->len] = node;
	r->len++;
	return (true);
}

static int	solution_alloc_storage(t_solution *s, int k)
{
	size_t	total_nodes;
	size_t	bytes;
	size_t	aligned_bytes;

	total_nodes = (size_t)k * (size_t)s->route_cap;
	bytes = total_nodes * sizeof(int);
	aligned_bytes = align_up_size(bytes, SOLUTION_ALIGNMENT);
	s->nodes_storage = aligned_alloc(SOLUTION_ALIGNMENT, aligned_bytes);
	if (!s->nodes_storage)
		return (0);
	memset(s->nodes_storage, 0, aligned_bytes);
	return (1);
}

static void	solution_init_routes(t_solution *s, int k)
{
	t_route	*r;
	int		i;

	i = 0;
	for (; i < k; i++)
	{
		r = &s->routes[i];
		r->nodes = s->nodes_storage + (size_t)i * (size_t)s->route_cap;
		r->cap = s->route_cap;
		r->len = 0;
	}
}

t_solution	*solution_create(int k, int n)
{
	t_solution	*s;

	if (k <= 0 || n < 0)
		return (NULL);
	s = calloc(1, sizeof(*s));
	if (!s)
		return (NULL);
	s->k = k;
	s->route_cap = n + 2;
	s->routes = calloc((size_t)k, sizeof(t_route));
	if (!s->routes || !solution_alloc_storage(s, k))
	{
		solution_free(s);
		return (NULL);
	}
	solution_init_routes(s, k);
	return (s);
}

void	solution_reset(t_solution *s)
{
	int	i;

	i = 0;
	for (; i < s->k; i++)
	{
		s->routes[i].len = 0;
	}
}

void	solution_free(t_solution *s)
{
	if (!s)
		return ;
	free(s->nodes_storage);
	free(s->routes);
	free(s);
}

void	solution_copy(t_solution *dst, const t_solution *src)
{
	const t_route	*r_src;
	t_route			*r_dst;
	int				i;

	i = 0;
	for (; i < src->k; i++)
	{
		r_src = &src->routes[i];
		r_dst = &dst->routes[i];
		r_dst->len = r_src->len;
		memcpy(r_dst->nodes, r_src->nodes, (size_t)r_src->len * sizeof(int));
	}
}

double	solution_cost(const t_solution *s, double **c)
{
	const t_route	*r;
	double			cost;
	int				i;
	int				t;

	cost = 0.0;
	i = 0;
	for (; i < s->k; i++)
	{
		r = &s->routes[i];
		t = 0;
		for (; t + 1 < r->len; t++)
		{
			cost += c[r->nodes[t]][r->nodes[t + 1]];
		}
	}
	return (cost);
}

static void	set_err(char *err, size_t err_len, const char *msg)
{
	if (!err || err_len == 0)
		return ;
	snprintf(err, err_len, "%s", msg);
}

static bool	validate_header(const t_solution *s, int n, int k, char *err,
		size_t err_len)
{
	if (!s)
	{
		set_err(err, err_len, "solution is NULL");
		return (false);
	}
	if (s->k != k || !s->routes || n < 0 || k < 0)
	{
		set_err(err, err_len, "invalid solution metadata");
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

static bool	validate_route(const t_route *r, int n, bool *seen, char *err,
		size_t err_len)
{
	int	node;
	int	t;

	if (r->len < 2 || r->nodes[0] != 0 || r->nodes[r->len - 1] != 0)
	{
		set_err(err, err_len, "route must start/end at depot 0");
		return (false);
	}
	t = 0;
	for (; t < r->len; t++)
	{
		node = r->nodes[t];
		if (node < 0 || node > n)
		{
			set_err(err, err_len, "route contains out-of-range node");
			return (false);
		}
		if (!validate_node(node, t, r, seen))
		{
			set_err(err, err_len, "invalid or repeated route node");
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

bool	solution_validate(const t_solution *s, int n, int k, char *err,
		size_t err_len)
{
	bool	*seen;
	int		i;
	bool	valid;

	set_err(err, err_len, "");
	if (!validate_header(s, n, k, err, err_len))
		return (false);
	seen = calloc((size_t)(n + 1), sizeof(bool));
	if (!seen)
	{
		set_err(err, err_len, "allocation failure in solution_validate");
		return (false);
	}
	valid = true;
	i = 0;
	for (; valid && i < s->k; i++)
	{
		valid = validate_route(&s->routes[i], n, seen, err, err_len);
	}
	if (valid)
		valid = validate_all_seen(seen, n, err, err_len);
	free(seen);
	return (valid);
}

static bool	validate_capacity_route(const t_route *r, const int *demands,
		int vehicle_capacity, char *err, size_t err_len)
{
	int	load;
	int	node;
	int	t;

	load = 0;
	t = 1;
	for (; t + 1 < r->len; t++)
	{
		node = r->nodes[t];
		if (demands[node] < 0)
		{
			set_err(err, err_len, "route contains customer with negative demand");
			return (false);
		}
		load += demands[node];
		if (load > vehicle_capacity)
		{
			set_err(err, err_len, "route exceeds vehicle capacity");
			return (false);
		}
	}
	return (true);
}

bool	solution_validate_cvrp(const t_solution *s, int n, int k,
		const int *demands, int vehicle_capacity, char *err, size_t err_len)
{
	int	i;

	if (!solution_validate(s, n, k, err, err_len))
		return (false);
	if (!demands || vehicle_capacity <= 0 || demands[0] != 0)
	{
		set_err(err, err_len, "invalid capacity validation input");
		return (false);
	}
	i = 0;
	for (; i < s->k; i++)
	{
		if (!validate_capacity_route(&s->routes[i], demands,
				vehicle_capacity, err, err_len))
			return (false);
	}
	set_err(err, err_len, "");
	return (true);
}
