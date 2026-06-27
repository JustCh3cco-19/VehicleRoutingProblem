#include "solution.h"

#include <stdbool.h>

bool	route_append(t_route *r, int node)
{
	if (r->len >= r->cap)
		return (false);
	r->nodes[r->len] = node;
	r->len++;
	return (true);
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
			cost += c[r->nodes[t]][r->nodes[t + 1]];
	}
	return (cost);
}
