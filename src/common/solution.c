#include "solution.h"

#include <stdbool.h>
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
