#ifndef SOLUTION_H
# define SOLUTION_H

# include <stdbool.h>
# include <stddef.h>

struct s_route
{
	int						*nodes;
	int						len;
	int						cap;
};
typedef struct s_route		t_route;

struct s_solution
{
	t_route					*routes;
	int						k;
	int						route_cap;
	int						*nodes_storage;
};
typedef struct s_solution	t_solution;

struct s_solution_validation
{
	const t_solution		*s;
	int						n;
	int						k;
	const int				*demands;
	int						vehicle_capacity;
	char					*err;
	size_t					err_len;
};
typedef struct s_solution_validation	t_solution_validation;

t_solution					*solution_create(int k, int n);
void						solution_free(t_solution *s);
void						solution_reset(t_solution *s);
void						solution_copy(t_solution *dst,
								const t_solution *src);
double						solution_cost(const t_solution *s, double **c);
bool						route_append(t_route *r, int node);
bool						solution_validate(const t_solution *s, int n, int k,
								char *err, size_t err_len);
bool						solution_validate_cvrp(
								t_solution_validation *validation);

#endif
