#ifndef SOLVER_H
# define SOLVER_H

# include "solution.h"

# define SOLVER_EPS 1e-9

enum e_status
{
	SOLVER_OK = 0,
	SOLVER_ERR_INVALID_INPUT = 1,
	SOLVER_ERR_ALLOCATION = 2,
	SOLVER_ERR_NO_SOLUTION = 3,
	SOLVER_ERR_BACKEND = 4
};
typedef enum e_status		t_status;

const char					*status_string(t_status status);

t_status					vrp_solve(int n, int k, int m, double **c,
								double alpha, double beta, double rho,
								double tau0, double q, unsigned int seed,
								t_solution *best_solution, double *best_cost);

t_status					vrp_solve_with_capacity(int n, int k,
								int vehicle_capacity_customers, int m,
								double **c, double alpha, double beta,
								double rho, double tau0, double q,
								unsigned int seed, t_solution *best_solution,
								double *best_cost);

double						rand01_state(unsigned int *state);
unsigned int				make_ant_seed(unsigned int base_seed, int iter,
								int ant_index);

#endif
