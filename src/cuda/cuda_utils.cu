extern "C" {
#include "solver.h"
}

#include "cuda/cuda_internal.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

int	cuda_check(cudaError_t err, t_status *status)
{
	if (err == cudaSuccess)
		return (1);
	fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
	*status = SOLVER_ERR_BACKEND;
	return (0);
}

double	cuda_wall_time_seconds(void)
{
	struct timespec	ts;

	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

int	cuda_is_significant_improvement(double prev_best, double new_best,
		double min_rel_improvement)
{
	double	abs_gain;
	double	rel_gain;

	if (prev_best >= DBL_MAX || new_best >= DBL_MAX)
		return (new_best < prev_best);
	if (new_best >= prev_best - SOLVER_EPS)
		return (0);
	abs_gain = prev_best - new_best;
	rel_gain = abs_gain / fmax(prev_best, SOLVER_EPS);
	return (rel_gain + SOLVER_EPS >= min_rel_improvement);
}

int	cuda_select_iter_best_host(const t_cuda_ant_summary *summary, int m,
		int *best_idx, float *best_cost)
{
	int		found;
	int		idx;
	float	cost;
	int		i;

	found = 0;
	idx = -1;
	cost = FLT_MAX;
	i = 0;
	for (; i < m; ++i)
	{
		if (!summary[i].feasible)
			continue ;
		if (!found || summary[i].cost < cost
			|| (fabsf(summary[i].cost - cost) <= (float)CUDA_EPS && i < idx))
		{
			found = 1;
			idx = i;
			cost = summary[i].cost;
		}
	}
	*best_idx = idx;
	*best_cost = cost;
	return (found);
}
