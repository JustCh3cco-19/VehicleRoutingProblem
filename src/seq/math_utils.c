# include "solver.h"
#include "config.h"
#include "seq/internal.h"
#include <float.h>
#include <math.h>

#ifdef _OPENMP
# include <omp.h>
#endif

int	seq_clamp(int x, int lo, int hi)
{
	if (x < lo)
		return (lo);
	if (x > hi)
		return (hi);
	return (x);
}

double	seq_fast_pow(double base, double exponent)
{
	if (exponent == 1.0)
		return (base);
	if (exponent == 2.0)
		return (base * base);
	if (exponent == 0.5)
		return (sqrt(base));
	return (pow(base, exponent));
}

int	seq_is_improvement(double prev_best, double new_best,
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

int	seq_choose_candidate_count(int n, int requested_candidate_k)
{
	if (requested_candidate_k > 0)
		return (seq_clamp(requested_candidate_k, 1, n));
	if (n <= kSeqDenseCandidateLimit)
		return (n);
	return (seq_clamp(kSeqDefaultSparseCandidateCount, 1, n));
}

#ifdef _OPENMP

int	seq_choose_auto_ants(int n)
{
	int	workers;
	int	ants_per_worker;
	int	total_ants;

	workers = omp_get_max_threads();
	if (workers < 1)
		workers = 1;
	ants_per_worker = 6;
	if (n <= 2000)
		ants_per_worker = 8;
	else if (n > 16000)
		ants_per_worker = 4;
	total_ants = workers * ants_per_worker;
	total_ants = seq_clamp(total_ants, workers * 4, workers * 8);
	if (total_ants < 8)
		total_ants = 8;
	return (total_ants);
}

#else

int	seq_choose_auto_ants(int n)
{
	int	ants_per_worker;
	int	total_ants;

	ants_per_worker = 6;
	if (n <= 2000)
		ants_per_worker = 8;
	else if (n > 16000)
		ants_per_worker = 4;
	total_ants = 1 * ants_per_worker;
	total_ants = seq_clamp(total_ants, 4, 8);
	if (total_ants < 8)
		total_ants = 8;
	return (total_ants);
}

#endif

