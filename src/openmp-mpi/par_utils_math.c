# include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static long	par_get_l3_cache_size(void)
{
	FILE	*f;
	char	buf[64];
	long	size;
	char	*unit;

	f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
	if (!f)
		return (0);
	if (!fgets(buf, sizeof(buf), f))
	{
		fclose(f);
		return (0);
	}
	fclose(f);
	size = atol(buf);
	unit = strpbrk(buf, "KMGTkmgt");
	if (unit)
	{
		if (*unit == 'k' || *unit == 'k')
			size *= 1024;
		else if (*unit == 'M' || *unit == 'm')
			size *= 1024 * 1024;
		else if (*unit == 'G' || *unit == 'g')
			size *= 1024 * 1024 * 1024;
	}
	return (size);
}

int	par_choose_candidate_count(int n, int requested_candidate_k)
{
	long	l3_size;
	double	target_bytes;
	int		k;

	if (requested_candidate_k > 0)
	{
		if (requested_candidate_k > n)
			return (n);
		return (requested_candidate_k);
	}
	l3_size = par_get_l3_cache_size();
	if (l3_size <= 0)
	{
		if (n < 32)
			return (n);
		return (32);
	}
	target_bytes = (double)l3_size * 0.7;
	k = (int)(target_bytes / ((double)(n + 1) * 8.0));
	if (k < 16)
		k = 16;
	if (k > kParMaxCandidates)
		k = kParMaxCandidates;
	if (k > n)
		return (n);
	return (k);
}

float	par_fast_powf(float base, float exp)
{
	if (exp == 1.0f)
		return (base);
	if (exp == 2.0f)
		return (base * base);
	if (exp == 0.5f)
		return (sqrtf(base));
	return (powf(base, exp));
}

double	par_rand01(unsigned int *state)
{
	unsigned int	x;

	x = *state;
	if (x == 0u)
		x = 1u;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return ((double)x / 4294967295.0);
}

int	par_is_improvement(double prev_best, double new_best,
		double min_rel_improvement)
{
	double	abs_gain;
	double	rel_gain;

	if (prev_best >= DBL_MAX || new_best >= DBL_MAX)
		return (new_best < prev_best);
	if (new_best >= prev_best - 1e-7f)
		return (0);
	abs_gain = prev_best - new_best;
	rel_gain = abs_gain / fmax(prev_best, 1e-7f);
	return (rel_gain + 1e-7f >= min_rel_improvement);
}
