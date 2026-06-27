#include "internal.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

const char	*aco_status_string(AcoStatus status)
{
	if (status == ACO_OK)
		return ("ok");
	if (status == ACO_ERR_INVALID_INPUT)
		return ("invalid input");
	if (status == ACO_ERR_ALLOCATION)
		return ("allocation failure");
	if (status == ACO_ERR_NO_SOLUTION)
		return ("no solution");
	if (status == ACO_ERR_BACKEND)
		return ("backend failure");
	return ("unknown status");
}

double	fast_pow_nonneg(double base, double exponent)
{
	if (exponent == 1.0)
		return (base);
	if (exponent == 2.0)
		return (base * base);
	if (exponent == 0.5)
		return (sqrt(base));
	return (pow(base, exponent));
}

static double	rand01(void)
{
	return ((double)rand() / (double)RAND_MAX);
}

double	aco_rand01_state(unsigned int *state)
{
	unsigned int	x;

	if (!state)
		return (rand01());
	x = *state;
	if (x == 0u)
		x = 1u;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return ((double)x / (double)UINT_MAX);
}

unsigned int	aco_make_ant_seed(unsigned int base_seed, int iter,
		int ant_index)
{
	unsigned int	x;

	x = base_seed ? base_seed : 1u;
	x ^= 0xa511e9b3u;
	x ^= (unsigned int)(iter + 1) * 0x9e3779b1u;
	x ^= (unsigned int)(ant_index + 1) * 0xc2b2ae3du;
	if (x)
		return (x);
	return (1u);
}
