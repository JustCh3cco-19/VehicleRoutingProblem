#include "aco.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include "solution.h"
#include <time.h>

#ifdef USE_MPI
# include <mpi.h>
#endif
#ifdef _OPENMP
# include <omp.h>
#endif

size_t	par_align_up(size_t value)
{
	size_t	aligned;

	aligned = 0;
	if (!matrix_align_up(value, kParAlignment, &aligned))
		return (0);
	return (aligned);
}

void	*par_aligned_calloc(size_t bytes)
{
	return (matrix_aligned_calloc(bytes, kParAlignment));
}

bool	par_route_append(Route *r, int node)
{
	if (r->len >= r->cap)
		return (false);
	r->nodes[r->len] = node;
	r->len++;
	return (true);
}

#ifdef USE_MPI

double	par_wall_time(void)
{
	int	init;

	init = 0;
	MPI_Initialized(&init);
	if (init)
		return (MPI_Wtime());
# ifdef _OPENMP
	return (omp_get_wtime());
# else
	struct timespec	ts;

	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
# endif
}

#else

double	par_wall_time(void)
{
# ifdef _OPENMP
	return (omp_get_wtime());
# else
	struct timespec	ts;

	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
# endif
}

#endif
