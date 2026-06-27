# include "solver.h"
#include "seq/internal.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
# include <omp.h>
#endif

size_t	seq_align_up(size_t value, size_t alignment)
{
	size_t	rem;

	rem = value % alignment;
	if (rem == 0u)
		return (value);
	return (value + (alignment - rem));
}

void	*seq_aligned_calloc(size_t bytes)
{
	size_t	alloc_bytes;
	void	*ptr;

	alloc_bytes = seq_align_up(bytes, seq_aco_alignment);
	ptr = aligned_alloc(seq_aco_alignment, alloc_bytes);
	if (!ptr)
		return (NULL);
	memset(ptr, 0, alloc_bytes);
	return (ptr);
}

int	seq_stride(int cols, size_t elem_size)
{
	size_t	row_bytes;
	size_t	padded;

	row_bytes = (size_t)cols * elem_size;
	padded = seq_align_up(row_bytes, seq_aco_alignment);
	return ((int)(padded / elem_size));
}

#ifdef _OPENMP

double	seq_wall_time(void)
{
	return (omp_get_wtime());
}

#else

double	seq_wall_time(void)
{
	struct timespec	ts;

	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

#endif
