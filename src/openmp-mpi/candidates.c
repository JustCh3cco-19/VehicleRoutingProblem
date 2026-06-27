# include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void	par_matrix_free(t_par_matrix *m)
{
	if (!m)
		return ;
	free(m->data);
	free(m->rows);
	free(m);
}

static size_t	calc_matrix_stride(size_t size, size_t *total_bytes)
{
	size_t	row_bytes;
	size_t	padded;
	size_t	total_elems;

	if (!matrix_mul_size(size, sizeof(float), &row_bytes) ||
		!matrix_align_up(row_bytes, kParAlignment, &padded))
		return (0);
	*total_bytes = 0;
	total_elems = 0;
	if (padded / sizeof(float) > (size_t)INT32_MAX ||
		!matrix_mul_size(size, padded / sizeof(float), &total_elems) ||
		!matrix_mul_size(total_elems, sizeof(float), total_bytes))
		return (0);
	return (padded / sizeof(float));
}

t_par_matrix	*par_matrix_create(int n)
{
	size_t			size;
	size_t			total_bytes;
	size_t			stride;
	t_par_matrix	*m;
	int				i;

	if (n < 0)
		return (NULL);
	size = (size_t)n + 1u;
	if (size > (size_t)INT32_MAX)
		return (NULL);
	stride = calc_matrix_stride(size, &total_bytes);
	if (stride == 0)
		return (NULL);
	m = malloc(sizeof(*m));
	if (!m)
		return (NULL);
	m->n = n;
	m->stride = (int)stride;
	m->data = par_aligned_calloc(total_bytes);
	m->rows = malloc(size * sizeof(float *));
	if (!m->data || !m->rows)
	{
		free(m->data);
		free(m->rows);
		free(m);
		return (NULL);
	}
	i = 0;
	while (i <= n)
	{
		m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
		i++;
	}
	return (m);
}

void	par_shared_free(t_par_shared *s)
{
	if (!s)
		return ;
	free(s->eta_beta);
	free(s->cand_idx);
}
