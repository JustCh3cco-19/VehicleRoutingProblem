#include "matrix.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int	matrix_mul_size(size_t a, size_t b, size_t *out)
{
	if (!out)
		return (0);
	if (a != 0u && b > SIZE_MAX / a)
		return (0);
	*out = a * b;
	return (1);
}

int	matrix_align_up(size_t value, size_t alignment, size_t *out)
{
	size_t	rem;
	size_t	padding;

	if (!out || alignment == 0u)
		return (0);
	rem = value % alignment;
	if (rem == 0u)
	{
		*out = value;
		return (1);
	}
	padding = alignment - rem;
	if (value > SIZE_MAX - padding)
		return (0);
	*out = value + padding;
	return (1);
}

void	*matrix_aligned_calloc(size_t bytes, size_t alignment)
{
	size_t	alloc_bytes;
	void	*ptr;

	alloc_bytes = 0;
	if (!matrix_align_up(bytes, alignment, &alloc_bytes))
		return (NULL);
	ptr = aligned_alloc(alignment, alloc_bytes);
	if (!ptr)
		return (NULL);
	memset(ptr, 0, alloc_bytes);
	return (ptr);
}

static int	matrix_calc_bytes(size_t size, size_t *stride,
				size_t *total_bytes)
{
	size_t	row_bytes;
	size_t	padded_row_bytes;
	size_t	total_elems;

	row_bytes = 0;
	padded_row_bytes = 0;
	total_elems = 0;
	if (!matrix_mul_size(size, sizeof(double), &row_bytes))
		return (0);
	if (!matrix_align_up(row_bytes, matrix_default_alignment,
			&padded_row_bytes))
		return (0);
	*stride = padded_row_bytes / sizeof(double);
	if (*stride > (size_t)INT32_MAX)
		return (0);
	if (!matrix_mul_size(size, *stride, &total_elems))
		return (0);
	return (matrix_mul_size(total_elems, sizeof(double), total_bytes));
}

static int	matrix_alloc_storage(t_matrix *m, size_t size)
{
	size_t	total_bytes;
	size_t	stride;

	total_bytes = 0;
	stride = 0;
	if (!matrix_calc_bytes(size, &stride, &total_bytes))
		return (0);
	m->stride = (int)stride;
	m->data = matrix_aligned_calloc(total_bytes, matrix_default_alignment);
	if (!m->data)
		return (0);
	m->rows = malloc(size * sizeof(double *));
	if (!m->rows)
	{
		free(m->data);
		return (0);
	}
	return (1);
}

t_matrix	*matrix_create(int n)
{
	t_matrix	*m;
	size_t		size;
	size_t		i;

	if (n < 0)
		return (NULL);
	size = (size_t)n + 1u;
	if (size > (size_t)INT32_MAX)
		return (NULL);
	m = malloc(sizeof(*m));
	if (!m)
		return (NULL);
	m->n = n;
	if (!matrix_alloc_storage(m, size))
	{
		free(m);
		return (NULL);
	}
	i = 0;
	while (i < size)
	{
		m->rows[i] = m->data + i * (size_t)m->stride;
		i++;
	}
	return (m);
}

void	matrix_free_handle(t_matrix *m)
{
	if (!m)
		return ;
	free(m->rows);
	free(m->data);
	free(m);
}

double	**matrix_alloc(int n)
{
	t_matrix	*m;
	double		**rows;

	m = matrix_create(n);
	if (!m)
		return (NULL);
	rows = m->rows;
	free(m);
	return (rows);
}

void	matrix_free(double **m)
{
	if (!m)
		return ;
	free(m[0]);
	free(m);
}
