# include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void	insert_candidate(int *nodes, float *dists, int cand_k, int node,
		float d)
{
	int	pos;
	int	m;

	pos = -1;
	m = 0;
	while (m < cand_k)
	{
		if (d < dists[m])
		{
			pos = m;
			break ;
		}
		m++;
	}
	if (pos >= 0)
	{
		m = cand_k - 1;
		while (m > pos)
		{
			dists[m] = dists[m - 1];
			nodes[m] = nodes[m - 1];
			m--;
		}
		dists[pos] = d;
		nodes[pos] = node;
	}
}

static void	par_build_candidate_row(t_par_shared *s, int i,
		const t_par_matrix *c_mat, double beta)
{
	int			nodes[kParMaxCandidates];
	float		dists[kParMaxCandidates];
	int			t;
	int			node;
	const float	*row;
	int			*c_row;
	float		*e_row;

	t = 0;
	while (t < s->cand_k)
	{
		nodes[t] = -1;
		dists[t] = FLT_MAX;
		t++;
	}
	row = c_mat->rows[i];
	node = 1;
	while (node <= s->n)
	{
		if (node != i)
			insert_candidate(nodes, dists, s->cand_k, node, row[node]);
		node++;
	}
	c_row = s->cand_idx + (size_t)i * (size_t)s->stride;
	e_row = s->eta_beta + (size_t)i * (size_t)s->stride;
	t = 0;
	while (t < s->cand_k)
	{
		c_row[t] = (nodes[t] > 0) ? nodes[t] : 0;
		e_row[t] = (nodes[t] > 0) ? par_fast_powf(1.0f / (dists[t] + 1e-7f),
				(float)beta) : 0.0f;
		t++;
	}
}

int	par_shared_init(t_par_shared *s, int n, int cand_k,
		const t_par_matrix *c_mat, double beta)
{
	size_t	bytes[4];
	size_t	stride;
	size_t	total_elems;
	int		i;

	memset(s, 0, sizeof(*s));
	s->n = n;
	s->cand_k = cand_k;
	bytes[0] = 0;
	bytes[1] = 0;
	if (n < 0 || n == INT32_MAX || cand_k <= 0 ||
		!matrix_mul_size((size_t)cand_k, sizeof(float), &bytes[0]) ||
		!matrix_align_up(bytes[0], kParAlignment, &bytes[1]))
		return (0);
	stride = bytes[1] / sizeof(float);
	if (stride > (size_t)INT32_MAX)
		return (0);
	s->stride = (int)stride;
	s->visited_words = (n / 64) + 1;
	s->meta_words = (s->visited_words / 64) + 1;
	total_elems = 0;
	if (!matrix_mul_size((size_t)(n + 1), (size_t)s->stride, &total_elems) ||
		!matrix_mul_size(total_elems, sizeof(int), &bytes[2]) ||
		!matrix_mul_size(total_elems, sizeof(float), &bytes[3]))
		return (0);
	s->cand_idx = par_aligned_calloc(bytes[2]);
	s->eta_beta = par_aligned_calloc(bytes[3]);
	if (!s->cand_idx || !s->eta_beta)
	{
		par_shared_free(s);
		return (0);
	}
#pragma omp parallel for schedule(static)
	for (i = 0; i <= n; i++)
		par_build_candidate_row(s, i, c_mat, beta);
	return (1);
}
