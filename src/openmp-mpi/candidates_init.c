#include "solver.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static void	insert_candidate(t_cand_list *list, int node, float d)
{
	int	pos;
	int	m;

	pos = -1;
	m = 0;
	while (m < list->k)
	{
		if (d < list->dists[m])
		{
			pos = m;
			break ;
		}
		m++;
	}
	if (pos >= 0)
	{
		m = list->k - 1;
		while (m > pos)
		{
			list->dists[m] = list->dists[m - 1];
			list->nodes[m] = list->nodes[m - 1];
			m--;
		}
		list->dists[pos] = d;
		list->nodes[pos] = node;
	}
}

static void	copy_candidates_to_shared(t_par_shared *s, int i,
				t_cand_list *list, double beta)
{
	int		*c_row;
	float	*e_row;
	int		t;

	c_row = s->cand_idx + (size_t)i * (size_t)s->stride;
	e_row = s->eta_beta + (size_t)i * (size_t)s->stride;
	t = 0;
	while (t < list->k)
	{
		c_row[t] = (list->nodes[t] > 0) ? list->nodes[t] : 0;
		e_row[t] = (list->nodes[t] > 0) ? par_fast_powf(1.0f
				/ (list->dists[t] + 1e-7f), (float)beta) : 0.0f;
		t++;
	}
}

static void	par_build_candidate_row(t_par_shared *s, int i,
		const t_par_matrix *c_mat, double beta)
{
	int			nodes[par_max_candidates];
	float		dists[par_max_candidates];
	int			t;
	int			node;
	t_cand_list	list;

	t = 0;
	while (t < s->cand_k)
	{
		nodes[t] = -1;
		dists[t] = FLT_MAX;
		t++;
	}
	list.nodes = nodes;
	list.dists = dists;
	list.k = s->cand_k;
	node = 1;
	while (node <= s->n)
	{
		if (node != i)
			insert_candidate(&list, node, c_mat->rows[i][node]);
		node++;
	}
	copy_candidates_to_shared(s, i, &list, beta);
}

static int	allocate_shared_buffers(t_par_shared *s,
				const t_cand_init_params *params)
{
	size_t	bytes[4];
	size_t	stride;
	size_t	total_elems;

	bytes[0] = 0;
	bytes[1] = 0;
	if (params->n < 0 || params->n == INT32_MAX || params->cand_k <= 0 ||
		!matrix_mul_size((size_t)params->cand_k, sizeof(float), &bytes[0]) ||
		!matrix_align_up(bytes[0], par_alignment, &bytes[1]))
		return (0);
	stride = bytes[1] / sizeof(float);
	if (stride > (size_t)INT32_MAX)
		return (0);
	s->stride = (int)stride;
	s->visited_words = (params->n / 64) + 1;
	s->meta_words = (s->visited_words / 64) + 1;
	total_elems = 0;
	if (!matrix_mul_size((size_t)(params->n + 1), (size_t)s->stride,
			&total_elems) ||
		!matrix_mul_size(total_elems, sizeof(int), &bytes[2]) ||
		!matrix_mul_size(total_elems, sizeof(float), &bytes[3]))
		return (0);
	s->cand_idx = par_aligned_calloc(bytes[2]);
	s->eta_beta = par_aligned_calloc(bytes[3]);
	return (s->cand_idx && s->eta_beta);
}

int	par_shared_init(t_par_shared *s, const t_cand_init_params *params)
{
	int	i;

	memset(s, 0, sizeof(*s));
	s->n = params->n;
	s->cand_k = params->cand_k;
	if (!allocate_shared_buffers(s, params))
	{
		par_shared_free(s);
		return (0);
	}
#pragma omp parallel for schedule(static)
	for (i = 0; i <= params->n; i++)
		par_build_candidate_row(s, i, params->c_mat, params->beta);
	return (1);
}
