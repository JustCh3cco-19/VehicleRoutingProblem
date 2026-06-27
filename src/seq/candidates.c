#include "solver.h"
#include "config.h"
#include "seq/internal.h"
#include "matrix.h"
#include "solution.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void	seq_shared_free(t_seq_shared *shared)
{
	if (!shared)
		return ;
	free(shared->score);
	free(shared->eta_beta);
	free(shared->candidate_idx);
	memset(shared, 0, sizeof(*shared));
}

int	seq_shared_init(t_seq_shared *shared, int n, int candidate_k)
{
	int		rows;
	size_t	elems;

	if (!shared || n < 1)
		return (0);
	memset(shared, 0, sizeof(*shared));
	shared->n = n;
	shared->candidate_k = seq_choose_candidate_count(n, candidate_k);
	shared->candidate_k = seq_clamp(shared->candidate_k, 1, n);
	shared->stride = seq_stride(shared->candidate_k, sizeof(float));
	if (shared->stride < shared->candidate_k)
		shared->stride = shared->candidate_k;
	shared->visited_words = (n / 64) + 1;
	rows = n + 1;
	elems = (size_t)rows * (size_t)shared->stride;
	shared->candidate_idx = seq_aligned_calloc(elems * sizeof(int));
	shared->eta_beta = seq_aligned_calloc(elems * sizeof(float));
	shared->score = seq_aligned_calloc(elems * sizeof(float));
	if (!shared->candidate_idx || !shared->eta_beta || !shared->score)
	{
		seq_shared_free(shared);
		return (0);
	}
	return (1);
}

static void	init_and_zero_row(int *cand_row, float *eta_row, int k,
				int stride)
{
	int	t;

	t = 0;
	while (t < k)
	{
		cand_row[t] = 0;
		eta_row[t] = 0.0f;
		t++;
	}
	t = k;
	while (t < stride)
	{
		cand_row[t] = 0;
		eta_row[t] = 0.0f;
		t++;
	}
}

static void	shift_and_insert(int *cand_row, float *eta_row, int k,
				int insert_at)
{
	int	t;

	t = k - 1;
	while (t > insert_at)
	{
		cand_row[t] = cand_row[t - 1];
		eta_row[t] = eta_row[t - 1];
		t--;
	}
}

static int	find_insert_pos(const int *cand_row, int k, const double *c_row,
				double dist)
{
	int	t;

	t = 0;
	while (t < k)
	{
		if (cand_row[t] == 0 || dist < c_row[cand_row[t]])
			return (t);
		t++;
	}
	return (-1);
}

static void	build_candidates_for_row(t_seq_shared *shared, int i, double **c,
				double beta)
{
	int		*cand_row;
	float	*eta_row;
	int		node;
	int		insert_at;

	cand_row = shared->candidate_idx + (size_t)i * (size_t)shared->stride;
	eta_row = shared->eta_beta + (size_t)i * (size_t)shared->stride;
	init_and_zero_row(cand_row, eta_row, shared->candidate_k,
		shared->stride);
	node = 1;
	while (node <= shared->n)
	{
		if (node != i)
		{
			insert_at = find_insert_pos(cand_row, shared->candidate_k, c[i],
					c[i][node]);
			if (insert_at >= 0)
			{
				shift_and_insert(cand_row, eta_row, shared->candidate_k,
					insert_at);
				cand_row[insert_at] = node;
				eta_row[insert_at] = (float)seq_fast_pow(1.0 / (c[i][node]
							+ SOLVER_EPS), beta);
			}
		}
		node++;
	}
}

void	seq_shared_build_candidates(t_seq_shared *shared, double **c,
		double beta)
{
	int	i;

	i = 0;
	while (i <= shared->n)
	{
		build_candidates_for_row(shared, i, c, beta);
		i++;
	}
}

static void	zero_tail(float *row, int start, int end)
{
	int	t;

	t = start;
	while (t < end)
	{
		row[t] = 0.0f;
		t++;
	}
}

static void	update_score_row_generic(t_score_row_params *params,
				double alpha)
{
	int		t;
	int		node;
	double	tau_term;

	t = 0;
	while (t < params->k)
	{
		node = params->cand_row[t];
		if (node > 0)
		{
			tau_term = seq_fast_pow(params->tau_row[node], alpha);
			params->score_row[t] = (float)(tau_term
					* (double)params->eta_row[t]);
		}
		else
			params->score_row[t] = 0.0f;
		t++;
	}
}

void	seq_shared_update_scores(t_seq_shared *shared, double **restrict tau,
		double alpha)
{
	int					n;
	int					stride;
	int					i;
	t_score_row_params	params;

	n = shared->n;
	stride = shared->stride;
	params.k = shared->candidate_k;
	i = 0;
	while (i <= n)
	{
		params.cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
		params.eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
		params.score_row = shared->score + (size_t)i * (size_t)stride;
		params.tau_row = tau[i];
		if (alpha == 1.0)
			update_score_row_alpha1(&params);
		else if (alpha == 2.0)
			update_score_row_alpha2(&params);
		else
			update_score_row_generic(&params, alpha);
		zero_tail(params.score_row, params.k, stride);
		i++;
	}
}
