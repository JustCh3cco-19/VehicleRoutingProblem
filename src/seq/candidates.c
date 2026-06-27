#include "aco.h"
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

void	seq_shared_build_candidates(t_seq_shared *shared, double **c,
		double beta)
{
	int		n;
	int		k;
	int		stride;
	int		i;
	int		node;
	int		insert_at;
	int		t;
	int		*cand_row;
	float	*eta_row;
	double	dist;
	double	eta;

	n = shared->n;
	k = shared->candidate_k;
	stride = shared->stride;
	i = 0;
	while (i <= n)
	{
		cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
		eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
		t = 0;
		while (t < k)
		{
			cand_row[t] = 0;
			eta_row[t] = 0.0f;
			t++;
		}
		node = 1;
		while (node <= n)
		{
			if (node == i)
			{
				node++;
				continue ;
			}
			dist = c[i][node];
			insert_at = -1;
			t = 0;
			while (t < k)
			{
				if (cand_row[t] == 0 || dist < c[i][cand_row[t]])
				{
					insert_at = t;
					break ;
				}
				t++;
			}
			if (insert_at >= 0)
			{
				t = k - 1;
				while (t > insert_at)
				{
					cand_row[t] = cand_row[t - 1];
					eta_row[t] = eta_row[t - 1];
					t--;
				}
				cand_row[insert_at] = node;
				eta = 1.0 / (dist + ACO_EPS);
				eta_row[insert_at] = (float)seq_fast_pow(eta, beta);
			}
			node++;
		}
		t = k;
		while (t < stride)
		{
			cand_row[t] = 0;
			eta_row[t] = 0.0f;
			t++;
		}
		i++;
	}
}

void	seq_shared_update_scores(t_seq_shared *shared, double **restrict tau,
		double alpha)
{
	int				n;
	int				k;
	int				stride;
	int				i;
	int				*cand_row;
	float			*eta_row;
	float			*score_row;
	const double	*tau_row;
	int				t;
	double			tau_term;

	n = shared->n;
	k = shared->candidate_k;
	stride = shared->stride;
	i = 0;
	while (i <= n)
	{
		cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
		eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
		score_row = shared->score + (size_t)i * (size_t)stride;
		tau_row = tau[i];
		if (alpha == 1.0)
		{
			update_score_row_alpha1(cand_row, eta_row, score_row, tau_row, k);
		}
		else if (alpha == 2.0)
		{
			update_score_row_alpha2(cand_row, eta_row, score_row, tau_row, k);
		}
		else
		{
			t = 0;
			while (t < k)
			{
				int node = cand_row[t];
				if (node > 0)
				{
					tau_term = seq_fast_pow(tau_row[node], alpha);
					score_row[t] = (float)(tau_term * (double)eta_row[t]);
				}
				else
				{
					score_row[t] = 0.0f;
				}
				t++;
			}
		}
		t = k;
		while (t < stride)
		{
			score_row[t] = 0.0f;
			t++;
		}
		i++;
	}
}
