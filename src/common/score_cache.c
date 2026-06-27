#include "internal.h"
#include <string.h>

static void	score_cache_line_copy(double *dst, const double *src, int len)
{
	memcpy(dst, src, (size_t)len * sizeof(double));
}

static void	score_cache_compute_row(double *dst, int current, int n,
		const t_score_params *params)
{
	double	*tau_row;
	double	*eta_row;
	int		node;

	tau_row = params->tau[current];
	eta_row = params->eta[current];
	node = 0;
	while (node <= n)
	{
		if (node == current)
			dst[node] = 0.0;
		else
		{
			dst[node] = fast_pow_nonneg(tau_row[node], params->alpha)
				* fast_pow_nonneg(eta_row[node], params->beta);
		}
		node++;
	}
}

static double	*score_cache_get_l1_line(t_score_cache *cache, int idx)
{
	return (cache->l1_rows + (size_t)idx * (size_t)cache->line_len);
}

static double	*score_cache_get_l2_line(t_score_cache *cache, int idx)
{
	return (cache->l2_rows + (size_t)idx * (size_t)cache->line_len);
}

static const double	*score_cache_get_row_l2(t_score_cache *cache,
						int current, double *l1_line,
						const t_score_params *params)
{
	int		l2_idx;
	double	*l2_line;
	int		l1_idx;

	l1_idx = current % cache->l1_lines;
	l2_idx = current % cache->l2_lines;
	l2_line = score_cache_get_l2_line(cache, l2_idx);
	if (cache->l2_keys[l2_idx] == current)
	{
		cache->l2_hits++;
		score_cache_line_copy(l1_line, l2_line, cache->line_len);
		cache->l1_keys[l1_idx] = current;
		return (l1_line);
	}
	cache->l3_misses++;
	score_cache_compute_row(l2_line, current, cache->n, params);
	cache->l2_keys[l2_idx] = current;
	score_cache_line_copy(l1_line, l2_line, cache->line_len);
	cache->l1_keys[l1_idx] = current;
	return (l1_line);
}

const double	*score_cache_get_row(t_score_cache *cache, int current,
		const t_score_params *params)
{
	int		l1_idx;
	double	*l1_line;

	if (!cache || current < 0 || current > cache->n || cache->l1_lines <= 0)
		return (NULL);
	l1_idx = current % cache->l1_lines;
	if (cache->l1_keys[l1_idx] == current)
	{
		cache->l1_hits++;
		return (score_cache_get_l1_line(cache, l1_idx));
	}
	l1_line = score_cache_get_l1_line(cache, l1_idx);
	if (cache->l2_lines > 0)
		return (score_cache_get_row_l2(cache, current, l1_line, params));
	cache->l3_misses++;
	score_cache_compute_row(l1_line, current, cache->n, params);
	cache->l1_keys[l1_idx] = current;
	return (l1_line);
}
