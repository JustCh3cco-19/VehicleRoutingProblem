#include "internal.h"
#include <stdlib.h>

t_aco_score_cache	*aco_score_cache_create(int n, int l1_lines, int l2_lines)
{
	t_aco_score_cache	*cache;

	if (n < 0)
		return (NULL);
	if (l1_lines < 1)
		l1_lines = 1;
	if (l2_lines < 0)
		l2_lines = 0;
	cache = calloc(1, sizeof(*cache));
	if (!cache)
		return (NULL);
	cache->n = n;
	cache->line_len = n + 1;
	cache->l1_lines = l1_lines;
	cache->l2_lines = l2_lines;
	cache->l1_keys = malloc((size_t)l1_lines * sizeof(int));
	cache->l1_rows = calloc((size_t)l1_lines * (size_t)cache->line_len,
			sizeof(double));
	if (!cache->l1_keys || !cache->l1_rows)
	{
		aco_score_cache_free(cache);
		return (NULL);
	}
	if (l2_lines > 0)
	{
		cache->l2_keys = malloc((size_t)l2_lines * sizeof(int));
		cache->l2_rows = calloc((size_t)l2_lines * (size_t)cache->line_len,
				sizeof(double));
		if (!cache->l2_keys || !cache->l2_rows)
		{
			aco_score_cache_free(cache);
			return (NULL);
		}
	}
	aco_score_cache_invalidate(cache);
	return (cache);
}

void	aco_score_cache_invalidate(t_aco_score_cache *cache)
{
	int	i;

	if (!cache)
		return ;
	i = 0;
	while (i < cache->l1_lines)
	{
		cache->l1_keys[i] = -1;
		i++;
	}
	i = 0;
	while (i < cache->l2_lines)
	{
		cache->l2_keys[i] = -1;
		i++;
	}
}

void	aco_score_cache_reset_stats(t_aco_score_cache *cache)
{
	if (!cache)
		return ;
	cache->l1_hits = 0;
	cache->l2_hits = 0;
	cache->l3_misses = 0;
}

void	aco_score_cache_get_stats(const t_aco_score_cache *cache,
		t_aco_cache_stats *out)
{
	if (!out)
		return ;
	out->l1_hits = 0;
	out->l2_hits = 0;
	out->l3_misses = 0;
	if (!cache)
		return ;
	out->l1_hits = cache->l1_hits;
	out->l2_hits = cache->l2_hits;
	out->l3_misses = cache->l3_misses;
}

void	aco_score_cache_free(t_aco_score_cache *cache)
{
	if (!cache)
		return ;
	free(cache->l2_rows);
	free(cache->l2_keys);
	free(cache->l1_rows);
	free(cache->l1_keys);
	free(cache);
}
