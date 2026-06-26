#ifndef ACO_INTERNAL_H
#define ACO_INTERNAL_H

#include "aco.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int n;
  int line_len;
  int l1_lines;
  int l2_lines;
  int *l1_keys;
  int *l2_keys;
  double *l1_rows;
  double *l2_rows;
  size_t l1_hits;
  size_t l2_hits;
  size_t l3_misses;
} AcoScoreCache;

typedef struct {
  size_t l1_hits;
  size_t l2_hits;
  size_t l3_misses;
} AcoCacheStats;

typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
} AcoRankShared;

typedef struct {
  Solution *sol;
  Solution *thread_best;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} AcoThreadWorkspace;

typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
  float *score;
} SeqShared;

typedef struct {
  Solution *sol;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} SeqWorkspace;

AcoScoreCache *aco_score_cache_create(int n, int l1_lines, int l2_lines);
void aco_score_cache_invalidate(AcoScoreCache *cache);
void aco_score_cache_reset_stats(AcoScoreCache *cache);
void aco_score_cache_get_stats(const AcoScoreCache *cache, AcoCacheStats *out);
void aco_score_cache_free(AcoScoreCache *cache);

#endif
