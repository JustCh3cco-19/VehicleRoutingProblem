#include "aco.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Function:  fast_pow_nonneg
 * --------------------------
 * evaluates base^exponent for non-negative base, with fast paths for common
 * exponents used in ACO.
 *
 *  base: non-negative input value
 *  exponent: power exponent
 *
 *  returns: base raised to exponent
 */
static double fast_pow_nonneg(double base, double exponent) {
  if (exponent == 1.0) {
    return base;
  }
  if (exponent == 2.0) {
    return base * base;
  }
  if (exponent == 0.5) {
    return sqrt(base);
  }
  return pow(base, exponent);
}

/*
 * Function:  score_cache_line_copy
 * --------------------------------
 * copies one full score row between cache levels.
 *
 *  dst: destination row
 *  src: source row
 *  line_len: number of row entries
 *
 *  returns: nothing
 */
static void score_cache_line_copy(double *dst, const double *src, int line_len) {
  memcpy(dst, src, (size_t)line_len * sizeof(double));
}

/*
 * Function:  score_cache_compute_row
 * ----------------------------------
 * computes one full desirability row:
 * score[current][node] = tau[current][node]^alpha * eta[current][node]^beta.
 *
 *  dst: output row buffer of length n+1
 *  current: source row index
 *  n: max node index
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *
 *  returns: nothing
 */
static void score_cache_compute_row(double *dst, int current, int n,
                                    double **tau, double **eta, double alpha,
                                    double beta) {
  double *tau_row = tau[current];
  double *eta_row = eta[current];

  for (int node = 0; node <= n; ++node) {
    if (node == current) {
      dst[node] = 0.0;
    } else {
      double tau_term = fast_pow_nonneg(tau_row[node], alpha);
      double eta_term = fast_pow_nonneg(eta_row[node], beta);
      dst[node] = tau_term * eta_term;
    }
  }
}

/*
 * Function:  score_cache_get_l1_line
 * ----------------------------------
 * returns pointer to one L1 line by index.
 */
static double *score_cache_get_l1_line(AcoScoreCache *cache, int idx) {
  return cache->l1_rows + (size_t)idx * (size_t)cache->line_len;
}

/*
 * Function:  score_cache_get_l2_line
 * ----------------------------------
 * returns pointer to one L2 line by index.
 */
static double *score_cache_get_l2_line(AcoScoreCache *cache, int idx) {
  return cache->l2_rows + (size_t)idx * (size_t)cache->line_len;
}

/*
 * Function:  score_cache_get_row
 * ------------------------------
 * looks up one desirability row in layered cache and fills it on miss.
 *
 *  cache: cache instance
 *  current: row key
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *
 *  returns: pointer to cached row on success
 *           NULL when cache is unavailable for this request
 */
static const double *score_cache_get_row(AcoScoreCache *cache, int current,
                                         double **tau, double **eta,
                                         double alpha, double beta) {
  if (!cache || current < 0 || current > cache->n || cache->l1_lines <= 0) {
    return NULL;
  }

  int l1_idx = current % cache->l1_lines;
  if (cache->l1_keys[l1_idx] == current) {
    cache->l1_hits++;
    return score_cache_get_l1_line(cache, l1_idx);
  }

  double *l1_line = score_cache_get_l1_line(cache, l1_idx);
  if (cache->l2_lines > 0) {
    int l2_idx = current % cache->l2_lines;
    double *l2_line = score_cache_get_l2_line(cache, l2_idx);

    if (cache->l2_keys[l2_idx] == current) {
      cache->l2_hits++;
      score_cache_line_copy(l1_line, l2_line, cache->line_len);
      cache->l1_keys[l1_idx] = current;
      return l1_line;
    }

    cache->l3_misses++;
    score_cache_compute_row(l2_line, current, cache->n, tau, eta, alpha, beta);
    cache->l2_keys[l2_idx] = current;
    score_cache_line_copy(l1_line, l2_line, cache->line_len);
    cache->l1_keys[l1_idx] = current;
    return l1_line;
  }

  cache->l3_misses++;
  score_cache_compute_row(l1_line, current, cache->n, tau, eta, alpha, beta);
  cache->l1_keys[l1_idx] = current;
  return l1_line;
}

/*
 * Function:  aco_score_cache_create
 * ---------------------------------
 * allocates and initializes layered score-cache storage.
 *
 *  n: highest node index
 *  l1_lines: number of L1 cache lines
 *  l2_lines: number of L2 cache lines
 *
 *  returns: cache pointer on success
 *           NULL on allocation failure or invalid input
 */
AcoScoreCache *aco_score_cache_create(int n, int l1_lines, int l2_lines) {
  if (n < 0) {
    return NULL;
  }

  if (l1_lines < 1) {
    l1_lines = 1;
  }
  if (l2_lines < 0) {
    l2_lines = 0;
  }

  AcoScoreCache *cache = calloc(1, sizeof(*cache));
  if (!cache) {
    return NULL;
  }

  cache->n = n;
  cache->line_len = n + 1;
  cache->l1_lines = l1_lines;
  cache->l2_lines = l2_lines;

  cache->l1_keys = malloc((size_t)l1_lines * sizeof(int));
  cache->l1_rows =
      calloc((size_t)l1_lines * (size_t)cache->line_len, sizeof(double));
  if (!cache->l1_keys || !cache->l1_rows) {
    aco_score_cache_free(cache);
    return NULL;
  }

  if (l2_lines > 0) {
    cache->l2_keys = malloc((size_t)l2_lines * sizeof(int));
    cache->l2_rows =
        calloc((size_t)l2_lines * (size_t)cache->line_len, sizeof(double));
    if (!cache->l2_keys || !cache->l2_rows) {
      aco_score_cache_free(cache);
      return NULL;
    }
  }

  aco_score_cache_invalidate(cache);
  return cache;
}

/*
 * Function:  aco_score_cache_invalidate
 * -------------------------------------
 * marks all L1/L2 cache lines as invalid after pheromone updates.
 *
 *  cache: cache instance
 *
 *  returns: nothing
 */
void aco_score_cache_invalidate(AcoScoreCache *cache) {
  if (!cache) {
    return;
  }

  for (int i = 0; i < cache->l1_lines; ++i) {
    cache->l1_keys[i] = -1;
  }
  for (int i = 0; i < cache->l2_lines; ++i) {
    cache->l2_keys[i] = -1;
  }
}

/*
 * Function:  aco_score_cache_reset_stats
 * --------------------------------------
 * resets hit/miss counters to zero.
 *
 *  cache: cache instance
 *
 *  returns: nothing
 */
void aco_score_cache_reset_stats(AcoScoreCache *cache) {
  if (!cache) {
    return;
  }
  cache->l1_hits = 0;
  cache->l2_hits = 0;
  cache->l3_misses = 0;
}

/*
 * Function:  aco_score_cache_get_stats
 * ------------------------------------
 * snapshots cache statistics into the provided output structure.
 *
 *  cache: cache instance
 *  out: destination stats structure
 *
 *  returns: nothing
 */
void aco_score_cache_get_stats(const AcoScoreCache *cache, AcoCacheStats *out) {
  if (!out) {
    return;
  }

  out->l1_hits = 0;
  out->l2_hits = 0;
  out->l3_misses = 0;
  if (!cache) {
    return;
  }

  out->l1_hits = cache->l1_hits;
  out->l2_hits = cache->l2_hits;
  out->l3_misses = cache->l3_misses;
}

/*
 * Function:  aco_score_cache_free
 * -------------------------------
 * releases all memory owned by a score cache instance.
 *
 *  cache: cache instance to free; NULL is accepted
 *
 *  returns: nothing
 */
void aco_score_cache_free(AcoScoreCache *cache) {
  if (!cache) {
    return;
  }

  free(cache->l2_rows);
  free(cache->l2_keys);
  free(cache->l1_rows);
  free(cache->l1_keys);
  free(cache);
}

/*
 * Function:  rand01
 * -----------------
 * returns a pseudo-random value in [0, 1] using the C runtime RNG.
 *
 *  returns: random double in [0, 1]
 */
static double rand01(void) {
  return (double)rand() / (double)RAND_MAX;
}

/*
 * Function:  aco_rand01_state
 * ---------------------------
 * generates random values in [0, 1] from a local xorshift RNG state when
 * provided; otherwise falls back to rand01().
 *
 *  state: optional RNG state pointer
 *
 *  returns: pseudo-random double in [0, 1]
 */
double aco_rand01_state(unsigned int *state) {
  if (!state) {
    return rand01();
  }

  unsigned int x = *state;
  if (x == 0u) {
    x = 1u;
  }

  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;

  return (double)x / (double)UINT_MAX;
}

/*
 * Function:  aco_make_ant_seed
 * ----------------------------
 * deterministically mixes base seed, iteration index, and ant index to build
 * a non-zero per-ant seed.
 *
 *  base_seed: user-provided base seed
 *  iter: iteration index
 *  ant_index: ant index
 *
 *  returns: non-zero seed value
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter,
                               int ant_index) {
  unsigned int x = base_seed ? base_seed : 1u;
  x ^= 0xa511e9b3u;
  x ^= (unsigned int)(iter + 1) * 0x9e3779b1u;
  x ^= (unsigned int)(ant_index + 1) * 0xc2b2ae3du;
  return x ? x : 1u;
}

/*
 * Function:  aco_select_next
 * --------------------------
 * performs roulette-wheel selection among unvisited nodes using scores:
 * score(node) = tau[current][node]^alpha * eta[current][node]^beta.
 *
 *  current: current node id
 *  unvisited_nodes: compact list of currently unvisited customers
 *  unvisited_count: number of valid entries in unvisited_nodes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  roulette_r: random value in [0, 1] for thresholding
 *  candidate_scores: scratch array to store per-candidate scores
 *  selected_index: optional output index in unvisited_nodes
 *  score_cache: optional layered cache for score rows
 *
 *  returns: selected customer id on success
 *           0 if unvisited_count is non-positive
 */
int aco_select_next(int current, const int *unvisited_nodes,
                    int unvisited_count, double **tau, double **eta,
                    double alpha, double beta, double roulette_r,
                    double *candidate_scores, int *selected_index,
                    AcoScoreCache *score_cache) {
  if (unvisited_count <= 0) {
    if (selected_index) {
      *selected_index = -1;
    }
    return 0;
  }

  const double *score_row =
      score_cache_get_row(score_cache, current, tau, eta, alpha, beta);
  double *tau_row = tau[current];
  double *eta_row = eta[current];
  double denom = 0.0;

  for (int idx = 0; idx < unvisited_count; ++idx) {
    int node = unvisited_nodes[idx];
    double score = 0.0;
    if (score_row) {
      score = score_row[node];
    } else {
      double tau_term = fast_pow_nonneg(tau_row[node], alpha);
      double eta_term = fast_pow_nonneg(eta_row[node], beta);
      score = tau_term * eta_term;
    }
    candidate_scores[idx] = score;
    denom += score;
  }

  double threshold = roulette_r * denom;
  double cumulative = 0.0;
  int chosen_idx = unvisited_count - 1;
  for (int idx = 0; idx < unvisited_count; ++idx) {
    cumulative += candidate_scores[idx];
    if (cumulative >= threshold) {
      chosen_idx = idx;
      break;
    }
  }

  if (selected_index) {
    *selected_index = chosen_idx;
  }
  return unvisited_nodes[chosen_idx];
}

/*
 * Function:  aco_build_ant_solution
 * ---------------------------------
 * builds one ant solution route-by-route.
 * algorithm outline:
 * 1) initialize unvisited list [1..n] and pre-generate random draws
 * 2) for each route, repeatedly select next customer with aco_select_next
 * 3) remove selected customer from unvisited list with swap-remove
 * 4) after K routes, append any remaining customers to the last route
 *
 *  sol: solution object to fill
 *  n: number of customers
 *  K: number of routes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  vehicle_capacity_customers: max customers per route (unit demand); <=0 means unlimited
 *  score_cache: optional layered cache for score rows
 *  rng_state: RNG state used to draw random numbers
 *  unvisited_nodes: scratch array of size >= n
 *  candidate_scores: scratch array of size >= n
 *  random_draws: scratch array of size >= n
 *
 *  returns: nothing
 */
void aco_build_ant_solution(
    Solution *sol, int n, int K, double **tau, double **eta, double alpha,
    double beta, int vehicle_capacity_customers, AcoScoreCache *score_cache,
    unsigned int *rng_state, int *unvisited_nodes, double *candidate_scores,
    double *random_draws) {
  solution_reset(sol);

  int route_customer_cap = vehicle_capacity_customers;
  if (route_customer_cap <= 0) {
    route_customer_cap = n;
  }

  for (int i = 0; i < n; ++i) {
    unvisited_nodes[i] = i + 1;
    random_draws[i] = aco_rand01_state(rng_state);
  }

  int unvisited_count = n;
  int draw_index = 0;

  for (int vehicle = 1; vehicle <= K; ++vehicle) {
    Route *r = &sol->routes[vehicle - 1];
    route_append(r, 0);
    int current = 0;
    int assigned_customers = 0;
    int remaining_vehicles = K - vehicle;
    int future_capacity = remaining_vehicles * route_customer_cap;

    while (unvisited_count > 0 && unvisited_count > future_capacity &&
           assigned_customers < route_customer_cap) {
      double roulette_r = random_draws[draw_index++];
      int selected_idx = -1;
      int next = aco_select_next(current, unvisited_nodes, unvisited_count,
                                 tau, eta, alpha, beta, roulette_r,
                                 candidate_scores, &selected_idx, score_cache);
      if (next <= 0) {
        break;
      }

      route_append(r, next);
      ++assigned_customers;
      --unvisited_count;
      unvisited_nodes[selected_idx] = unvisited_nodes[unvisited_count];
      current = next;
    }

    route_append(r, 0);
  }

  if (unvisited_count > 0) {
    Route *last = &sol->routes[K - 1];
    if (last->len > 0 && last->nodes[last->len - 1] == 0) {
      --last->len;
    }

    int last_customers = last->len > 0 ? (last->len - 1) : 0;

    while (unvisited_count > 0 && last_customers < route_customer_cap) {
      route_append(last, unvisited_nodes[--unvisited_count]);
      ++last_customers;
    }

    route_append(last, 0);
  }
}
