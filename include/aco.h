#ifndef ACO_H
#define ACO_H

#include "solution.h"
#include <stddef.h>
#include <stdint.h>

#define ACO_EPS 1e-9

/**
 * @brief Cache used for fast score lookups in ACO transitions.
 */
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

/**
 * @brief Aggregated cache hit/miss statistics.
 */
typedef struct {
  size_t l1_hits;
  size_t l2_hits;
  size_t l3_misses;
} AcoCacheStats;

/**
 * @brief Shared OpenMP/MPI structures for candidate lists and heuristics.
 */
typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
} AcoRankShared;

/**
 * @brief Per-thread workspace for OpenMP/MPI solver execution.
 */
typedef struct {
  Solution *sol;
  Solution *thread_best;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} AcoThreadWorkspace;

/**
 * @brief Shared structures used by the sequential backend.
 */
typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
  float *score;
} SeqShared;

/**
 * @brief Per-ant workspace used by the sequential backend.
 */
typedef struct {
  Solution *sol;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} SeqWorkspace;

typedef enum {
  ACO_OK = 0,
  ACO_ERR_INVALID_INPUT = 1,
  ACO_ERR_ALLOCATION = 2,
  ACO_ERR_NO_SOLUTION = 3,
  ACO_ERR_BACKEND = 4
} AcoStatus;

const char *aco_status_string(AcoStatus status);

/**
 * @brief Runs the ACO solver with an auto-derived per-vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param m Number of ants (0 enables backend default tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp(int n, int K, int m, double **c, double alpha, double beta,
                  double rho, double tau0, double Q, unsigned int seed,
                  Solution *best_solution, double *best_cost);

/**
 * @brief Runs the ACO solver with an explicit per-vehicle customer capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param vehicle_capacity_customers Max customers served by each vehicle.
 * @param m Number of ants (0 enables backend default tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers,
                                int m, double **c, double alpha, double beta,
                                double rho, double tau0, double Q,
                                unsigned int seed, Solution *best_solution,
                                double *best_cost);

/**
 * @brief Returns a uniform random value in [0, 1) using the provided RNG state.
 * @param state Mutable RNG state.
 * @return Random value in [0, 1).
 */
double aco_rand01_state(unsigned int *state);

/**
 * @brief Builds a deterministic per-ant seed from base seed, iteration and ant id.
 * @param base_seed Global base seed.
 * @param iter Iteration index.
 * @param ant_index Ant index.
 * @return Derived seed for the ant.
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter, int ant_index);

/**
 * @brief Allocates an ACO score cache.
 * @param n Number of customers.
 * @param l1_lines Number of L1 cache lines.
 * @param l2_lines Number of L2 cache lines.
 * @return Allocated cache handle or NULL on failure.
 */
AcoScoreCache *aco_score_cache_create(int n, int l1_lines, int l2_lines);

/**
 * @brief Invalidates all cache entries.
 * @param cache Cache handle.
 */
void aco_score_cache_invalidate(AcoScoreCache *cache);

/**
 * @brief Resets cache statistics counters to zero.
 * @param cache Cache handle.
 */
void aco_score_cache_reset_stats(AcoScoreCache *cache);

/**
 * @brief Copies cache statistics into the provided output structure.
 * @param cache Cache handle.
 * @param out Output stats structure.
 */
void aco_score_cache_get_stats(const AcoScoreCache *cache, AcoCacheStats *out);

/**
 * @brief Frees a cache allocated with aco_score_cache_create().
 * @param cache Cache handle.
 */
void aco_score_cache_free(AcoScoreCache *cache);

#endif
