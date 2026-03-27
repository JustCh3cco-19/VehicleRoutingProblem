#ifndef ACO_H
#define ACO_H

#include "solution.h"

#include <stddef.h>
#include <stdint.h>

#define ACO_EPS 1e-9

/*
 * Struct:  AcoScoreCache
 * ----------------------
 * layered row cache for tau^alpha * eta^beta scores used in probabilistic
 * node selection.
 *
 *  n: highest node index handled by cache rows
 *  line_len: number of score entries per cached row
 *  l1_lines: number of direct-mapped lines in L1 cache
 *  l2_lines: number of direct-mapped lines in L2 cache
 *  l1_keys: row-key tags for L1 lines
 *  l2_keys: row-key tags for L2 lines
 *  l1_rows: contiguous L1 row storage
 *  l2_rows: contiguous L2 row storage
 *  l1_hits: accumulated L1 hit count
 *  l2_hits: accumulated L2 hit count
 *  l3_misses: accumulated misses requiring row recomputation
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

/*
 * Struct:  AcoCacheStats
 * ----------------------
 * immutable snapshot of cache access counters for reporting.
 *
 *  l1_hits: number of L1 cache hits
 *  l2_hits: number of L2 cache hits
 *  l3_misses: number of misses requiring recomputation
 */
typedef struct {
  size_t l1_hits;
  size_t l2_hits;
  size_t l3_misses;
} AcoCacheStats;

/*
 * Struct:  AcoRankShared
 * ----------------------
 * per-rank shared workspace reused by all OpenMP threads of one MPI rank.
 *
 *  n: number of customers
 *  candidate_k: number of nearest-neighbor candidates per node
 *  stride: row stride for flattened row-major arrays
 *  visited_words: number of uint64 words for visited bitsets
 *  candidate_idx: candidate list matrix [node][candidate_k]
 *  eta_beta: precomputed eta^beta row-major matrix
 *  score: thread-shared scratch score matrix/buffer
 *  ls_pos: local-search scratch positions
 *  ls_node_route: node->route mapping scratch
 *  ls_node_pos: node->position-in-route mapping scratch
 */
typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
  float *score;
  int *ls_pos;
  int *ls_node_route;
  int *ls_node_pos;
} AcoRankShared;

/*
 * Struct:  AcoThreadWorkspace
 * ---------------------------
 * private workspace owned by one OpenMP thread in parallel solver.
 *
 *  sol: current thread ant solution buffer
 *  thread_best: best solution found by the thread
 *  visited: thread-local visited bitset
 *  route_loads: thread-local per-route load counters
 *  rng_state: thread-local RNG state
 */
typedef struct {
  Solution *sol;
  Solution *thread_best;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} AcoThreadWorkspace;

/*
 * Struct:  SeqShared
 * ------------------
 * sequential solver shared workspace reused across iterations.
 *
 *  n: number of customers
 *  candidate_k: number of nearest-neighbor candidates per node
 *  stride: row stride for flattened row-major arrays
 *  visited_words: number of uint64 words for visited bitsets
 *  candidate_idx: candidate list matrix [node][candidate_k]
 *  eta_beta: precomputed eta^beta row-major matrix
 *  score: scratch score matrix/buffer
 *  ls_pos: local-search scratch positions
 *  ls_node_route: node->route mapping scratch
 *  ls_node_pos: node->position-in-route mapping scratch
 */
typedef struct {
  int n;
  int candidate_k;
  int stride;
  int visited_words;
  int *candidate_idx;
  float *eta_beta;
  float *score;
  int *ls_pos;
  int *ls_node_route;
  int *ls_node_pos;
} SeqShared;

/*
 * Struct:  SeqWorkspace
 * ---------------------
 * private sequential worker workspace reused across ants/iterations.
 *
 *  sol: current ant solution buffer
 *  visited: visited bitset
 *  route_loads: per-route load counters
 *  rng_state: RNG state
 */
typedef struct {
  Solution *sol;
  uint64_t *visited;
  int *route_loads;
  unsigned int rng_state;
} SeqWorkspace;

/*
 * Function:  aco_vrp
 * ------------------
 * runs Ant Colony Optimization for the Vehicle Routing Problem and writes the
 * best solution found.
 *
 *  n: number of customers (customer ids are 1..n, depot is 0)
 *  K: number of vehicles/routes
 *  m: number of ants per iteration; when m <= 0 the solver auto-selects
 *     a value based on available workers and problem size
 *  T: number of iterations
 *  c: cost matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  rho: pheromone evaporation rate
 *  tau0: initial pheromone value
 *  Q: pheromone deposit scaling factor
 *  seed: base random seed
 *  best_solution: output best solution container
 *  best_cost: output best cost
 *
 *  returns: nothing; on allocation/synchronization errors the function returns
 *           early and leaves partial progress only
 */
void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost);

/*
 * Function:  aco_rand01_state
 * ---------------------------
 * generates a random value in [0, 1] using either a local xorshift state or
 * the global rand() fallback when state is NULL.
 *
 *  state: optional pointer to per-thread/per-ant rng state
 *
 *  returns: pseudo-random double in [0, 1]
 */
double aco_rand01_state(unsigned int *state);

/*
 * Function:  aco_make_ant_seed
 * ----------------------------
 * derives a deterministic seed from base seed, iteration index, and ant index.
 *
 *  base_seed: user-provided seed
 *  iter: iteration index
 *  ant_index: ant index (local or global, depending on caller)
 *
 *  returns: non-zero mixed seed value
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter,
                               int ant_index);

/*
 * Function:  aco_score_cache_create
 * ---------------------------------
 * allocates a per-worker layered score cache:
 * - L1: direct-mapped small cache (fastest, hottest rows)
 * - L2: direct-mapped backing cache (larger working set)
 * - L3: implicit fallback path when L1/L2 miss (row recomputation)
 *
 *  n: highest node index (matrix side is n+1)
 *  l1_lines: number of L1 cache lines (>= 1 recommended)
 *  l2_lines: number of L2 cache lines (>= 0)
 *
 *  returns: cache pointer on success
 *           NULL on allocation error
 */
AcoScoreCache *aco_score_cache_create(int n, int l1_lines, int l2_lines);

/*
 * Function:  aco_score_cache_invalidate
 * -------------------------------------
 * invalidates all cached rows (required after pheromone updates).
 *
 *  cache: cache instance; NULL is allowed
 *
 *  returns: nothing
 */
void aco_score_cache_invalidate(AcoScoreCache *cache);

/*
 * Function:  aco_score_cache_reset_stats
 * --------------------------------------
 * resets L1/L2/L3 counters to zero.
 *
 *  cache: cache instance; NULL is allowed
 *
 *  returns: nothing
 */
void aco_score_cache_reset_stats(AcoScoreCache *cache);

/*
 * Function:  aco_score_cache_get_stats
 * ------------------------------------
 * snapshots cache hit/miss counters.
 *
 *  cache: cache instance; NULL is allowed
 *  out: destination stats struct
 *
 *  returns: nothing; out is zeroed when cache is NULL
 */
void aco_score_cache_get_stats(const AcoScoreCache *cache, AcoCacheStats *out);

/*
 * Function:  aco_score_cache_free
 * -------------------------------
 * releases a cache created by aco_score_cache_create.
 *
 *  cache: cache instance; NULL is allowed
 *
 *  returns: nothing
 */
void aco_score_cache_free(AcoScoreCache *cache);

/*
 * Function:  aco_select_next
 * --------------------------
 * selects the next customer from the unvisited list using roulette-wheel
 * sampling over tau^alpha * eta^beta scores.
 *
 *  current: current node id
 *  unvisited_nodes: array of unvisited customer ids
 *  unvisited_count: number of valid entries in unvisited_nodes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  roulette_r: random value in [0, 1] used for roulette threshold
 *  candidate_scores: scratch array sized at least unvisited_count
 *  selected_index: optional output index inside unvisited_nodes
 *  score_cache: optional layered cache for arc scores
 *
 *  returns: selected customer id on success
 *           0 when no unvisited nodes are available
 */
int aco_select_next(int current, const int *unvisited_nodes,
                    int unvisited_count, double **tau, double **eta,
                    double alpha, double beta, double roulette_r,
                    double *candidate_scores, int *selected_index,
                    AcoScoreCache *score_cache);

/*
 * Function:  aco_build_ant_solution
 * ---------------------------------
 * constructs one complete ant solution by repeatedly selecting customers from
 * an in-place unvisited list and assigning them to routes.
 *
 *  sol: solution output container for the current ant
 *  n: number of customers
 *  K: number of routes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  vehicle_capacity_customers: max customers assignable to each vehicle;
 *                              <=0 disables the limit
 *  score_cache: optional layered cache for arc scores
 *  rng_state: rng state for deterministic random draws
 *  unvisited_nodes: scratch array of size at least n
 *  candidate_scores: scratch array of size at least n
 *  random_draws: scratch array of size at least n
 *
 *  returns: nothing
 */
void aco_build_ant_solution(
    Solution *sol, int n, int K, double **tau, double **eta, double alpha,
    double beta, int vehicle_capacity_customers, AcoScoreCache *score_cache,
    unsigned int *rng_state, int *unvisited_nodes, double *candidate_scores,
    double *random_draws);

#endif
