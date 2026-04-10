#include "aco.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define ACO_ALIGNMENT 64u
#define ACO_MAX_CANDIDATES 64

/*
 * Function:  align_up_size
 * ------------------------
 * rounds a value up to a multiple of alignment.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}

/*
 * Function:  aligned_calloc_bytes
 * -------------------------------
 * allocates cache-line aligned zeroed memory.
 */
static void *aligned_calloc_bytes(size_t bytes) {
  size_t alloc_bytes = align_up_size(bytes, ACO_ALIGNMENT);
  void *ptr = aligned_alloc(ACO_ALIGNMENT, alloc_bytes);
  if (!ptr) {
    return NULL;
  }
  memset(ptr, 0, alloc_bytes);
  return ptr;
}

/*
 * Function:  clamp_int
 * --------------------
 * clamps an integer value within an inclusive [lo, hi] interval.
 */
static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/*
 * Function:  choose_auto_total_ants
 * ---------------------------------
 * selects an automatic ant count based on problem size and available workers.
 */
static int choose_auto_total_ants(int n) {
  int workers = 1;
#ifdef _OPENMP
  workers = omp_get_max_threads();
#endif
  if (workers < 1) {
    workers = 1;
  }

  int ants_per_worker = 6;
  if (n <= 2000) {
    ants_per_worker = 8;
  } else if (n > 16000) {
    ants_per_worker = 4;
  }

  int total_ants = workers * ants_per_worker;
  total_ants = clamp_int(total_ants, workers * 4, workers * 8);
  if (total_ants < 8) {
    total_ants = 8;
  }
  return total_ants;
}

/*
 * Function:  choose_stagnation_level
 * ----------------------------------
 * maps the global stagnation counter to a discrete level:
 * 0 = low, 1 = medium, 2 = high.
 */
static int choose_stagnation_level(int stagnation_iters,
                                   int stagnation_trigger) {
  if (stagnation_iters >= 2 * stagnation_trigger) {
    return 2;
  }
  if (stagnation_iters >= stagnation_trigger) {
    return 1;
  }
  return 0;
}

/*
 * Function:  choose_ant_batch_size
 * --------------------------------
 * picks a batch size for adaptive in-iteration ant evaluation.
 */
static int choose_ant_batch_size(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 0;
  }

  int divisor = 8;
  if (stagnation_level <= 0) {
    divisor = 6;
  } else if (stagnation_level >= 2) {
    divisor = 12;
  }

  int batch = total_ants / divisor;
  return clamp_int(batch, 1, total_ants);
}

/*
 * Function:  choose_min_ants_before_stop
 * --------------------------------------
 * picks the minimum ants to evaluate before allowing early stop.
 */
static int choose_min_ants_before_stop(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 0;
  }

  int min_ants = total_ants / 4;
  if (stagnation_level == 1) {
    min_ants = total_ants / 2;
  } else if (stagnation_level >= 2) {
    min_ants = (3 * total_ants) / 4;
  }
  return clamp_int(min_ants, 1, total_ants);
}

/*
 * Function:  choose_no_improve_patience
 * -------------------------------------
 * picks tolerated non-improving batches before early stop.
 */
static int choose_no_improve_patience(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 1;
  }

  int patience = (total_ants >= 128) ? 3 : 2;
  if (stagnation_level == 1) {
    ++patience;
  } else if (stagnation_level >= 2) {
    patience += 2;
  }
  return clamp_int(patience, 1, 6);
}

/*
 * Function:  fast_pow_nonneg
 * --------------------------
 * evaluates base^exponent with fast paths for common exponents.
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
 * Function:  aligned_row_stride
 * -----------------------------
 * computes row stride in elements with 64-byte row alignment.
 */
static int aligned_row_stride(int cols, size_t elem_size) {
  size_t row_bytes = (size_t)cols * elem_size;
  size_t padded = align_up_size(row_bytes, ACO_ALIGNMENT);
  return (int)(padded / elem_size);
}

/*
 * Function:  choose_candidate_count
 * ---------------------------------
 * chooses nearest-neighbor candidate count.
 */
static int choose_candidate_count(int n) {
  if (n <= 8) {
    return n;
  }
  if (n <= 256) {
    return 16;
  }
  if (n <= 4096) {
    return 24;
  }
  return 32;
}

static double wall_time_seconds(void) {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void load_timer_directives(double *max_runtime_sec,
                                  int *max_stagnation_iters,
                                  double *improve_eps) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_ITERS");
  const char *s_eps = getenv("ACO_SOLVER_IMPROVE_EPS");

  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_iters =
      (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *improve_eps = (s_eps && *s_eps) ? atof(s_eps) : ACO_EPS;

  if (*max_stagnation_iters < 0) {
    *max_stagnation_iters = 0;
  }
  if (*improve_eps <= 0.0) {
    *improve_eps = ACO_EPS;
  }
}

/*
 * Function:  seq_shared_free
 * --------------------------
 * releases per-run shared candidate/score structures.
 */
static void seq_shared_free(SeqShared *shared) {
  if (!shared) {
    return;
  }
  free(shared->score);
  free(shared->eta_beta);
  free(shared->candidate_idx);
  memset(shared, 0, sizeof(*shared));
}

/*
 * Function:  seq_shared_init
 * --------------------------
 * allocates shared candidate, heuristic and score matrices.
 */
static int seq_shared_init(SeqShared *shared, int n) {
  if (!shared || n < 1) {
    return 0;
  }

  memset(shared, 0, sizeof(*shared));
  shared->n = n;
  shared->candidate_k = choose_candidate_count(n);
  shared->candidate_k = clamp_int(shared->candidate_k, 1, ACO_MAX_CANDIDATES);
  if (shared->candidate_k > n) {
    shared->candidate_k = n;
  }
  shared->stride = aligned_row_stride(shared->candidate_k, sizeof(float));
  if (shared->stride < shared->candidate_k) {
    shared->stride = shared->candidate_k;
  }
  shared->visited_words = (n / 64) + 1;

  int rows = n + 1;
  size_t elems = (size_t)rows * (size_t)shared->stride;

  shared->candidate_idx = aligned_calloc_bytes(elems * sizeof(int));
  shared->eta_beta = aligned_calloc_bytes(elems * sizeof(float));
  shared->score = aligned_calloc_bytes(elems * sizeof(float));
  if (!shared->candidate_idx || !shared->eta_beta || !shared->score) {
    seq_shared_free(shared);
    return 0;
  }

  for (size_t i = 0; i < elems; ++i) {
    shared->candidate_idx[i] = 0;
  }

  return 1;
}

/*
 * Function:  seq_shared_build_candidates
 * --------------------------------------
 * builds nearest-neighbor indices and eta^beta matrix once.
 */
static void seq_shared_build_candidates(SeqShared *shared, double **c,
                                        double beta) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

  for (int i = 0; i <= n; ++i) {
    int best_nodes[ACO_MAX_CANDIDATES];
    double best_dists[ACO_MAX_CANDIDATES];

    for (int t = 0; t < k; ++t) {
      best_nodes[t] = 0;
      best_dists[t] = DBL_MAX;
    }

    for (int node = 1; node <= n; ++node) {
      if (node == i) {
        continue;
      }

      double d = c[i][node];
      int pos = -1;
      for (int t = 0; t < k; ++t) {
        if (d < best_dists[t]) {
          pos = t;
          break;
        }
      }

      if (pos < 0) {
        continue;
      }

      for (int s = k - 1; s > pos; --s) {
        best_dists[s] = best_dists[s - 1];
        best_nodes[s] = best_nodes[s - 1];
      }
      best_dists[pos] = d;
      best_nodes[pos] = node;
    }

    int *cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
    float *eta_row = shared->eta_beta + (size_t)i * (size_t)stride;

    for (int t = 0; t < k; ++t) {
      int node = best_nodes[t];
      cand_row[t] = node;
      if (node > 0) {
        double eta = 1.0 / (c[i][node] + ACO_EPS);
        eta_row[t] = (float)fast_pow_nonneg(eta, beta);
      } else {
        eta_row[t] = 0.0f;
      }
    }

    for (int t = k; t < stride; ++t) {
      cand_row[t] = 0;
      eta_row[t] = 0.0f;
    }
  }
}

/*
 * Function:  update_score_row_alpha1
 * ----------------------------------
 * AVX2/scalar kernel for score = tau * eta_beta.
 */
static void update_score_row_alpha1(const int *restrict cand_row,
                                    const float *restrict eta_row,
                                    float *restrict score_row,
                                    const double *restrict tau_row, int k) {
  int t = 0;
#if defined(__AVX2__)
  for (; t + 4 <= k; t += 4) {
    __m128i idx = _mm_loadu_si128((const __m128i *)(cand_row + t));
    __m256d tau_v = _mm256_i32gather_pd(tau_row, idx, (int)sizeof(double));
    __m128 eta_f = _mm_loadu_ps(eta_row + t);
    __m256d eta_v = _mm256_cvtps_pd(eta_f);
    __m256d prod = _mm256_mul_pd(tau_v, eta_v);
    _mm_storeu_ps(score_row + t, _mm256_cvtpd_ps(prod));
  }
#endif
  for (; t < k; ++t) {
    int node = cand_row[t];
    score_row[t] = (node > 0) ? (float)(tau_row[node] * (double)eta_row[t]) : 0.0f;
  }
}

/*
 * Function:  update_score_row_alpha2
 * ----------------------------------
 * AVX2/scalar kernel for score = tau^2 * eta_beta.
 */
static void update_score_row_alpha2(const int *restrict cand_row,
                                    const float *restrict eta_row,
                                    float *restrict score_row,
                                    const double *restrict tau_row, int k) {
  int t = 0;
#if defined(__AVX2__)
  for (; t + 4 <= k; t += 4) {
    __m128i idx = _mm_loadu_si128((const __m128i *)(cand_row + t));
    __m256d tau_v = _mm256_i32gather_pd(tau_row, idx, (int)sizeof(double));
    __m256d tau2_v = _mm256_mul_pd(tau_v, tau_v);
    __m128 eta_f = _mm_loadu_ps(eta_row + t);
    __m256d eta_v = _mm256_cvtps_pd(eta_f);
    __m256d prod = _mm256_mul_pd(tau2_v, eta_v);
    _mm_storeu_ps(score_row + t, _mm256_cvtpd_ps(prod));
  }
#endif
  for (; t < k; ++t) {
    int node = cand_row[t];
    if (node > 0) {
      double tau_val = tau_row[node];
      score_row[t] = (float)(tau_val * tau_val * (double)eta_row[t]);
    } else {
      score_row[t] = 0.0f;
    }
  }
}

/*
 * Function:  seq_shared_update_scores
 * -----------------------------------
 * updates score matrix S = tau^alpha * eta^beta for candidate arcs.
 */
static void seq_shared_update_scores(SeqShared *shared, double **restrict tau,
                                     double alpha) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

  for (int i = 0; i <= n; ++i) {
    int *cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
    float *eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
    float *score_row = shared->score + (size_t)i * (size_t)stride;
    const double *tau_row = tau[i];

    if (alpha == 1.0) {
      update_score_row_alpha1(cand_row, eta_row, score_row, tau_row, k);
    } else if (alpha == 2.0) {
      update_score_row_alpha2(cand_row, eta_row, score_row, tau_row, k);
    } else {
      for (int t = 0; t < k; ++t) {
        int node = cand_row[t];
        if (node > 0) {
          double tau_term = fast_pow_nonneg(tau_row[node], alpha);
          score_row[t] = (float)(tau_term * (double)eta_row[t]);
        } else {
          score_row[t] = 0.0f;
        }
      }
    }

    for (int t = k; t < stride; ++t) {
      score_row[t] = 0.0f;
    }
  }
}

/*
 * Function:  visited_is_set
 * -------------------------
 * tests whether a customer has already been visited.
 */
static inline int visited_is_set(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >>
                ((unsigned int)node & 63u)) &
               1u);
}

/*
 * Function:  visited_set
 * ----------------------
 * marks a customer as visited.
 */
static inline void visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |= (uint64_t)1u
                                     << ((unsigned int)node & 63u);
}

/*
 * Function:  find_nearest_unvisited
 * ---------------------------------
 * fallback selector that returns the nearest unvisited customer.
 */
static int find_nearest_unvisited(const SeqShared *shared, int current,
                                  const uint64_t *restrict visited,
                                  double **restrict c) {
  int best = 0;
  double best_dist = DBL_MAX;
  for (int node = 1; node <= shared->n; ++node) {
    if (visited_is_set(visited, node)) {
      continue;
    }
    double d = c[current][node];
    if (d < best_dist) {
      best_dist = d;
      best = node;
    }
  }
  return best;
}

/*
 * Function:  sum_weights
 * ----------------------
 * sums a float buffer using AVX2 when available.
 */
static double sum_weights(const float *weights, int count) {
#if defined(__AVX2__)
  __m256 vsum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= count; i += 8) {
    __m256 w = _mm256_loadu_ps(weights + i);
    vsum = _mm256_add_ps(vsum, w);
  }

  float tmp[8];
  _mm256_storeu_ps(tmp, vsum);
  double sum = (double)tmp[0] + (double)tmp[1] + (double)tmp[2] +
               (double)tmp[3] + (double)tmp[4] + (double)tmp[5] +
               (double)tmp[6] + (double)tmp[7];
  for (; i < count; ++i) {
    sum += (double)weights[i];
  }
  return sum;
#else
  double sum = 0.0;
  for (int i = 0; i < count; ++i) {
    sum += (double)weights[i];
  }
  return sum;
#endif
}

/*
 * Function:  select_next_customer
 * -------------------------------
 * candidate-first roulette selection with nearest-neighbor fallback.
 */
static int select_next_customer(const SeqShared *shared, int current,
                                const uint64_t *restrict visited,
                                double **restrict c,
                                unsigned int *restrict rng_state) {
  float masked_weights[ACO_MAX_CANDIDATES];

  const int *cand_row =
      shared->candidate_idx + (size_t)current * (size_t)shared->stride;
  const float *score_row =
      shared->score + (size_t)current * (size_t)shared->stride;
  int k = shared->candidate_k;

  for (int t = 0; t < k; ++t) {
    int node = cand_row[t];
    int available = 0;
    if (node > 0) {
      uint64_t word = visited[(unsigned int)node >> 6];
      uint64_t bit = (word >> ((unsigned int)node & 63u)) & 1u;
      available = (bit == 0u);
    }
    masked_weights[t] = score_row[t] * (float)available;
  }

  double denom = sum_weights(masked_weights, k);
  if (denom > 0.0) {
    double threshold = aco_rand01_state(rng_state) * denom;
    double cumulative = 0.0;

    for (int i = 0; i < k; ++i) {
      float w = masked_weights[i];
      if (w <= 0.0f) {
        continue;
      }
      cumulative += (double)w;
      if (cumulative >= threshold) {
        return cand_row[i];
      }
    }

    for (int i = k - 1; i >= 0; --i) {
      if (masked_weights[i] > 0.0f) {
        return cand_row[i];
      }
    }
  }

  return find_nearest_unvisited(shared, current, visited, c);
}

/*
 * Function:  seq_workspace_init
 * -----------------------------
 * allocates sequential workspace used across ants.
 */
static int seq_workspace_init(SeqWorkspace *ws, int K, int n,
                              int visited_words) {
  memset(ws, 0, sizeof(*ws));

  ws->sol = solution_create(K, n);
  ws->visited = aligned_calloc_bytes((size_t)visited_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
  ws->rng_state = 1u;

  if (!ws->sol || !ws->visited || !ws->route_loads) {
    free(ws->route_loads);
    free(ws->visited);
    solution_free(ws->sol);
    memset(ws, 0, sizeof(*ws));
    return 0;
  }

  return 1;
}

/*
 * Function:  seq_workspace_free
 * -----------------------------
 * releases workspace buffers.
 */
static void seq_workspace_free(SeqWorkspace *ws) {
  if (!ws) {
    return;
  }
  free(ws->route_loads);
  free(ws->visited);
  solution_free(ws->sol);
  memset(ws, 0, sizeof(*ws));
}

/*
 * Function:  build_ant_solution
 * -----------------------------
 * builds one ant solution with bitmask-tracked visited customers.
 */
static void build_ant_solution(SeqWorkspace *ws, const SeqShared *shared, int K,
                               int vehicle_capacity_customers,
                               double **restrict c) {
  solution_reset(ws->sol);
  memset(ws->visited, 0, (size_t)shared->visited_words * sizeof(uint64_t));
  memset(ws->route_loads, 0, (size_t)K * sizeof(int));

  int route_customer_cap = vehicle_capacity_customers;
  if (route_customer_cap <= 0) {
    route_customer_cap = shared->n;
  }

  int remaining = shared->n;

  for (int vehicle = 0; vehicle < K; ++vehicle) {
    Route *r = &ws->sol->routes[vehicle];
    route_append(r, 0);

    int current = 0;
    int remaining_vehicles = K - vehicle - 1;
    int future_capacity = remaining_vehicles * route_customer_cap;

    while (remaining > 0 && remaining > future_capacity &&
           ws->route_loads[vehicle] < route_customer_cap) {
      int next =
          select_next_customer(shared, current, ws->visited, c, &ws->rng_state);
      if (next <= 0 || visited_is_set(ws->visited, next)) {
        next = find_nearest_unvisited(shared, current, ws->visited, c);
      }
      if (next <= 0 || visited_is_set(ws->visited, next)) {
        break;
      }

      route_append(r, next);
      visited_set(ws->visited, next);
      ++ws->route_loads[vehicle];
      --remaining;
      current = next;
    }

    route_append(r, 0);
  }

  if (remaining > 0) {
    for (int vehicle = K - 1; vehicle >= 0 && remaining > 0; --vehicle) {
      Route *r = &ws->sol->routes[vehicle];
      if (r->len > 0 && r->nodes[r->len - 1] == 0) {
        --r->len;
      }

      while (remaining > 0 && ws->route_loads[vehicle] < route_customer_cap) {
        int current = (r->len > 0) ? r->nodes[r->len - 1] : 0;
        int next = find_nearest_unvisited(shared, current, ws->visited, c);
        if (next <= 0 || visited_is_set(ws->visited, next)) {
          break;
        }
        route_append(r, next);
        visited_set(ws->visited, next);
        ++ws->route_loads[vehicle];
        --remaining;
      }

      route_append(r, 0);
    }
  }
}

/*
 * Function:  aco_vrp_run_with_timer
 * ------------------
 * internal sequential solver routine with timer directives already resolved.
 */
static void aco_vrp_run_with_timer(int n, int K, int vehicle_capacity_customers,
                                   int m, int T, double **c, double alpha,
                                   double beta, double rho, double tau0,
                                   double Q, unsigned int seed,
                                   Solution *best_solution, double *best_cost,
                                   double max_runtime_sec,
                                   int max_stagnation_iters,
                                   double improve_eps) {
  int total_m = m;
  if (total_m <= 0) {
    total_m = choose_auto_total_ants(n);
  }
  if (improve_eps <= 0.0) {
    improve_eps = ACO_EPS;
  }

  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  SeqShared shared = {0};
  int shared_ok = seq_shared_init(&shared, n);
  SeqWorkspace ws;
  int ws_ok = 0;
  if (shared_ok) {
    ws_ok = seq_workspace_init(&ws, K, n, shared.visited_words);
  }

  if (!tau || !iter_best || !shared_ok || !ws_ok) {
    fprintf(stderr, "allocation failure in aco_vrp\n");
    matrix_free(tau);
    solution_free(iter_best);
    if (ws_ok) {
      seq_workspace_free(&ws);
    }
    if (shared_ok) {
      seq_shared_free(&shared);
    }
    return;
  }

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      tau[i][j] = (i == j) ? 0.0 : tau0;
    }
  }

  seq_shared_build_candidates(&shared, c, beta);
  seq_shared_update_scores(&shared, tau, alpha);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  if (vehicle_capacity_customers < 1) {
    vehicle_capacity_customers = 1;
  }

  int stagnation_iters = 0;
  int stagnation_trigger = T / 4;
  if (stagnation_trigger < 4) {
    stagnation_trigger = 4;
  }

  const double iter_deposit_weight = 0.3;
  const double global_deposit_weight = 0.7;
  double tau_max = tau0;
  double tau_min = tau0 * 0.05;
  double start_wall = wall_time_seconds();
  int no_improve_iters = 0;

  for (int iter = 0; iter < T; ++iter) {
    double iter_start_wall = wall_time_seconds();
    if (max_runtime_sec > 0.0 &&
        (iter_start_wall - start_wall) >= max_runtime_sec) {
      break;
    }

    int stagnation_level =
        choose_stagnation_level(stagnation_iters, stagnation_trigger);
    int batch_size = choose_ant_batch_size(total_m, stagnation_level);
    int min_ants_before_stop =
        choose_min_ants_before_stop(total_m, stagnation_level);
    int no_improve_patience =
        choose_no_improve_patience(total_m, stagnation_level);

    seq_shared_update_scores(&shared, tau, alpha);

    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;

    int ants_evaluated = 0;
    int consecutive_no_improve_batches = 0;

    for (int batch_start = 0; batch_start < total_m; batch_start += batch_size) {
      int batch_end = batch_start + batch_size;
      if (batch_end > total_m) {
        batch_end = total_m;
      }

      double prev_best_cost = iter_best_cost;
      int prev_best_ant = iter_best_ant;

      for (int ant = batch_start; ant < batch_end; ++ant) {
        ws.rng_state = aco_make_ant_seed(seed, iter, ant);

        build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c);
        double cost = solution_cost(ws.sol, c);

        if (cost < iter_best_cost ||
            (fabs(cost - iter_best_cost) <= ACO_EPS && ant < iter_best_ant)) {
          iter_best_cost = cost;
          iter_best_ant = ant;
          solution_copy(iter_best, ws.sol);
        }
      }

      ants_evaluated += (batch_end - batch_start);
      if (iter_best_cost < prev_best_cost - ACO_EPS ||
          (fabs(iter_best_cost - prev_best_cost) <= ACO_EPS &&
           iter_best_ant < prev_best_ant)) {
        consecutive_no_improve_batches = 0;
      } else {
        ++consecutive_no_improve_batches;
      }

      if (ants_evaluated >= min_ants_before_stop &&
          consecutive_no_improve_batches >= no_improve_patience) {
        break;
      }
    }

    int improved_global = 0;
    if (iter_best_cost < (*best_cost - improve_eps)) {
      *best_cost = iter_best_cost;
      solution_copy(best_solution, iter_best);
      stagnation_iters = 0;
      no_improve_iters = 0;
      improved_global = 1;

      if (*best_cost > ACO_EPS) {
        tau_max = 1.0 / ((1.0 - rho) * (*best_cost));
        tau_min = tau_max * 0.05;
      }
    } else {
      ++stagnation_iters;
      ++no_improve_iters;
    }

    if (iter_best_cost < DBL_MAX) {
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i != j) {
            tau[i][j] *= (1.0 - rho);
          }
        }
      }

      double iter_deposit = (iter_deposit_weight * Q) / iter_best_cost;
      for (int i = 0; i < K; ++i) {
        Route *r = &iter_best->routes[i];
        for (int t = 0; t + 1 < r->len; ++t) {
          int u = r->nodes[t];
          int v = r->nodes[t + 1];
          tau[u][v] += iter_deposit;
          tau[v][u] += iter_deposit;
        }
      }

      if (*best_cost < DBL_MAX) {
        double global_deposit = (global_deposit_weight * Q) / (*best_cost);
        for (int i = 0; i < K; ++i) {
          Route *r = &best_solution->routes[i];
          for (int t = 0; t + 1 < r->len; ++t) {
            int u = r->nodes[t];
            int v = r->nodes[t + 1];
            tau[u][v] += global_deposit;
            tau[v][u] += global_deposit;
          }
        }
      }

      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i == j) {
            continue;
          }
          if (tau[i][j] < tau_min) {
            tau[i][j] = tau_min;
          } else if (tau[i][j] > tau_max) {
            tau[i][j] = tau_max;
          }
        }
      }

      if (stagnation_iters >= stagnation_trigger) {
        for (int i = 0; i <= n; ++i) {
          for (int j = 0; j <= n; ++j) {
            if (i != j) {
              tau[i][j] = 0.5 * tau[i][j] + 0.5 * tau0;
            }
          }
        }
        stagnation_iters = 0;
      }
    }

    double iter_end_wall = wall_time_seconds();
    if (max_runtime_sec > 0.0 &&
        (iter_end_wall - start_wall) >= max_runtime_sec) {
      break;
    }
    if (!improved_global && max_stagnation_iters > 0 &&
        no_improve_iters >= max_stagnation_iters) {
      break;
    }
  }

  seq_workspace_free(&ws);
  seq_shared_free(&shared);
  solution_free(iter_best);
  matrix_free(tau);
}

void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost) {
  int vehicle_capacity_customers = (K > 0) ? ((n + K - 1) / K) : n;
  aco_vrp_with_capacity(n, K, vehicle_capacity_customers, m, T, c, alpha,
                        beta, rho, tau0, Q, seed, best_solution, best_cost);
}

void aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers, int m,
                           int T, double **c, double alpha, double beta,
                           double rho, double tau0, double Q,
                           unsigned int seed, Solution *best_solution,
                           double *best_cost) {
  double max_runtime_sec = 0.0;
  int max_stagnation_iters = 0;
  double improve_eps = ACO_EPS;
  load_timer_directives(&max_runtime_sec, &max_stagnation_iters, &improve_eps);
  aco_vrp_run_with_timer(n, K, vehicle_capacity_customers, m, T, c, alpha,
                         beta, rho, tau0, Q, seed, best_solution, best_cost,
                         max_runtime_sec, max_stagnation_iters, improve_eps);
}
