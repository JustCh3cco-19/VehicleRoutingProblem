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


/**
 * @brief Executes `align_up_size`.
 * @param value Function parameter.
 * @param alignment Function parameter.
 * @return Function result.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}


/**
 * @brief Executes `aligned_calloc_bytes`.
 * @param bytes Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `clamp_int`.
 * @param x Function parameter.
 * @param lo Function parameter.
 * @param hi Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `choose_auto_total_ants`.
 * @param n Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `fast_pow_nonneg`.
 * @param base Function parameter.
 * @param exponent Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `aligned_row_stride`.
 * @param cols Function parameter.
 * @param elem_size Function parameter.
 * @return Function result.
 */
static int aligned_row_stride(int cols, size_t elem_size) {
  size_t row_bytes = (size_t)cols * elem_size;
  size_t padded = align_up_size(row_bytes, ACO_ALIGNMENT);
  return (int)(padded / elem_size);
}


/**
 * @brief Executes `choose_candidate_count`.
 * @param n Function parameter.
 * @return Function result.
 */
static int choose_candidate_count(int n) { return n; }

/**
 * @brief Executes `wall_time_seconds`.
 * @return Function result.
 */
static double wall_time_seconds(void) {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

/**
 * @brief Parses a percentage threshold into a relative fraction.
 * @param s Input percentage string, e.g. "10" means 10%.
 * @param default_fraction Fallback relative fraction.
 * @return Relative fraction used internally, e.g. 0.10 for 10%.
 */
static double parse_min_rel_improvement_percent(const char *s,
                                                double default_fraction) {
  if (!s || !*s) {
    return default_fraction;
  }

  double percent = atof(s);
  if (percent <= 0.0) {
    return default_fraction;
  }
  return percent / 100.0;
}

/**
 * @brief Executes `load_timer_directives`.
 * @param max_runtime_sec Function parameter.
 * @param max_stagnation_epochs Function parameter.
 * @param min_rel_improvement Function parameter.
 */
static void load_timer_directives(double *max_runtime_sec,
                                  int *max_stagnation_epochs,
                                  double *min_rel_improvement) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  const char *s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");

  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_epochs =
      (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = parse_min_rel_improvement_percent(s_rel, 1e-3);

  if (*max_stagnation_epochs < 0) {
    *max_stagnation_epochs = 0;
  }
  if (*min_rel_improvement <= 0.0) {
    *min_rel_improvement = 1e-3;
  }
}

/**
 * @brief Executes `is_significant_improvement`.
 * @param prev_best Function parameter.
 * @param new_best Function parameter.
 * @param min_rel_improvement Function parameter.
 * @return Function result.
 */
static int is_significant_improvement(double prev_best, double new_best,
                                      double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) {
    return (new_best < prev_best);
  }
  if (new_best >= prev_best - ACO_EPS) {
    return 0;
  }
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, ACO_EPS);
  return rel_gain + ACO_EPS >= min_rel_improvement;
}


/**
 * @brief Executes `seq_shared_free`.
 * @param shared Function parameter.
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


/**
 * @brief Executes `seq_shared_init`.
 * @param shared Function parameter.
 * @param n Function parameter.
 * @return Function result.
 */
static int seq_shared_init(SeqShared *shared, int n) {
  if (!shared || n < 1) {
    return 0;
  }

  memset(shared, 0, sizeof(*shared));
  shared->n = n;
  shared->candidate_k = choose_candidate_count(n);
  shared->candidate_k = clamp_int(shared->candidate_k, 1, n);
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


/**
 * @brief Executes `seq_shared_build_candidates`.
 * @param shared Function parameter.
 * @param c Function parameter.
 * @param beta Function parameter.
 */
static void seq_shared_build_candidates(SeqShared *shared, double **c,
                                        double beta) {
  int n = shared->n;
  int stride = shared->stride;

  for (int i = 0; i <= n; ++i) {
    int *cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
    float *eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
    int pos = 0;

    for (int node = 1; node <= n; ++node) {
      if (node == i) {
        continue;
      }
      cand_row[pos] = node;
      {
        double eta = 1.0 / (c[i][node] + ACO_EPS);
        eta_row[pos] = (float)fast_pow_nonneg(eta, beta);
      }
      ++pos;
    }

    for (int t = pos; t < stride; ++t) {
      cand_row[t] = 0;
      eta_row[t] = 0.0f;
    }
  }
}


/**
 * @brief Executes `update_score_row_alpha1`.
 * @param cand_row Function parameter.
 * @param eta_row Function parameter.
 * @param score_row Function parameter.
 * @param tau_row Function parameter.
 * @param k Function parameter.
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


/**
 * @brief Executes `update_score_row_alpha2`.
 * @param cand_row Function parameter.
 * @param eta_row Function parameter.
 * @param score_row Function parameter.
 * @param tau_row Function parameter.
 * @param k Function parameter.
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


/**
 * @brief Executes `seq_shared_update_scores`.
 * @param shared Function parameter.
 * @param tau Function parameter.
 * @param alpha Function parameter.
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


/**
 * @brief Executes `visited_is_set`.
 * @param visited Function parameter.
 * @param node Function parameter.
 * @return Function result.
 */
static inline int visited_is_set(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >>
                ((unsigned int)node & 63u)) &
               1u);
}


/**
 * @brief Executes `visited_set`.
 * @param visited Function parameter.
 * @param node Function parameter.
 */
static inline void visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |= (uint64_t)1u
                                     << ((unsigned int)node & 63u);
}


/**
 * @brief Executes `find_nearest_unvisited`.
 * @param shared Function parameter.
 * @param current Function parameter.
 * @param visited Function parameter.
 * @param c Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `select_next_customer`.
 * @param shared Function parameter.
 * @param current Function parameter.
 * @param visited Function parameter.
 * @param c Function parameter.
 * @param rng_state Function parameter.
 * @return Function result.
 */
static int select_next_customer(const SeqShared *shared, int current,
                                const uint64_t *restrict visited,
                                double **restrict c,
                                unsigned int *restrict rng_state) {
  const int *cand_row =
      shared->candidate_idx + (size_t)current * (size_t)shared->stride;
  const float *score_row =
      shared->score + (size_t)current * (size_t)shared->stride;
  int k = shared->candidate_k;

  double denom = 0.0;
  for (int t = 0; t < k; ++t) {
    int node = cand_row[t];
    if (node <= 0) {
      continue;
    }
    if (visited_is_set(visited, node)) {
      continue;
    }
    double w = (double)score_row[t];
    if (w > 0.0) {
      denom += w;
    }
  }

  if (denom > 0.0) {
    double threshold = aco_rand01_state(rng_state) * denom;
    double cumulative = 0.0;
    int last_valid = 0;

    for (int t = 0; t < k; ++t) {
      int node = cand_row[t];
      if (node <= 0) {
        continue;
      }
      if (visited_is_set(visited, node)) {
        continue;
      }
      double w = (double)score_row[t];
      if (w <= 0.0) {
        continue;
      }

      cumulative += w;
      last_valid = node;
      if (cumulative >= threshold) {
        return node;
      }
    }

    if (last_valid > 0) {
      return last_valid;
    }
  }

  return find_nearest_unvisited(shared, current, visited, c);
}


/**
 * @brief Executes `seq_workspace_init`.
 * @param ws Function parameter.
 * @param K Function parameter.
 * @param n Function parameter.
 * @param visited_words Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `seq_workspace_free`.
 * @param ws Function parameter.
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


/**
 * @brief Executes `build_ant_solution`.
 * @param ws Function parameter.
 * @param shared Function parameter.
 * @param K Function parameter.
 * @param vehicle_capacity_customers Function parameter.
 * @param c Function parameter.
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


/**
 * @brief Executes `aco_vrp_run_with_timer`.
 * @param n Function parameter.
 * @param K Function parameter.
 * @param vehicle_capacity_customers Function parameter.
 * @param m Function parameter.
 * @param c Function parameter.
 * @param alpha Function parameter.
 * @param beta Function parameter.
 * @param rho Function parameter.
 * @param tau0 Function parameter.
 * @param Q Function parameter.
 * @param seed Function parameter.
 * @param best_solution Function parameter.
 * @param best_cost Function parameter.
 * @param max_runtime_sec Function parameter.
 * @param max_stagnation_epochs Function parameter.
 * @param min_rel_improvement Function parameter.
 */
static void aco_vrp_run_with_timer(int n, int K, int vehicle_capacity_customers,
                                   int m, double **c, double alpha,
                                   double beta, double rho, double tau0,
                                   double Q, unsigned int seed,
                                   Solution *best_solution, double *best_cost,
                                   double max_runtime_sec,
                                   int max_stagnation_epochs,
                                   double min_rel_improvement) {
  int total_m = m;
  if (total_m <= 0) {
    total_m = choose_auto_total_ants(n);
  }
  if (min_rel_improvement <= 0.0) {
    min_rel_improvement = 1e-3;
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
  int stagnation_trigger =
      (max_stagnation_epochs > 0) ? (max_stagnation_epochs / 2) : 32;
  if (stagnation_trigger < 4) {
    stagnation_trigger = 4;
  }

  const double iter_deposit_weight = 0.3;
  const double global_deposit_weight = 0.7;
  double tau_max = tau0;
  double tau_min = tau0 * 0.05;
  double start_wall = wall_time_seconds();
  int no_improve_epochs = 0;

  for (int iter = 0;; ++iter) {
    double iter_start_wall = wall_time_seconds();
    if (max_runtime_sec > 0.0 &&
        (iter_start_wall - start_wall) >= max_runtime_sec) {
      break;
    }

    seq_shared_update_scores(&shared, tau, alpha);

    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;

    for (int ant = 0; ant < total_m; ++ant) {
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

    int improved_global = 0;
    if (is_significant_improvement(*best_cost, iter_best_cost,
                                   min_rel_improvement)) {
      *best_cost = iter_best_cost;
      solution_copy(best_solution, iter_best);
      stagnation_iters = 0;
      no_improve_epochs = 0;
      improved_global = 1;

      if (*best_cost > ACO_EPS) {
        tau_max = 1.0 / ((1.0 - rho) * (*best_cost));
        tau_min = tau_max * 0.05;
      }
    } else {
      ++stagnation_iters;
      ++no_improve_epochs;
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
    if (!improved_global && max_stagnation_epochs > 0 &&
        no_improve_epochs >= max_stagnation_epochs) {
      break;
    }
  }

  seq_workspace_free(&ws);
  seq_shared_free(&shared);
  solution_free(iter_best);
  matrix_free(tau);
}

/**
 * @brief Runs the sequential ACO solver with auto-derived vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param m Number of ants (0 enables backend auto-tuning).
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
void aco_vrp(int n, int K, int m, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost) {
  int vehicle_capacity_customers =
      (K > 0) ? (int)(((long long)120 * (long long)n +
                       (long long)100 * (long long)K - 1) /
                      ((long long)100 * (long long)K))
              : n;
  aco_vrp_with_capacity(n, K, vehicle_capacity_customers, m, c, alpha,
                        beta, rho, tau0, Q, seed, best_solution, best_cost);
}

/**
 * @brief Runs the sequential ACO solver with explicit vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param vehicle_capacity_customers Per-vehicle customer capacity.
 * @param m Number of ants (0 enables backend auto-tuning).
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
void aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers, int m,
                           double **c, double alpha, double beta,
                           double rho, double tau0, double Q,
                           unsigned int seed, Solution *best_solution,
                           double *best_cost) {
  double max_runtime_sec = 0.0;
  int max_stagnation_epochs = 0;
  double min_rel_improvement = 1e-3;
  load_timer_directives(&max_runtime_sec, &max_stagnation_epochs,
                        &min_rel_improvement);
  aco_vrp_run_with_timer(n, K, vehicle_capacity_customers, m, c, alpha,
                         beta, rho, tau0, Q, seed, best_solution, best_cost,
                         max_runtime_sec, max_stagnation_epochs,
                         min_rel_improvement);
}
