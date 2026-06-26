#include "aco.h"

#include "aco_config.h"
#include "aco_internal.h"
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

enum {
  kAcoAlignment = 64,
  kSeqDefaultSparseCandidateCount = 64,
  kSeqDenseCandidateLimit = 512,
};


static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}


static void *aligned_calloc_bytes(size_t bytes) {
  size_t alloc_bytes = align_up_size(bytes, kAcoAlignment);
  void *ptr = aligned_alloc(kAcoAlignment, alloc_bytes);
  if (!ptr) {
    return NULL;
  }
  memset(ptr, 0, alloc_bytes);
  return ptr;
}


static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}


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


static int aligned_row_stride(int cols, size_t elem_size) {
  size_t row_bytes = (size_t)cols * elem_size;
  size_t padded = align_up_size(row_bytes, kAcoAlignment);
  return (int)(padded / elem_size);
}


static int choose_candidate_count(int n, int requested_candidate_k) {
  if (requested_candidate_k > 0) {
    return clamp_int(requested_candidate_k, 1, n);
  }
  if (n <= kSeqDenseCandidateLimit) {
    return n;
  }
  return clamp_int(kSeqDefaultSparseCandidateCount, 1, n);
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


static void seq_shared_free(SeqShared *shared) {
  if (!shared) {
    return;
  }
  free(shared->score);
  free(shared->eta_beta);
  free(shared->candidate_idx);
  memset(shared, 0, sizeof(*shared));
}


static int seq_shared_init(SeqShared *shared, int n, int candidate_k) {
  if (!shared || n < 1) {
    return 0;
  }

  memset(shared, 0, sizeof(*shared));
  shared->n = n;
  shared->candidate_k = choose_candidate_count(n, candidate_k);
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


static void seq_shared_build_candidates(SeqShared *shared, double **c,
                                        double beta) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

  for (int i = 0; i <= n; ++i) {
    int *cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
    float *eta_row = shared->eta_beta + (size_t)i * (size_t)stride;

    for (int t = 0; t < k; ++t) {
      cand_row[t] = 0;
      eta_row[t] = 0.0f;
    }

    for (int node = 1; node <= n; ++node) {
      if (node == i) {
        continue;
      }
      double dist = c[i][node];
      int insert_at = -1;
      for (int t = 0; t < k; ++t) {
        if (cand_row[t] == 0 || dist < c[i][cand_row[t]]) {
          insert_at = t;
          break;
        }
      }
      if (insert_at < 0) {
        continue;
      }

      for (int t = k - 1; t > insert_at; --t) {
        cand_row[t] = cand_row[t - 1];
        eta_row[t] = eta_row[t - 1];
      }

      cand_row[insert_at] = node;
      double eta = 1.0 / (dist + ACO_EPS);
      eta_row[insert_at] = (float)fast_pow_nonneg(eta, beta);
    }

    for (int t = k; t < stride; ++t) {
      cand_row[t] = 0;
      eta_row[t] = 0.0f;
    }
  }
}


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


static inline int visited_is_set(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >>
                ((unsigned int)node & 63u)) &
               1u);
}


static inline void visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |= (uint64_t)1u
                                     << ((unsigned int)node & 63u);
}


static double seq_rand01(unsigned int *state) {
  unsigned int x = *state;
  if (x == 0u) {
    x = 1u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return (double)x / 4294967295.0;
}


static bool seq_route_append(Route *r, int node) {
  if (r->len >= r->cap) {
    return false;
  }
  r->nodes[r->len] = node;
  r->len++;
  return true;
}


static int find_nearest_unvisited(const SeqShared *shared, int current,
                                  const uint64_t *restrict visited,
                                  double **restrict c) {
  int best = 0;
  double best_dist = DBL_MAX;
  int n = shared->n;
  const double *restrict c_row = c[current];
  for (int node = 1; node <= n; ++node) {
    if (visited_is_set(visited, node)) {
      continue;
    }
    double d = c_row[node];
    if (d < best_dist) {
      best_dist = d;
      best = node;
    }
  }
  return best;
}


static int select_next_customer(const SeqShared *shared, int current,
                                const uint64_t *restrict visited,
                                double **restrict c,
                                unsigned int *restrict rng_state) {
  const int *cand_row =
      shared->candidate_idx + (size_t)current * (size_t)shared->stride;
  const float *score_row =
      shared->score + (size_t)current * (size_t)shared->stride;
  int k = shared->candidate_k;

  enum { kMaxLocalCandidates = 1024 };
  if (k <= kMaxLocalCandidates) {
    int active_nodes[kMaxLocalCandidates];
    double active_scores[kMaxLocalCandidates];
    int active_count = 0;
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
        active_nodes[active_count] = node;
        active_scores[active_count] = w;
        active_count++;
      }
    }

    if (denom > 0.0) {
      double threshold = seq_rand01(rng_state) * denom;
      double cumulative = 0.0;
      int last_valid = 0;

      for (int i = 0; i < active_count; ++i) {
        double w = active_scores[i];
        cumulative += w;
        last_valid = active_nodes[i];
        if (cumulative >= threshold) {
          return active_nodes[i];
        }
      }

      if (last_valid > 0) {
        return last_valid;
      }
    }
  } else {
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
      double threshold = seq_rand01(rng_state) * denom;
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
  }

  return find_nearest_unvisited(shared, current, visited, c);
}


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


static void seq_workspace_free(SeqWorkspace *ws) {
  if (!ws) {
    return;
  }
  free(ws->route_loads);
  free(ws->visited);
  solution_free(ws->sol);
  memset(ws, 0, sizeof(*ws));
}


static bool build_ant_solution(SeqWorkspace *ws, const SeqShared *shared, int K,
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
    if (!seq_route_append(r, 0)) {
      return false;
    }

    int current = 0;
    int remaining_vehicles = K - vehicle - 1;
    int future_capacity = remaining_vehicles * route_customer_cap;

    while (remaining > 0 && remaining > future_capacity &&
           ws->route_loads[vehicle] < route_customer_cap) {
      int next =
          select_next_customer(shared, current, ws->visited, c, &ws->rng_state);
      if (next <= 0) {
        break;
      }

      if (!seq_route_append(r, next)) {
        return false;
      }
      visited_set(ws->visited, next);
      ++ws->route_loads[vehicle];
      --remaining;
      current = next;
    }

    if (!seq_route_append(r, 0)) {
      return false;
    }
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
        if (next <= 0) {
          break;
        }
        if (!seq_route_append(r, next)) {
          return false;
        }
        visited_set(ws->visited, next);
        ++ws->route_loads[vehicle];
        --remaining;
      }

      if (!seq_route_append(r, 0)) {
        return false;
      }
    }
  }

  return true;
}


static AcoStatus aco_vrp_run_with_config(int n, int K,
                                         int vehicle_capacity_customers, int m,
                                         double **c, double alpha, double beta,
                                         double rho, double tau0, double Q,
                                         unsigned int seed,
                                         Solution *best_solution,
                                         double *best_cost,
                                         const AcoRuntimeConfig *config) {
  int total_m = m;
  double max_runtime_sec = config ? config->timeout_seconds : 0.0;
  int max_stagnation_epochs = config ? config->stagnation_epochs : 0;
  double min_rel_improvement =
      config ? config->min_rel_improvement : 1e-3;
  double progress_interval_sec =
      config ? config->progress_interval_seconds : 10.0;
  int log_level = config ? config->log_level : ACO_LOG_PROGRESS;

  if (n <= 0 || K <= 0 || !c || !best_solution || !best_cost) {
    return ACO_ERR_INVALID_INPUT;
  }

  if (total_m <= 0) {
    total_m = (config && config->ants > 0) ? config->ants
                                           : choose_auto_total_ants(n);
  }
  if (min_rel_improvement <= 0.0) {
    min_rel_improvement = 1e-3;
  }

  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  SeqShared shared = {0};
  int candidate_k = config ? config->candidate_k : 0;
  int shared_ok = seq_shared_init(&shared, n, candidate_k);
  SeqWorkspace ws;
  int ws_ok = 0;
  if (shared_ok) {
    ws_ok = seq_workspace_init(&ws, K, n, shared.visited_words);
    if (ws_ok && config && config->log_level > ACO_LOG_SILENT) {
      fprintf(stderr,
              "Sequential Solver starting... (N=%d, K=%d, M=%d, candidate_k=%d, "
              "seed=%u)\n",
              n, K, total_m, shared.candidate_k, config->seed);
    }
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
    return ACO_ERR_ALLOCATION;
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
  double next_progress_wall =
      (progress_interval_sec > 0.0) ? start_wall + progress_interval_sec : 0.0;
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

      if (!build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c)) {
        continue;
      }
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
    if (log_level > ACO_LOG_SILENT && progress_interval_sec > 0.0 &&
        iter_end_wall >= next_progress_wall) {
      double elapsed = iter_end_wall - start_wall;
      if (max_runtime_sec > 0.0) {
        double remaining = max_runtime_sec - elapsed;
        if (remaining < 0.0) {
          remaining = 0.0;
        }
        fprintf(stderr,
                "[seq] elapsed %.1fs, remaining %.1fs, iter %d, best %.3f\n",
                elapsed, remaining, iter + 1, *best_cost);
      } else {
        fprintf(stderr, "[seq] elapsed %.1fs, iter %d, best %.3f\n", elapsed,
                iter + 1, *best_cost);
      }
      next_progress_wall = iter_end_wall + progress_interval_sec;
    }
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
  return (*best_cost < DBL_MAX) ? ACO_OK : ACO_ERR_NO_SOLUTION;
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
AcoStatus aco_vrp(int n, int K, int m, double **c, double alpha, double beta,
                  double rho, double tau0, double Q, unsigned int seed,
                  Solution *best_solution, double *best_cost) {
  int vehicle_capacity_customers =
      (K > 0) ? (int)(((long long)120 * (long long)n +
                       (long long)100 * (long long)K - 1) /
                      ((long long)100 * (long long)K))
              : n;
  return aco_vrp_with_capacity(n, K, vehicle_capacity_customers, m, c, alpha,
                               beta, rho, tau0, Q, seed, best_solution,
                               best_cost);
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
AcoStatus aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers,
                                int m, double **c, double alpha, double beta,
                                double rho, double tau0, double Q,
                                unsigned int seed, Solution *best_solution,
                                double *best_cost) {
  AcoRuntimeConfig config;
  aco_runtime_config_load_env(&config);
  config.ants = m;
  config.seed = seed;
  return aco_vrp_run_with_config(n, K, vehicle_capacity_customers, m, c, alpha,
                                 beta, rho, tau0, Q, seed, best_solution,
                                 best_cost, &config);
}
