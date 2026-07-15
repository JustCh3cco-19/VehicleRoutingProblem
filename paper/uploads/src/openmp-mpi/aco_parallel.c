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

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
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
 * clamps an integer value between a lower and an upper bound.
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
 * estimates a robust ant count from problem size and total worker count
 * (MPI ranks x OpenMP threads).
 */
static int choose_auto_total_ants(int n, int mpi_size) {
  int omp_threads = 1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif
  if (omp_threads < 1) {
    omp_threads = 1;
  }

  int workers = mpi_size * omp_threads;
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
#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    return MPI_Wtime();
  }
#endif
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void load_timer_directives(double *max_runtime_sec,
                                  int *max_stagnation_epochs,
                                  double *min_rel_improvement) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  const char *s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");

  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_epochs =
      (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = (s_rel && *s_rel) ? atof(s_rel) : 1e-3;

  if (*max_stagnation_epochs < 0) {
    *max_stagnation_epochs = 0;
  }
  if (*min_rel_improvement <= 0.0) {
    *min_rel_improvement = 1e-3;
  }
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

/*
 * Function:  rank_shared_free
 * ---------------------------
 * releases per-rank shared candidate/score structures.
 */
static void rank_shared_free(AcoRankShared *shared) {
  if (!shared) {
    return;
  }
  free(shared->score);
  free(shared->eta_beta);
  free(shared->candidate_idx);
  memset(shared, 0, sizeof(*shared));
}

/*
 * Function:  rank_shared_init
 * ---------------------------
 * allocates per-rank candidate, heuristic and score matrices.
 */
static int rank_shared_init(AcoRankShared *shared, int n) {
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
    rank_shared_free(shared);
    return 0;
  }

  for (size_t i = 0; i < elems; ++i) {
    shared->candidate_idx[i] = -1;
  }

  return 1;
}

/*
 * Function:  rank_shared_build_candidates
 * ---------------------------------------
 * builds nearest-neighbor indices and eta^beta matrix once per rank.
 */
static void rank_shared_build_candidates(AcoRankShared *shared, double **c,
                                         double beta) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 64)
#endif
  for (int i = 0; i <= n; ++i) {
    int best_nodes[ACO_MAX_CANDIDATES];
    double best_dists[ACO_MAX_CANDIDATES];

    for (int t = 0; t < k; ++t) {
      best_nodes[t] = -1;
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
      cand_row[t] = (node > 0) ? node : 0;
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
 * Function:  rank_shared_update_scores
 * ------------------------------------
 * updates score matrix S = tau^alpha * eta^beta for candidate arcs.
 */
static void rank_shared_update_scores(AcoRankShared *shared,
                                      double **restrict tau, double alpha) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 64)
#endif
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
#pragma omp simd
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
static int find_nearest_unvisited(const AcoRankShared *shared, int current,
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
static int select_next_customer(const AcoRankShared *shared, int current,
                                const uint64_t *restrict visited,
                                double **restrict c,
                                unsigned int *restrict rng_state) {
  float masked_weights[ACO_MAX_CANDIDATES];

  const int *cand_row =
      shared->candidate_idx + (size_t)current * (size_t)shared->stride;
  const float *score_row =
      shared->score + (size_t)current * (size_t)shared->stride;
  int k = shared->candidate_k;

  /* Dense mask application over S[current] keeps the hot path SIMD-friendly. */
#pragma omp simd
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
 * Function:  thread_workspace_free
 * --------------------------------
 * releases per-thread persistent workspace.
 */
static void thread_workspace_free(AcoThreadWorkspace *ws) {
  if (!ws) {
    return;
  }
  free(ws->route_loads);
  free(ws->visited);
  solution_free(ws->thread_best);
  solution_free(ws->sol);
  memset(ws, 0, sizeof(*ws));
}

/*
 * Function:  thread_workspace_init
 * --------------------------------
 * allocates per-thread persistent structures once per epoch.
 */
static int thread_workspace_init(AcoThreadWorkspace *ws, int K, int n,
                                 int visited_words) {
  memset(ws, 0, sizeof(*ws));

  ws->sol = solution_create(K, n);
  ws->thread_best = solution_create(K, n);
  ws->visited = aligned_calloc_bytes((size_t)visited_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
  ws->rng_state = 1u;

  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->route_loads) {
    thread_workspace_free(ws);
    return 0;
  }

  return 1;
}

/*
 * Function:  build_ant_solution
 * -----------------------------
 * builds one ant solution with bitmask-tracked visited customers.
 */
static void build_ant_solution(AcoThreadWorkspace *ws,
                               const AcoRankShared *shared, int K,
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

#ifdef USE_MPI
/*
 * Function:  solution_pack_size_ints
 * ----------------------------------
 * computes integer count required to serialize a solution.
 */
static int solution_pack_size_ints(int K, int n) {
  return 1 + K + K * (n + 2);
}

/*
 * Function:  solution_pack
 * ------------------------
 * serializes a solution into an integer buffer.
 */
static void solution_pack(const Solution *sol, int n, int *buf) {
  int route_cap = n + 2;
  buf[0] = sol->K;

  for (int i = 0; i < sol->K; ++i) {
    const Route *src = &sol->routes[i];
    int len = src->len;
    if (len < 0) {
      len = 0;
    }
    if (len > route_cap) {
      len = route_cap;
    }

    buf[1 + i] = len;

    int *dst_nodes = &buf[1 + sol->K + i * route_cap];
    memset(dst_nodes, 0, (size_t)route_cap * sizeof(int));
    if (len > 0) {
      memcpy(dst_nodes, src->nodes, (size_t)len * sizeof(int));
    }
  }
}

/*
 * Function:  solution_unpack
 * --------------------------
 * deserializes an integer buffer into an existing solution.
 */
static bool solution_unpack(Solution *sol, int n, const int *buf) {
  int route_cap = n + 2;
  if (buf[0] != sol->K) {
    return false;
  }

  for (int i = 0; i < sol->K; ++i) {
    int len = buf[1 + i];
    if (len < 0 || len > route_cap) {
      return false;
    }

    Route *dst = &sol->routes[i];
    dst->len = len;
    if (len > 0) {
      memcpy(dst->nodes, &buf[1 + sol->K + i * route_cap],
             (size_t)len * sizeof(int));
    }
  }

  return true;
}

/*
 * Function:  mpi_sync_tau_epoch
 * -----------------------------
 * synchronizes pheromone rows across ranks with non-blocking all-reduce.
 */
static int mpi_sync_tau_epoch(double **tau, int n, int mpi_size,
                              MPI_Request *reqs) {
  int side = n + 1;

  for (int i = 0; i < side; ++i) {
    int rc = MPI_Iallreduce(MPI_IN_PLACE, tau[i], side, MPI_DOUBLE, MPI_SUM,
                            MPI_COMM_WORLD, &reqs[i]);
    if (rc != MPI_SUCCESS) {
      return 0;
    }
  }

  if (MPI_Waitall(side, reqs, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    return 0;
  }

  double inv = 1.0 / (double)mpi_size;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 32)
#endif
  for (int i = 0; i < side; ++i) {
    for (int j = 0; j < side; ++j) {
      if (i == j) {
        tau[i][j] = 0.0;
      } else {
        tau[i][j] *= inv;
      }
    }
  }

  return 1;
}
#endif

/*
 * Function:  aco_vrp_run_with_timer
 * ------------------
 * internal hybrid solver routine with timer directives already resolved.
 */
static void aco_vrp_run_with_timer(int n, int K, int vehicle_capacity_customers,
                                   int m, double **c, double alpha,
                                   double beta, double rho, double tau0,
                                   double Q, unsigned int seed,
                                   Solution *best_solution, double *best_cost,
                                   double max_runtime_sec,
                                   int max_stagnation_epochs,
                                   double min_rel_improvement) {
  int mpi_rank = 0;
  int mpi_size = 1;
  bool mpi_enabled = false;
  if (min_rel_improvement <= 0.0) {
    min_rel_improvement = 1e-3;
  }

#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_enabled = (mpi_size > 1);
  }
#endif

  int total_m = m;
  if (total_m <= 0) {
    total_m = choose_auto_total_ants(n, mpi_size);
  }

  int local_m = total_m;
  int ant_offset = 0;
  if (mpi_enabled) {
    int base = (mpi_size > 0) ? (total_m / mpi_size) : total_m;
    int rem = (mpi_size > 0) ? (total_m % mpi_size) : 0;
    local_m = base + ((mpi_rank < rem) ? 1 : 0);
    ant_offset = mpi_rank * base + ((mpi_rank < rem) ? mpi_rank : rem);
  }

  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  AcoRankShared shared = {0};
  int rank_shared_ok = rank_shared_init(&shared, n);

#ifdef USE_MPI
  int packed_len = mpi_enabled ? solution_pack_size_ints(K, n) : 0;
  int *packed_solution =
      mpi_enabled ? calloc((size_t)packed_len, sizeof(int)) : NULL;
  MPI_Request *tau_sync_reqs =
      mpi_enabled ? malloc((size_t)(n + 1) * sizeof(MPI_Request)) : NULL;
#endif

  int local_ok = (tau && iter_best && rank_shared_ok) ? 1 : 0;
#ifdef USE_MPI
  if (mpi_enabled && (!packed_solution || !tau_sync_reqs)) {
    local_ok = 0;
  }
  if (mpi_enabled) {
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    local_ok = global_ok;
  }
#endif

  if (!local_ok) {
    if (!mpi_enabled || mpi_rank == 0) {
      fprintf(stderr, "allocation failure in aco_vrp\n");
    }
    matrix_free(tau);
    solution_free(iter_best);
    rank_shared_free(&shared);
#ifdef USE_MPI
    free(tau_sync_reqs);
    free(packed_solution);
#endif
    return;
  }

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      tau[i][j] = (i == j) ? 0.0 : tau0;
    }
  }

  rank_shared_build_candidates(&shared, c, beta);
  rank_shared_update_scores(&shared, tau, alpha);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  if (vehicle_capacity_customers < 1) {
    vehicle_capacity_customers = 1;
  }

#ifdef _OPENMP
  bool omp_enabled = (omp_get_max_threads() > 1 && local_m > 1);
#else
  bool omp_enabled = false;
#endif

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
    int stop_now = 0;
    if (max_runtime_sec > 0.0) {
      double now_wall = wall_time_seconds();
      stop_now = ((now_wall - start_wall) >= max_runtime_sec) ? 1 : 0;
#ifdef USE_MPI
      if (mpi_enabled) {
        int global_stop = 0;
        MPI_Allreduce(&stop_now, &global_stop, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        stop_now = global_stop;
      }
#endif
    }
    if (stop_now) {
      break;
    }

    rank_shared_update_scores(&shared, tau, alpha);

    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;
    int iter_failed = 0;

    if (omp_enabled) {
#ifdef _OPENMP
#pragma omp parallel default(shared) proc_bind(close)
      {
        AcoThreadWorkspace ws;
        double thread_best_cost = DBL_MAX;
        int thread_best_ant = INT_MAX;
        int thread_id = 0;
#ifdef _OPENMP
        thread_id = omp_get_thread_num();
#endif

        if (!thread_workspace_init(&ws, K, n, shared.visited_words)) {
#pragma omp critical
          { iter_failed = 1; }
        } else {
          ws.rng_state = aco_make_ant_seed(seed, iter, ant_offset + thread_id);
#pragma omp for schedule(guided, 1)
          for (int ant = 0; ant < local_m; ++ant) {
            int global_ant = ant_offset + ant;
            ws.rng_state ^= (unsigned int)(global_ant + 1) * 0x9e3779b1u;
            if (ws.rng_state == 0u) {
              ws.rng_state = 1u;
            }

            build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c);
            double cost = solution_cost(ws.sol, c);

            if (cost < thread_best_cost ||
                (fabs(cost - thread_best_cost) <= ACO_EPS &&
                 global_ant < thread_best_ant)) {
              thread_best_cost = cost;
              thread_best_ant = global_ant;
              solution_copy(ws.thread_best, ws.sol);
            }
          }

          if (thread_best_cost < DBL_MAX) {
#pragma omp critical
            {
              if (thread_best_cost < iter_best_cost ||
                  (fabs(thread_best_cost - iter_best_cost) <= ACO_EPS &&
                   thread_best_ant < iter_best_ant)) {
                iter_best_cost = thread_best_cost;
                iter_best_ant = thread_best_ant;
                solution_copy(iter_best, ws.thread_best);
              }
            }
          }

          thread_workspace_free(&ws);
        }
      }
#endif
    } else {
      AcoThreadWorkspace ws;
      if (!thread_workspace_init(&ws, K, n, shared.visited_words)) {
        iter_failed = 1;
      } else {
        ws.rng_state = aco_make_ant_seed(seed, iter, ant_offset);
        for (int ant = 0; ant < local_m; ++ant) {
          int global_ant = ant_offset + ant;
          ws.rng_state ^= (unsigned int)(global_ant + 1) * 0x9e3779b1u;
          if (ws.rng_state == 0u) {
            ws.rng_state = 1u;
          }

          build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c);
          double cost = solution_cost(ws.sol, c);

          if (cost < iter_best_cost ||
              (fabs(cost - iter_best_cost) <= ACO_EPS &&
               global_ant < iter_best_ant)) {
            iter_best_cost = cost;
            iter_best_ant = global_ant;
            solution_copy(iter_best, ws.sol);
          }
        }
      }
      thread_workspace_free(&ws);
    }

#ifdef USE_MPI
    if (mpi_enabled) {
      int global_failed = 0;
      MPI_Allreduce(&iter_failed, &global_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (global_failed) {
        if (mpi_rank == 0) {
          fprintf(stderr, "allocation failure during iteration\n");
        }
        matrix_free(tau);
        solution_free(iter_best);
        rank_shared_free(&shared);
        free(tau_sync_reqs);
        free(packed_solution);
        return;
      }

      int local_has_solution = (iter_best_cost < DBL_MAX) ? 1 : 0;
      int global_has_solution = 0;
      MPI_Allreduce(&local_has_solution, &global_has_solution, 1, MPI_INT,
                    MPI_MAX, MPI_COMM_WORLD);
      if (!global_has_solution) {
        continue;
      }

      double global_best_cost = DBL_MAX;
      MPI_Allreduce(&iter_best_cost, &global_best_cost, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);

      int local_ant = INT_MAX;
      if (local_has_solution &&
          fabs(iter_best_cost - global_best_cost) <= ACO_EPS) {
        local_ant = iter_best_ant;
      }

      int global_best_ant = INT_MAX;
      MPI_Allreduce(&local_ant, &global_best_ant, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      int owner_rank = mpi_size;
      if (global_best_ant >= ant_offset &&
          global_best_ant < ant_offset + local_m) {
        owner_rank = mpi_rank;
      }

      int winner_rank = mpi_size;
      MPI_Allreduce(&owner_rank, &winner_rank, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      if (mpi_rank == winner_rank) {
        solution_pack(iter_best, n, packed_solution);
      }

      MPI_Bcast(packed_solution, packed_len, MPI_INT, winner_rank,
                MPI_COMM_WORLD);

      if (mpi_rank != winner_rank &&
          !solution_unpack(iter_best, n, packed_solution)) {
        if (mpi_rank == 0) {
          fprintf(stderr, "solution synchronization failure\n");
        }
        matrix_free(tau);
        solution_free(iter_best);
        rank_shared_free(&shared);
        free(tau_sync_reqs);
        free(packed_solution);
        return;
      }

      iter_best_cost = global_best_cost;
      iter_best_ant = global_best_ant;
    }
#endif

    if (iter_failed) {
      if (!mpi_enabled || mpi_rank == 0) {
        fprintf(stderr, "allocation failure during iteration\n");
      }
      break;
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 32)
#endif
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

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 32)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n > 32)
#endif
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

#ifdef USE_MPI
    if (mpi_enabled) {
      if (!mpi_sync_tau_epoch(tau, n, mpi_size, tau_sync_reqs)) {
        if (mpi_rank == 0) {
          fprintf(stderr, "MPI pheromone synchronization failure\n");
        }
        matrix_free(tau);
        solution_free(iter_best);
        rank_shared_free(&shared);
        free(tau_sync_reqs);
        free(packed_solution);
        return;
      }
    }
#endif

    stop_now = 0;
    double iter_end_wall = wall_time_seconds();
    if (max_runtime_sec > 0.0 &&
        (iter_end_wall - start_wall) >= max_runtime_sec) {
      stop_now = 1;
    }
    if (!improved_global && max_stagnation_epochs > 0 &&
        no_improve_epochs >= max_stagnation_epochs) {
      stop_now = 1;
    }
#ifdef USE_MPI
    if (mpi_enabled) {
      int global_stop = 0;
      MPI_Allreduce(&stop_now, &global_stop, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      stop_now = global_stop;
    }
#endif
    if (stop_now) {
      break;
    }
  }

  matrix_free(tau);
  solution_free(iter_best);
  rank_shared_free(&shared);
#ifdef USE_MPI
  free(tau_sync_reqs);
  free(packed_solution);
#endif
}

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
