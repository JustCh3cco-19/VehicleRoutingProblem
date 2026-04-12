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

#define ACO_ALIGNMENT 64u
#define ACO_MAX_CANDIDATES 64

/* -- Static Helper Functions -- */

static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  return (rem == 0u) ? value : value + (alignment - rem);
}

static void *aligned_calloc_bytes(size_t bytes) {
  size_t alloc_bytes = align_up_size(bytes, ACO_ALIGNMENT);
  void *ptr = aligned_alloc(ACO_ALIGNMENT, alloc_bytes);
  if (!ptr) return NULL;
  memset(ptr, 0, alloc_bytes);
  return ptr;
}

static int clamp_int(int x, int lo, int hi) {
  return (x < lo) ? lo : (x > hi) ? hi : x;
}

static int choose_auto_total_ants(int n, int mpi_size) {
  int omp_threads = 1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif
  int workers = mpi_size * omp_threads;
  int ants_per_worker = (n <= 2000) ? 8 : (n > 16000) ? 4 : 6;
  int total_ants = workers * ants_per_worker;
  return clamp_int(total_ants, workers * 4, workers * 12);
}

static double fast_pow_nonneg(double base, double exponent) {
  if (exponent == 1.0) return base;
  if (exponent == 2.0) return base * base;
  if (exponent == 0.5) return sqrt(base);
  return pow(base, exponent);
}

static int aligned_row_stride(int cols, size_t elem_size) {
  size_t row_bytes = (size_t)cols * elem_size;
  size_t padded = align_up_size(row_bytes, ACO_ALIGNMENT);
  return (int)(padded / elem_size);
}

static int choose_candidate_count(int n) {
  if (n <= 8) return n;
  if (n <= 256) return 16;
  if (n <= 4096) return 24;
  return 32;
}

static double wall_time_seconds(void) {
#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) return MPI_Wtime();
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
  *max_stagnation_epochs = (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = (s_rel && *s_rel) ? atof(s_rel) : 1e-3;
}

static int is_significant_improvement(double prev_best, double new_best,
                                      double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) return (new_best < prev_best);
  if (new_best >= prev_best - 1e-9) return 0;
  return ((prev_best - new_best) / fmax(prev_best, 1e-9)) + 1e-9 >= min_rel_improvement;
}

/* -- Shared Matrix / Workspace Support -- */

static void rank_shared_free(AcoRankShared *shared) {
  if (!shared) return;
  free(shared->eta_beta);
  free(shared->candidate_idx);
  memset(shared, 0, sizeof(*shared));
}

static int rank_shared_init(AcoRankShared *shared, int n) {
  memset(shared, 0, sizeof(*shared));
  shared->n = n;
  shared->candidate_k = clamp_int(choose_candidate_count(n), 1, ACO_MAX_CANDIDATES);
  shared->stride = aligned_row_stride(shared->candidate_k, sizeof(float));
  shared->visited_words = (n / 64) + 1;
  size_t elems = (size_t)(n + 1) * (size_t)shared->stride;
  shared->candidate_idx = aligned_calloc_bytes(elems * sizeof(int));
  shared->eta_beta = aligned_calloc_bytes(elems * sizeof(float));
  if (!shared->candidate_idx || !shared->eta_beta) { rank_shared_free(shared); return 0; }
  for (size_t i = 0; i < elems; ++i) shared->candidate_idx[i] = -1;
  return 1;
}

static void rank_shared_build_candidates(AcoRankShared *shared, const Matrix *c_mat, double beta) {
  int n = shared->n, k = shared->candidate_k, stride = shared->stride;
#pragma omp parallel for schedule(static) if (n > 64)
  for (int i = 0; i <= n; ++i) {
    int nodes[ACO_MAX_CANDIDATES]; double dists[ACO_MAX_CANDIDATES];
    const double *row = c_mat->data + (size_t)i * (size_t)c_mat->stride;
    for (int t = 0; t < k; t++) { nodes[t] = -1; dists[t] = DBL_MAX; }
    for (int node = 1; node <= n; node++) {
      if (node == i) continue;
      double d = row[node]; int pos = -1;
      for (int t = 0; t < k; t++) { if (d < dists[t]) { pos = t; break; } }
      if (pos >= 0) {
        for (int s = k-1; s > pos; s--) { dists[s] = dists[s-1]; nodes[s] = nodes[s-1]; }
        dists[pos] = d; nodes[pos] = node;
      }
    }
    int *c_row = shared->candidate_idx + (size_t)i * (size_t)stride;
    float *e_row = shared->eta_beta + (size_t)i * (size_t)stride;
    for (int t = 0; t < k; t++) {
      c_row[t] = (nodes[t] > 0) ? nodes[t] : 0;
      e_row[t] = (nodes[t] > 0) ? (float)fast_pow_nonneg(1.0 / (dists[t] + 1e-9), beta) : 0.0f;
    }
  }
}

static void thread_workspace_free(AcoThreadWorkspace *ws) {
  if (!ws) return;
  free(ws->route_loads); free(ws->visited);
  solution_free(ws->thread_best); solution_free(ws->sol);
}

static int thread_workspace_init(AcoThreadWorkspace *ws, int K, int n, int visited_words) {
  ws->sol = solution_create(K, n); ws->thread_best = solution_create(K, n);
  ws->visited = aligned_calloc_bytes((size_t)visited_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->route_loads) { thread_workspace_free(ws); return 0; }
  return 1;
}

/* -- Core Algorithm Functions -- */

static inline int visited_is_set(const uint64_t *visited, int node) {
  return (int)((visited[(unsigned int)node >> 6] >> ((unsigned int)node & 63u)) & 1u);
}

static inline void visited_set(uint64_t *visited, int node) {
  visited[(unsigned int)node >> 6] |= (uint64_t)1u << ((unsigned int)node & 63u);
}

static int find_nearest_unvisited_accel(const AcoRankShared *shared, int current,
                                        const uint64_t *restrict visited,
                                        const Matrix *restrict c_mat) {
  int best = 0; double best_dist = DBL_MAX; int n = shared->n;
  const double *restrict dist_row = c_mat->data + (size_t)current * (size_t)c_mat->stride;
  int num_words = (n >> 6) + 1;
  for (int w = 0; w < num_words; ++w) {
    uint64_t v = visited[w]; if (v == 0xFFFFFFFFFFFFFFFFull) continue;
    uint64_t mask = ~v; int base = w << 6;
    if (w == num_words - 1) {
        int bits = (n % 64) + 1;
        if (bits < 64) mask &= (1ull << bits) - 1;
    }
    if (w == 0) mask &= ~1ull;
    while (mask != 0) {
      int bit = __builtin_ctzll(mask); int node = base + bit;
      double d = dist_row[node];
      if (d < best_dist) { best_dist = d; best = node; }
      mask &= mask - 1;
    }
  }
  return best;
}

static int select_next_customer(const AcoRankShared *shared, int current,
                                const uint64_t *restrict visited,
                                const Matrix *restrict c_mat,
                                const float *restrict score_matrix,
                                unsigned int *restrict rng_state) {
  float masked_weights[ACO_MAX_CANDIDATES];
  int stride = shared->stride, k = shared->candidate_k;
  const int *cands = shared->candidate_idx + (size_t)current * (size_t)stride;
  const float *scores = score_matrix + (size_t)current * (size_t)stride;
  double denom = 0.0;
  for (int t = 0; t < k; ++t) {
    int node = cands[t];
    if (node > 0 && !visited_is_set(visited, node)) {
      masked_weights[t] = scores[t]; denom += (double)scores[t];
    } else masked_weights[t] = 0.0f;
  }
  if (denom > 0.0) {
    double threshold = aco_rand01_state(rng_state) * denom, cumulative = 0.0;
    for (int i = 0; i < k; ++i) {
      if (masked_weights[i] <= 0.0f) continue;
      cumulative += (double)masked_weights[i];
      if (cumulative >= threshold) return cands[i];
    }
    for (int i = k - 1; i >= 0; --i) if (masked_weights[i] > 0.0f) return cands[i];
  }
  return find_nearest_unvisited_accel(shared, current, visited, c_mat);
}

static void build_ant_solution(AcoThreadWorkspace *ws, const AcoRankShared *shared, int K,
                               int vehicle_capacity_customers, const Matrix *restrict c_mat,
                               const float *restrict score_matrix) {
  solution_reset(ws->sol);
  memset(ws->visited, 0, (size_t)shared->visited_words * sizeof(uint64_t));
  memset(ws->route_loads, 0, (size_t)K * sizeof(int));
  int route_cap = (vehicle_capacity_customers <= 0) ? shared->n : vehicle_capacity_customers;
  int remaining = shared->n;
  for (int vehicle = 0; vehicle < K; ++vehicle) {
    Route *r = &ws->sol->routes[vehicle]; route_append(r, 0);
    int current = 0, remaining_v = K - vehicle - 1, fut_cap = remaining_v * route_cap;
    while (remaining > 0 && remaining > fut_cap && ws->route_loads[vehicle] < route_cap) {
      int next = select_next_customer(shared, current, ws->visited, c_mat,
                                      score_matrix, &ws->rng_state);
      if (next <= 0 || visited_is_set(ws->visited, next)) {
        next = find_nearest_unvisited_accel(shared, current, ws->visited, c_mat);
      }
      if (next <= 0 || visited_is_set(ws->visited, next)) break;
      route_append(r, next); visited_set(ws->visited, next);
      ++ws->route_loads[vehicle]; --remaining; current = next;
    }
    route_append(r, 0);
  }
  if (remaining > 0) {
    for (int v = K - 1; v >= 0 && remaining > 0; --v) {
      Route *r = &ws->sol->routes[v]; if (r->len > 0 && r->nodes[r->len - 1] == 0) --r->len;
      while (remaining > 0 && ws->route_loads[v] < route_cap) {
        int cur = (r->len > 0) ? r->nodes[r->len - 1] : 0;
        int next = find_nearest_unvisited_accel(shared, cur, ws->visited, c_mat);
        if (next <= 0) break;
        route_append(r, next); visited_set(ws->visited, next);
        ++ws->route_loads[v]; --remaining;
      }
      route_append(r, 0);
    }
  }
}

#ifdef USE_MPI
static int mpi_sync_tau_epoch(Matrix *tau_mat, int mpi_size) {
  int n = tau_mat->n;
  int side = n + 1;
  size_t total_elems = (size_t)side * (size_t)tau_mat->stride;
  
  if (MPI_Allreduce(MPI_IN_PLACE, tau_mat->data, (int)total_elems, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) != MPI_SUCCESS) {
      return 0;
  }
  
  double inv = 1.0 / (double)mpi_size;
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < side; ++i) {
    for (int j = 0; j < side; ++j) {
      if (i == j) tau_mat->rows[i][j] = 0.0;
      else tau_mat->rows[i][j] *= inv;
    }
  }
  return 1;
}
#endif

/* -- Main Parallel Routine -- */

static void aco_vrp_run_with_timer(int n, int K, int vehicle_capacity_customers,
                                   int m, double **c, double alpha,
                                   double beta, double rho, double tau0,
                                   double Q, unsigned int seed,
                                   Solution *best_solution, double *best_cost,
                                   double max_runtime_sec,
                                   int max_stagnation_epochs,
                                   double min_rel_improvement) {
  int mpi_rank = 0, mpi_size = 1; bool mpi_enabled = false;
#ifdef USE_MPI
  int mpi_init = 0; MPI_Initialized(&mpi_init);
  if (mpi_init) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_enabled = (mpi_size > 1);
  }
#endif
  int total_m = (m <= 0) ? choose_auto_total_ants(n, mpi_size) : m;
  int local_m = total_m / mpi_size + ((mpi_rank < (total_m % mpi_size)) ? 1 : 0);
  int ant_offset = mpi_rank * (total_m / mpi_size) + ((mpi_rank < (total_m % mpi_size)) ? mpi_rank : (total_m % mpi_size));

  Matrix *tau_mat = matrix_create(n); Matrix *c_mat = matrix_create(n);
  AcoRankShared shared = {0};
  int shared_ok = rank_shared_init(&shared, n);
  float *score_matrix =
      shared_ok ? aligned_calloc_bytes((size_t)(n + 1) *
                                       (size_t)shared.stride * sizeof(float))
                : NULL;
  if (c_mat) { for (int i = 0; i <= n; i++) memcpy(c_mat->rows[i], c[i], (size_t)(n + 1) * sizeof(double)); }
  Solution *iter_best = solution_create(K, n);
  double iter_best_cost = DBL_MAX; int improved_global = 0;

  if (!tau_mat || !c_mat || !shared_ok || !score_matrix || !iter_best) {
    if (mpi_rank == 0) fprintf(stderr, "allocation failure in openmp-mpi solver\n");
    matrix_free_handle(tau_mat);
    matrix_free_handle(c_mat);
    free(score_matrix);
    solution_free(iter_best);
    rank_shared_free(&shared);
    return;
  }
  for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) tau_mat->rows[i][j] = (i == j) ? 0.0 : tau0;
  rank_shared_build_candidates(&shared, c_mat, beta);
  *best_cost = DBL_MAX; double start_wall = wall_time_seconds();
  int no_improve_epochs = 0, stagnation_iters = 0, stop_now = 0, iter_failed = 0;
  double tau_max = tau0, tau_min = tau0 * 0.05;
  int total_iters = 0;

  #ifdef _OPENMP
  #pragma omp parallel default(shared) proc_bind(close)
  {
    AcoThreadWorkspace ws;
    bool ws_ok = thread_workspace_init(&ws, K, n, shared.visited_words);
    if (!ws_ok) {
      #pragma omp critical
      { iter_failed = 1; }
    }
    int actual_threads = omp_get_num_threads();
    #pragma omp master
    { if (mpi_rank == 0) printf("Hybrid V2 solver: %d MPI ranks, %d OpenMP threads/rank\n", mpi_size, actual_threads); }

    for (int iter = 0;; ++iter) {
      #pragma omp master
      {
        if (max_runtime_sec > 0.0 && (wall_time_seconds() - start_wall) >= max_runtime_sec) stop_now = 1;
      }
      #pragma omp barrier
      if (stop_now || iter_failed) break;
      #pragma omp master
      { total_iters++; }
      #pragma omp barrier

      // 1. Score Update (Parallel) - Optimized for Cache L3
      #pragma omp for schedule(static)
      for (int i = 0; i <= n; i++) {
        int stride = shared.stride, k = shared.candidate_k;
        int *cands = shared.candidate_idx + (size_t)i * (size_t)stride;
        float *etas = shared.eta_beta + (size_t)i * (size_t)stride;
        float *scores = score_matrix + (size_t)i * (size_t)stride;
        const double *tau_row = tau_mat->rows[i];
        for (int t = 0; t < k; t++) {
          int node = cands[t];
          scores[t] = (node > 0) ? (float)(fast_pow_nonneg(tau_row[node], alpha) * (double)etas[t]) : 0.0f;
        }
      }
      #pragma omp barrier

      #pragma omp single
      { iter_best_cost = DBL_MAX; improved_global = 0; }

      // 2. Ant Construction (Strong Scaling Focus)
      double t_thread_best_c = DBL_MAX;
      if (ws_ok) {
        #pragma omp for schedule(dynamic, 1) nowait
        for (int ant = 0; ant < local_m; ant++) {
          ws.rng_state = aco_make_ant_seed(seed, iter, ant_offset + ant);
          build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c_mat, score_matrix);
          double cost = solution_cost(ws.sol, c_mat->rows);
          if (cost < t_thread_best_c) {
            t_thread_best_c = cost;
            solution_copy(ws.thread_best, ws.sol);
          }
        }
        
        #pragma omp critical
        {
          if (t_thread_best_c < iter_best_cost) {
            iter_best_cost = t_thread_best_c;
            solution_copy(iter_best, ws.thread_best);
          }
        }
      }
      #pragma omp barrier

      // 3. Global Sync & Stagnation Logic
  #ifdef USE_MPI
      #pragma omp master
      if (mpi_enabled) {
        double g_min = iter_best_cost;
        MPI_Allreduce(MPI_IN_PLACE, &g_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        iter_best_cost = g_min;
      }
      #pragma omp barrier
  #endif

      #pragma omp single
      {
        if (is_significant_improvement(*best_cost, iter_best_cost, min_rel_improvement)) {
          *best_cost = iter_best_cost;
          solution_copy(best_solution, iter_best);
          stagnation_iters = 0; no_improve_epochs = 0; improved_global = 1;
          tau_max = 1.0 / ((1.0 - rho) * (*best_cost));
          tau_min = tau_max * 0.05;
        } else {
          stagnation_iters++; no_improve_epochs++;
        }
      }

      // 4. Pheromone Update (Parallelized Deposit for Weak Scaling)
      #pragma omp for collapse(2) schedule(static)
      for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
          if (i != j) tau_mat->rows[i][j] *= (1.0 - rho);
        }
      }

      // Parallel deposit of Iter Best
      double dep = (0.3 * Q) / fmax(iter_best_cost, 1e-9);
      #pragma omp for schedule(static)
      for (int v = 0; v < K; v++) {
        Route *r = &iter_best->routes[v];
        for (int t = 0; t + 1 < r->len; t++) {
          int from = r->nodes[t], to = r->nodes[t+1];
          #pragma omp atomic
          tau_mat->rows[from][to] += dep;
          #pragma omp atomic
          tau_mat->rows[to][from] += dep;
        }
      }

      // Parallel deposit of Global Best
      if (*best_cost < DBL_MAX) {
        double g_dep = (0.7 * Q) / (*best_cost);
        #pragma omp for schedule(static)
        for (int v = 0; v < K; v++) {
          Route *r = &best_solution->routes[v];
          for (int t = 0; t + 1 < r->len; t++) {
            int from = r->nodes[t], to = r->nodes[t+1];
            #pragma omp atomic
            tau_mat->rows[from][to] += g_dep;
            #pragma omp atomic
            tau_mat->rows[to][from] += g_dep;
          }
        }
      }

      // 5. Limits & Stagnation Reset (Parallel)
      #pragma omp for collapse(2) schedule(static)
      for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
          if (i == j) continue;
          if (tau_mat->rows[i][j] < tau_min) tau_mat->rows[i][j] = tau_min;
          else if (tau_mat->rows[i][j] > tau_max) tau_mat->rows[i][j] = tau_max;
        }
      }

      if (stagnation_iters >= 32) {
        #pragma omp for schedule(static)
        for (int i = 0; i <= n; i++) {
          for (int j = 0; j <= n; j++) {
            if (i != j) tau_mat->rows[i][j] = 0.5 * tau_mat->rows[i][j] + 0.5 * tau0;
          }
        }
        #pragma omp master
        { stagnation_iters = 0; }
      }

  #ifdef USE_MPI
      #pragma omp master
      if (mpi_enabled) { mpi_sync_tau_epoch(tau_mat, mpi_size); }
      #pragma omp barrier
  #endif
      #pragma omp master
      { 
        if (!improved_global && max_stagnation_epochs > 0 && no_improve_epochs >= max_stagnation_epochs) stop_now = 1;
      }
      #pragma omp barrier
      if (stop_now) break;
    }
    if (ws_ok) thread_workspace_free(&ws);
  }
  #else
  // Serial fallback simplified
  AcoThreadWorkspace ws;
  if (thread_workspace_init(&ws, K, n, shared.visited_words)) {
    for (int iter = 0;; iter++) {
      if (max_runtime_sec > 0.0 && (wall_time_seconds() - start_wall) >= max_runtime_sec) break;
      total_iters++;
      for (int i = 0; i <= n; i++) {
        int stride = shared.stride, k = shared.candidate_k;
        int *cands = shared.candidate_idx + (size_t)i * (size_t)stride;
        float *etas = shared.eta_beta + (size_t)i * (size_t)stride;
        float *scores = score_matrix + (size_t)i * (size_t)stride;
        const double *tau_row = tau_mat->rows[i];
        for (int t = 0; t < k; t++) {
          int node = cands[t];
          scores[t] = (node > 0) ? (float)(fast_pow_nonneg(tau_row[node], alpha) * (double)etas[t]) : 0.0f;
        }
      }
      double t_best_c = DBL_MAX;
      for (int ant = 0; ant < local_m; ant++) {
        ws.rng_state = aco_make_ant_seed(seed, iter, ant_offset + ant);
        build_ant_solution(&ws, &shared, K, vehicle_capacity_customers, c_mat, score_matrix);
        double cost = solution_cost(ws.sol, c_mat->rows);
        if (cost < t_best_c) {
          t_best_c = cost;
          solution_copy(ws.thread_best, ws.sol);
        }
      }
      if (is_significant_improvement(*best_cost, t_best_c, min_rel_improvement)) {
        *best_cost = t_best_c;
        solution_copy(best_solution, ws.thread_best);
        stagnation_iters = 0; no_improve_epochs = 0;
      } else { stagnation_iters++; no_improve_epochs++; }
      if (max_stagnation_epochs > 0 && no_improve_epochs >= max_stagnation_epochs) break;
    }
    thread_workspace_free(&ws);
  }
#endif

  if (mpi_rank == 0) {
    printf("Executed Epochs: %d\n", total_iters);
    printf("Total Wall Time: %.3f s\n", wall_time_seconds() - start_wall);
  }

  matrix_free_handle(tau_mat); 
  matrix_free_handle(c_mat); 
  free(score_matrix);
  solution_free(iter_best); 
  rank_shared_free(&shared);
}

void aco_vrp(int n, int K, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
  int cap = (K > 0) ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K)) : n;
  aco_vrp_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}

void aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
  double max_runtime_sec = 0.0; int max_stagnation_epochs = 0; double min_rel_improvement = 1e-3;
  load_timer_directives(&max_runtime_sec, &max_stagnation_epochs, &min_rel_improvement);
  aco_vrp_run_with_timer(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost, max_runtime_sec, max_stagnation_epochs, min_rel_improvement);
}
