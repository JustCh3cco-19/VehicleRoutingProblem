#include "aco.h"
#include "aco_config.h"
#include "aco_mpi_internal.h"
#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#define ACO_MPI_EPS 1e-7f

static double wall_time(void) {
#ifdef USE_MPI
  int init = 0;
  MPI_Initialized(&init);
  if (init) {
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

static float fast_powf(float base, float exp) {
  if (exp == 1.0f) {
    return base;
  }
  if (exp == 2.0f) {
    return base * base;
  }
  if (exp == 0.5f) {
    return sqrtf(base);
  }
  return powf(base, exp);
}

typedef struct {
  Solution *sol;
  uint64_t *visited;
  uint64_t *meta_active;
  int *route_loads;
  unsigned int rng_state;
  Solution *thread_best;
} aco_mpi_workspace_t;

static void aco_mpi_ws_free(aco_mpi_workspace_t *ws) {
  if (!ws) {
    return;
  }
  free(ws->route_loads);
  free(ws->visited);
  free(ws->meta_active);
  solution_free(ws->thread_best);
  solution_free(ws->sol);
}


static double aco_mpi_rand01(unsigned int *state) {
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


static bool aco_mpi_route_append(Route *r, int node) {
  if (r->len >= r->cap) {
    return false;
  }
  r->nodes[r->len] = node;
  r->len++;
  return true;
}

static int aco_mpi_ws_init(aco_mpi_workspace_t *ws, int K, int n, int words,
                           int meta_words) {
  size_t visited_bytes = 0;
  size_t meta_bytes = 0;
  size_t route_load_bytes = 0;
  if (!matrix_mul_size((size_t)words, sizeof(uint64_t), &visited_bytes) ||
      !matrix_mul_size((size_t)meta_words, sizeof(uint64_t), &meta_bytes) ||
      !matrix_mul_size((size_t)K, sizeof(int), &route_load_bytes)) {
    return 0;
  }

  ws->sol = solution_create(K, n);
  ws->thread_best = solution_create(K, n);
  ws->visited = aco_mpi_aligned_calloc_64(visited_bytes);
  ws->meta_active = aco_mpi_aligned_calloc_64(meta_bytes);
  ws->route_loads = calloc(1u, route_load_bytes);
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->meta_active ||
      !ws->route_loads) {
    aco_mpi_ws_free(ws);
    return 0;
  }
  return 1;
}

static int find_nearest_unvisited_h(const aco_mpi_rank_shared_t *s, int curr,
                                    const aco_mpi_workspace_t *ws,
                                    const aco_mpi_matrix_float_t *c) {
  int best = 0;
  float best_d = FLT_MAX;
  const float *row = c->rows[curr];

  for (int mw = 0; mw < s->meta_words; mw++) {
    uint64_t m_mask = ws->meta_active[mw];
    while (m_mask != 0) {
      int w_off = __builtin_ctzll(m_mask);
      int w = (mw << 6) + w_off;
      if (w >= s->visited_words) {
        break;
      }

      uint64_t visited = ws->visited[w];
      uint64_t mask = ~visited;
      int base = w << 6;
      if (w == s->visited_words - 1) {
        int bits = (s->n % 64) + 1;
        if (bits < 64) {
          mask &= (1ull << bits) - 1;
        }
      }
      if (w == 0) {
        mask &= ~1ull;
      }

      while (mask != 0) {
        int bit = __builtin_ctzll(mask);
        int node = base + bit;
        float d = row[node];
        if (d < best_d) {
          best_d = d;
          best = node;
        }
        mask &= mask - 1;
      }
      m_mask &= m_mask - 1;
    }
  }
  return best;
}

static bool build_ant_h(aco_mpi_workspace_t *ws,
                        const aco_mpi_rank_shared_t *s, int K, int cap,
                        const aco_mpi_matrix_float_t *c,
                        const float *scores) {
  solution_reset(ws->sol);
  memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
  memset(ws->meta_active, 0xFF, (size_t)s->meta_words * sizeof(uint64_t));
  memset(ws->route_loads, 0, (size_t)K * sizeof(int));

  int rem = s->n;
  for (int v = 0; v < K; v++) {
    if (!aco_mpi_route_append(&ws->sol->routes[v], 0)) {
      return false;
    }
    int curr = 0;
    while (true) {
      int rem_v = K - v - 1;
      int fut_cap = rem_v * cap;
      if (!(rem > 0 && rem > fut_cap && ws->route_loads[v] < cap)) {
        break;
      }

      enum { kMaxMpiLocalCandidates = 1024 };
      int next = 0;
      if (s->cand_k <= kMaxMpiLocalCandidates) {
        int active_nodes[kMaxMpiLocalCandidates];
        float active_scores[kMaxMpiLocalCandidates];
        int active_count = 0;
        float denom = 0.0f;

        const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
        const float *sc = scores + (size_t)curr * (size_t)s->stride;

        for (int t = 0; t < s->cand_k; t++) {
          int node = cands[t];
          if (node <= 0) {
            continue;
          }
          int word = (unsigned)node >> 6;
          int bit = (unsigned)node & 63u;
          if (!((ws->visited[word] >> bit) & 1u)) {
            float w = sc[t];
            if (w > 0.0f) {
              denom += w;
              active_nodes[active_count] = node;
              active_scores[active_count] = w;
              active_count++;
            }
          }
        }

        if (denom > 0.0f) {
          float thres = (float)aco_mpi_rand01(&ws->rng_state) * denom;
          float cum = 0.0f;
          for (int i = 0; i < active_count; i++) {
            float w = active_scores[i];
            cum += w;
            if (cum >= thres) {
              next = active_nodes[i];
              break;
            }
          }
          if (next <= 0 && active_count > 0) {
            next = active_nodes[active_count - 1];
          }
        }
      } else {
        float denom = 0.0f;
        const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
        const float *sc = scores + (size_t)curr * (size_t)s->stride;
        for (int t = 0; t < s->cand_k; t++) {
          int node = cands[t];
          int word = (unsigned)node >> 6;
          int bit = (unsigned)node & 63u;
          if (node > 0 && !((ws->visited[word] >> bit) & 1u)) {
            denom += sc[t];
          }
        }

        if (denom > 0.0f) {
          float thres = (float)aco_mpi_rand01(&ws->rng_state) * denom;
          float cum = 0.0f;
          for (int t = 0; t < s->cand_k; t++) {
            int node = cands[t];
            int word = (unsigned)node >> 6;
            int bit = (unsigned)node & 63u;
            if (node > 0 && !((ws->visited[word] >> bit) & 1u)) {
              cum += sc[t];
              if (cum >= thres) {
                next = node;
                break;
              }
            }
          }
        }
      }

      if (next <= 0) {
        next = find_nearest_unvisited_h(s, curr, ws, c);
      }
      if (next <= 0) {
        break;
      }

      if (!aco_mpi_route_append(&ws->sol->routes[v], next)) {
        return false;
      }
      int word_idx = (unsigned)next >> 6;
      ws->visited[word_idx] |= (1ull << ((unsigned)next & 63u));
      if (ws->visited[word_idx] == 0xFFFFFFFFFFFFFFFFull) {
        ws->meta_active[word_idx >> 6] &= ~(1ull << (word_idx & 63u));
      }
      ws->route_loads[v]++;
      rem--;
      curr = next;
    }
    if (!aco_mpi_route_append(&ws->sol->routes[v], 0)) {
      return false;
    }
  }

  return true;
}

static double solution_cost_f(const Solution *s, float **c) {
  double total = 0.0;
  for (int i = 0; i < s->K; i++) {
    const Route *r = &s->routes[i];
    if (r->len <= 2) {
      continue;
    }
    for (int t = 0; t + 1 < r->len; t++) {
      total += (double)c[r->nodes[t]][r->nodes[t + 1]];
    }
  }
  return total;
}

static int is_sig_imp(double prev_best, double new_best,
                      double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) {
    return (new_best < prev_best);
  }
  if (new_best >= prev_best - ACO_MPI_EPS) {
    return 0;
  }
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, (double)ACO_MPI_EPS);
  return rel_gain + (double)ACO_MPI_EPS >= min_rel_improvement;
}

/**
 * @brief Core MPI/OpenMP ACO execution loop.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param cap Per-vehicle customer capacity.
 * @param m Number of ants (0 enables backend auto-tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_sol Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp_run(int n, int K, int cap, int m, double **c, double alpha,
                      double beta, double rho, double tau0, double Q,
                      unsigned int seed, Solution *best_sol,
                      double *best_cost,
                      const AcoRuntimeConfig *config) {
  if (n <= 0 || K <= 0 || !c || !best_sol || !best_cost) {
    return ACO_ERR_INVALID_INPUT;
  }

  int mpi_rank = 0;
  int mpi_size = 1;
#ifdef USE_MPI
  int mpi_init = 0;
  MPI_Initialized(&mpi_init);
  if (mpi_init) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  }
#endif

  int requested_candidate_k = config ? config->candidate_k : 0;
  int cand_k = aco_mpi_choose_candidate_count(n, requested_candidate_k);
  int default_ants = (n / 2) * mpi_size;
  int total_m = (m <= 0) ? default_ants : m;
  if (m <= 0 && config && config->ants > 0) {
    total_m = config->ants;
  }

  int rank_extra = (mpi_rank < (total_m % mpi_size)) ? mpi_rank
                                                     : (total_m % mpi_size);
  int ant_off = mpi_rank * (total_m / mpi_size) + rank_extra;
  int local_m = total_m / mpi_size + (mpi_rank < (total_m % mpi_size));
  double max_runtime_sec = config ? config->timeout_seconds : 0.0;
  int max_stagnation_epochs = config ? config->stagnation_epochs : 0;
  double min_rel_improvement = config ? config->min_rel_improvement : 1e-3;
  double progress_interval_sec =
      config ? config->progress_interval_seconds : 10.0;
  int log_level = config ? config->log_level : ACO_LOG_PROGRESS;
  int fixed_epochs = config ? config->fixed_epochs : 0;
  if (max_stagnation_epochs <= 0) {
    max_stagnation_epochs = 100;
  }

  aco_mpi_matrix_float_t *tau_mat = aco_mpi_matrix_create_float(n);
  aco_mpi_matrix_float_t *c_mat = aco_mpi_matrix_create_float(n);
  if (!tau_mat || !c_mat) {
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    return ACO_ERR_ALLOCATION;
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i <= n; i++) {
    for (int j = 0; j <= n; j++) {
      c_mat->rows[i][j] = (float)c[i][j];
      tau_mat->rows[i][j] = (i == j) ? 0.0f : (float)tau0;
    }
  }

  aco_mpi_rank_shared_t shared = {
      .n = 0,
      .cand_k = 0,
      .stride = 0,
      .visited_words = 0,
      .meta_words = 0,
      .cand_idx = NULL,
      .eta_beta = NULL,
  };
  if (!aco_mpi_shared_init(&shared, n, cand_k, c_mat, beta)) {
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    return ACO_ERR_ALLOCATION;
  }

  size_t score_count = 0;
  size_t score_bytes = 0;
  if (!matrix_mul_size((size_t)(n + 1), (size_t)shared.stride,
                       &score_count) ||
      !matrix_mul_size(score_count, sizeof(float), &score_bytes)) {
    aco_mpi_shared_free(&shared);
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    return ACO_ERR_ALLOCATION;
  }
  float *score_mat = matrix_aligned_calloc(score_bytes, kAcoMpiAlignment);
  Solution *iter_best_sol_rank = solution_create(K, n);
  double iter_best_cost_g = DBL_MAX;
  *best_cost = DBL_MAX;
  double start_time = wall_time();
  double next_progress_time =
      (progress_interval_sec > 0.0) ? start_time + progress_interval_sec : 0.0;
  if (!score_mat || !iter_best_sol_rank) {
    free(score_mat);
    solution_free(iter_best_sol_rank);
    aco_mpi_shared_free(&shared);
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    return ACO_ERR_ALLOCATION;
  }

#ifdef USE_MPI
  aco_mpi_async_sparse_context_t async_ctx;
  aco_mpi_async_sparse_init(&async_ctx, mpi_size);
#endif

  int iter_since_best = 0;
  size_t rank_delta_capacity = 0;
  size_t rank_delta_bytes = 0;
  if (!matrix_mul_size((size_t)n + (size_t)K + 500u, 2u,
                       &rank_delta_capacity) ||
      !matrix_mul_size(rank_delta_capacity, sizeof(aco_mpi_sparse_delta_t),
                       &rank_delta_bytes)) {
    solution_free(iter_best_sol_rank);
    aco_mpi_shared_free(&shared);
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    free(score_mat);
    return ACO_ERR_ALLOCATION;
  }
  aco_mpi_sparse_delta_t *rank_deltas = malloc(rank_delta_bytes);
  int rank_delta_count = 0;
  if (!rank_deltas) {
    solution_free(iter_best_sol_rank);
    aco_mpi_shared_free(&shared);
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    free(score_mat);
    return ACO_ERR_ALLOCATION;
  }

  int workspace_failed = 0;
#pragma omp parallel default(shared) proc_bind(close)
  {
    aco_mpi_workspace_t ws = {
        .sol = NULL,
        .visited = NULL,
        .meta_active = NULL,
        .route_loads = NULL,
        .rng_state = 0u,
        .thread_best = NULL,
    };
    if (!aco_mpi_ws_init(&ws, K, n, shared.visited_words,
                         shared.meta_words)) {
#pragma omp atomic write
      workspace_failed = 1;
    }
#pragma omp barrier

    if (log_level > ACO_LOG_SILENT && mpi_rank == 0 &&
        omp_get_thread_num() == 0 && !workspace_failed) {
      fprintf(stderr,
              "ACO Parallel Starting with %d threads. N=%d K=%d Cap=%d "
              "candidate_k=%d seed=%u\n",
              omp_get_num_threads(), n, K, cap, cand_k, seed);
    }

    if (!workspace_failed) {
      for (int iter = 0;
           (fixed_epochs > 0 && iter < fixed_epochs) ||
           (fixed_epochs <= 0 && iter_since_best < max_stagnation_epochs);
           iter++) {
        if (max_runtime_sec > 0.0 &&
            (wall_time() - start_time) > max_runtime_sec) {
          break;
        }
#ifdef USE_MPI
#pragma omp master
        aco_mpi_async_sparse_wait_and_apply(&async_ctx, tau_mat, mpi_rank,
                                            mpi_size);
#pragma omp barrier
#endif

#pragma omp for schedule(static) nowait
        for (int i = 0; i <= n; i++) {
          int *cands = shared.cand_idx + (size_t)i * (size_t)shared.stride;
          float *etas = shared.eta_beta + (size_t)i * (size_t)shared.stride;
          float *sc = score_mat + (size_t)i * (size_t)shared.stride;
          const float *tau_row = tau_mat->rows[i];
          for (int t = 0; t < shared.cand_k; t++) {
            int node = cands[t];
            sc[t] = (node > 0)
                        ? (fast_powf(tau_row[node], (float)alpha) * etas[t])
                        : 0.0f;
          }
        }

#pragma omp barrier
        double t_best_c = DBL_MAX;
#pragma omp for schedule(runtime) nowait
        for (int a = 0; a < local_m; a++) {
          ws.rng_state = aco_make_ant_seed(seed, iter, ant_off + a);
          if (!build_ant_h(&ws, &shared, K, cap, c_mat, score_mat)) {
            continue;
          }
          double cost = solution_cost_f(ws.sol, c_mat->rows);
          if (cost < t_best_c) {
            t_best_c = cost;
            solution_copy(ws.thread_best, ws.sol);
          }
        }

#pragma omp critical
        {
          if (t_best_c < iter_best_cost_g) {
            iter_best_cost_g = t_best_c;
            solution_copy(iter_best_sol_rank, ws.thread_best);
          }
        }

#pragma omp barrier
#pragma omp master
        {
          double g_min = iter_best_cost_g;
#ifdef USE_MPI
          if (mpi_size > 1) {
            MPI_Allreduce(MPI_IN_PLACE, &g_min, 1, MPI_DOUBLE, MPI_MIN,
                          MPI_COMM_WORLD);
          }
#endif
          if (is_sig_imp(*best_cost, g_min, min_rel_improvement)) {
            iter_since_best = 0;
          } else {
            iter_since_best++;
          }
          if (g_min < *best_cost) {
            *best_cost = g_min;
            solution_copy(best_sol, iter_best_sol_rank);
          }

          double now = wall_time();
          if (log_level > ACO_LOG_SILENT && mpi_rank == 0 &&
              progress_interval_sec > 0.0 && now >= next_progress_time) {
            double elapsed = now - start_time;
            if (max_runtime_sec > 0.0) {
              double remaining = max_runtime_sec - elapsed;
              if (remaining < 0.0) {
                remaining = 0.0;
              }
              fprintf(stderr,
                      "[mpi] elapsed %.1fs, remaining %.1fs, iter %d, "
                      "best %.3f\n",
                      elapsed, remaining, iter + 1, *best_cost);
            } else {
              fprintf(stderr, "[mpi] elapsed %.1fs, iter %d, best %.3f\n",
                      elapsed, iter + 1, *best_cost);
            }
            next_progress_time = now + progress_interval_sec;
          }
          iter_best_cost_g = DBL_MAX;
          rank_delta_count = 0;
        }

#pragma omp barrier
        float rho_f = (float)rho;
#pragma omp for schedule(static)
        for (size_t i = 0; i < (size_t)(n + 1) * tau_mat->stride; i++) {
          tau_mat->data[i] *= (1.0f - rho_f);
        }

        float dep = (float)(Q / fmax(*best_cost, 1e-9));
        float weighted_dep = dep / (float)mpi_size;
#pragma omp for schedule(static)
        for (int v = 0; v < K; v++) {
          Route *r = &best_sol->routes[v];
          if (r->len <= 2) {
            continue;
          }
          for (int t = 0; t + 1 < r->len; t++) {
            int from = r->nodes[t];
            int to = r->nodes[t + 1];
            uint32_t idx1 = (uint32_t)((size_t)from * tau_mat->stride + to);
            uint32_t idx2 = (uint32_t)((size_t)to * tau_mat->stride + from);
#pragma omp atomic
            tau_mat->data[idx1] += weighted_dep;
#pragma omp atomic
            tau_mat->data[idx2] += weighted_dep;

            int d_idx;
#pragma omp atomic capture
            d_idx = rank_delta_count++;
            rank_deltas[d_idx] = (aco_mpi_sparse_delta_t){
                .edge_idx = idx1,
                .increment = weighted_dep,
            };
#pragma omp atomic capture
            d_idx = rank_delta_count++;
            rank_deltas[d_idx] = (aco_mpi_sparse_delta_t){
                .edge_idx = idx2,
                .increment = weighted_dep,
            };
          }
        }

#pragma omp barrier
#pragma omp master
        {
#ifdef USE_MPI
          if (mpi_size > 1) {
            aco_mpi_async_sparse_start(&async_ctx, rank_deltas,
                                       rank_delta_count, mpi_size);
          }
#endif
        }
#pragma omp barrier
      }
    }
#ifdef USE_MPI
#pragma omp master
    aco_mpi_async_sparse_cleanup(&async_ctx);
#endif
    aco_mpi_ws_free(&ws);
  }

  free(rank_deltas);
  if (workspace_failed) {
    aco_mpi_matrix_free_float(tau_mat);
    aco_mpi_matrix_free_float(c_mat);
    free(score_mat);
    solution_free(iter_best_sol_rank);
    aco_mpi_shared_free(&shared);
    return ACO_ERR_ALLOCATION;
  }

  if (log_level > ACO_LOG_SILENT && mpi_rank == 0) {
    fprintf(stderr,
            "ACO Parallel Ultimate Completion. Best: %.3f. Time: %.3fs\n",
            *best_cost, wall_time() - start_time);
  }
  aco_mpi_matrix_free_float(tau_mat);
  aco_mpi_matrix_free_float(c_mat);
  free(score_mat);
  solution_free(iter_best_sol_rank);
  aco_mpi_shared_free(&shared);
  return (*best_cost < DBL_MAX) ? ACO_OK : ACO_ERR_NO_SOLUTION;
}

/**
 * @brief Runs the MPI/OpenMP ACO solver with explicit vehicle capacity.
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
  return aco_vrp_run(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho,
                     tau0, Q, seed, best_solution, best_cost, &config);
}

/**
 * @brief Runs the MPI/OpenMP ACO solver with auto-derived vehicle capacity.
 */
AcoStatus aco_vrp(int n, int K, int m, double **c, double alpha, double beta,
                  double rho, double tau0, double Q, unsigned int seed,
                  Solution *best_solution, double *best_cost) {
  int cap = (K > 0)
                ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K))
                : n;
  return aco_vrp_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q,
                               seed, best_solution, best_cost);
}
