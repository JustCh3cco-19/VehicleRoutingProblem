#include "aco.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#define EPS 1e-9

static double rand01(void) {
  return (double)rand() / (double)RAND_MAX;
}

static double rand01_state(unsigned int *state) {
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

static unsigned int make_ant_seed(unsigned int base_seed, int iter,
                                  int ant_index) {
  unsigned int x = base_seed ? base_seed : 1u;
  x ^= 0xa511e9b3u;
  x ^= (unsigned int)(iter + 1) * 0x9e3779b1u;
  x ^= (unsigned int)(ant_index + 1) * 0xc2b2ae3du;
  return x ? x : 1u;
}

static int select_next(int current, const bool *visited, int n,
                       double **tau, double **eta, double alpha, double beta,
                       unsigned int *rng_state) {
  double denom = 0.0;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      denom += score;
    }
  }
  if (denom <= 0.0) {
    for (int j = 1; j <= n; ++j) {
      if (!visited[j]) return j;
    }
    return 0;
  }

  double r = rand01_state(rng_state);
  double cumulative = 0.0;
  int last = 1;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      double p = score / denom;
      cumulative += p;
      last = j;
      if (cumulative >= r) {
        return j;
      }
    }
  }
  return last;
}

static void build_ant_solution(Solution *sol, bool *visited, int n, int K,
                               double **tau, double **eta, double alpha,
                               double beta, unsigned int *rng_state) {
  memset(visited, 0, (size_t)(n + 1) * sizeof(bool));
  solution_reset(sol);

  int unvisited_count = n;

  for (int vehicle = 1; vehicle <= K; ++vehicle) {
    Route *r = &sol->routes[vehicle - 1];
    route_append(r, 0);
    int current = 0;

    while (unvisited_count > 0 && unvisited_count > (K - vehicle)) {
      int next = select_next(current, visited, n, tau, eta, alpha, beta,
                             rng_state);
      route_append(r, next);
      visited[next] = true;
      --unvisited_count;
      current = next;
    }

    route_append(r, 0);
  }

  if (unvisited_count > 0) {
    Route *last = &sol->routes[K - 1];
    int end_idx = last->len - 1;
    for (int j = 1; j <= n; ++j) {
      if (!visited[j]) {
        last->nodes[end_idx++] = j;
        visited[j] = true;
      }
    }
    last->nodes[end_idx++] = 0;
    last->len = end_idx;
  }
}

#ifdef USE_MPI
static int solution_pack_size_ints(int K, int n) {
  return 1 + K + K * (n + 2);
}

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
#endif

void aco_vrp_sequential(int n, int K, int m, int T, double **c,
                        double alpha, double beta, double rho, double tau0,
                        double Q, unsigned int seed, Solution *best_solution,
                        double *best_cost) {
  int mpi_rank = 0;
  int mpi_size = 1;
  bool mpi_enabled = false;

#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_enabled = (mpi_size > 1);
  }
#endif

  int local_m = m;
  int ant_offset = 0;
  if (mpi_enabled) {
    int base = (mpi_size > 0) ? (m / mpi_size) : m;
    int rem = (mpi_size > 0) ? (m % mpi_size) : 0;
    local_m = base + ((mpi_rank < rem) ? 1 : 0);
    ant_offset = mpi_rank * base + ((mpi_rank < rem) ? mpi_rank : rem);
  }

  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);

#ifdef USE_MPI
  int packed_len = mpi_enabled ? solution_pack_size_ints(K, n) : 0;
  int *packed_solution = mpi_enabled
                             ? calloc((size_t)packed_len, sizeof(int))
                             : NULL;
#endif

  int local_ok = (eta && tau && iter_best) ? 1 : 0;
#ifdef USE_MPI
  if (mpi_enabled && !packed_solution) {
    local_ok = 0;
  }
  if (mpi_enabled) {
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    local_ok = global_ok;
  }
#endif
  if (!local_ok) {
    fprintf(stderr, "allocation failure in aco_vrp_sequential\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
#ifdef USE_MPI
    free(packed_solution);
#endif
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (n > 16)
#endif
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        eta[i][j] = 0.0;
        tau[i][j] = 0.0;
      } else {
        eta[i][j] = 1.0 / (c[i][j] + EPS);
        tau[i][j] = tau0;
      }
    }
  }

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

#ifdef _OPENMP
  bool omp_enabled = (omp_get_max_threads() > 1 && local_m > 1);
#else
  bool omp_enabled = false;
#endif

  for (int iter = 0; iter < T; ++iter) {
    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;
    int iter_failed = 0;

    if (omp_enabled) {
#ifdef _OPENMP
#pragma omp parallel default(shared)
      {
        double thread_best_cost = DBL_MAX;
        int thread_best_ant = INT_MAX;
        Solution *sol = solution_create(K, n);
        Solution *thread_best = solution_create(K, n);
        bool *visited = calloc((size_t)(n + 1), sizeof(bool));

        if (!sol || !thread_best || !visited) {
#pragma omp critical
          { iter_failed = 1; }
        } else {
#pragma omp for schedule(static)
          for (int ant = 0; ant < local_m; ++ant) {
            int global_ant = ant_offset + ant;
            unsigned int rng_state = make_ant_seed(seed, iter, global_ant);

            build_ant_solution(sol, visited, n, K, tau, eta, alpha, beta,
                               &rng_state);
            double cost = solution_cost(sol, c);

            if (cost < thread_best_cost ||
                (fabs(cost - thread_best_cost) <= EPS &&
                 global_ant < thread_best_ant)) {
              thread_best_cost = cost;
              thread_best_ant = global_ant;
              solution_copy(thread_best, sol);
            }
          }

          if (thread_best_cost < DBL_MAX) {
#pragma omp critical
            {
              if (thread_best_cost < iter_best_cost ||
                  (fabs(thread_best_cost - iter_best_cost) <= EPS &&
                   thread_best_ant < iter_best_ant)) {
                iter_best_cost = thread_best_cost;
                iter_best_ant = thread_best_ant;
                solution_copy(iter_best, thread_best);
              }
            }
          }
        }

        free(visited);
        solution_free(thread_best);
        solution_free(sol);
      }
#endif
    } else {
      Solution *sol = solution_create(K, n);
      bool *visited = calloc((size_t)(n + 1), sizeof(bool));
      if (!sol || !visited) {
        iter_failed = 1;
      } else {
        for (int ant = 0; ant < local_m; ++ant) {
          int global_ant = ant_offset + ant;
          unsigned int rng_state = make_ant_seed(seed, iter, global_ant);

          build_ant_solution(sol, visited, n, K, tau, eta, alpha, beta,
                             &rng_state);
          double cost = solution_cost(sol, c);

          if (cost < iter_best_cost ||
              (fabs(cost - iter_best_cost) <= EPS &&
               global_ant < iter_best_ant)) {
            iter_best_cost = cost;
            iter_best_ant = global_ant;
            solution_copy(iter_best, sol);
          }
        }
      }

      free(visited);
      solution_free(sol);
    }

#ifdef USE_MPI
    if (mpi_enabled) {
      int global_failed = 0;
      MPI_Allreduce(&iter_failed, &global_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (global_failed) {
        fprintf(stderr, "allocation failure during iteration\n");
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
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
      if (local_has_solution && fabs(iter_best_cost - global_best_cost) <= EPS) {
        local_ant = iter_best_ant;
      }

      int global_best_ant = INT_MAX;
      MPI_Allreduce(&local_ant, &global_best_ant, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      int owner_rank = mpi_size;
      if (global_best_ant >= ant_offset && global_best_ant < ant_offset + local_m) {
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

      if (mpi_rank != winner_rank && !solution_unpack(iter_best, n, packed_solution)) {
        fprintf(stderr, "solution synchronization failure\n");
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
        free(packed_solution);
        return;
      }

      iter_best_cost = global_best_cost;
      iter_best_ant = global_best_ant;
    }
#endif

    if (iter_failed) {
      fprintf(stderr, "allocation failure during iteration\n");
      matrix_free(eta);
      matrix_free(tau);
      solution_free(iter_best);
#ifdef USE_MPI
      free(packed_solution);
#endif
      return;
    }

    if (iter_best_cost < *best_cost) {
      *best_cost = iter_best_cost;
      solution_copy(best_solution, iter_best);
    }

    if (iter_best_cost < DBL_MAX) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (n > 16)
#endif
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i != j) {
            tau[i][j] *= (1.0 - rho);
          }
        }
      }

      double deposit = Q / iter_best_cost;
      for (int i = 0; i < K; ++i) {
        Route *r = &iter_best->routes[i];
        for (int t = 0; t + 1 < r->len; ++t) {
          int u = r->nodes[t];
          int v = r->nodes[t + 1];
          tau[u][v] += deposit;
          tau[v][u] += deposit;
        }
      }
    }
  }

  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
#ifdef USE_MPI
  free(packed_solution);
#endif
}
