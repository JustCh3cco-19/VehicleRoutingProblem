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

/*
 * Function:  choose_l1_lines
 * --------------------------
 * picks L1 cache line count for desirability-row cache.
 */
static int choose_l1_lines(int n) {
  int side = n + 1;
  return (side < 8) ? side : 8;
}

/*
 * Function:  choose_l2_lines
 * --------------------------
 * picks L2 cache line count for desirability-row cache.
 */
static int choose_l2_lines(int n) {
  int side = n + 1;
  return (side < 64) ? side : 64;
}

#ifdef USE_MPI
/*
 * Function:  solution_pack_size_ints
 * ----------------------------------
 * computes the integer buffer length required to serialize a Solution with K
 * routes and per-route capacity n+2.
 *
 *  K: number of routes
 *  n: number of customers
 *
 *  returns: number of ints needed in packed representation
 */
static int solution_pack_size_ints(int K, int n) {
  return 1 + K + K * (n + 2);
}

/*
 * Function:  solution_pack
 * ------------------------
 * serializes a Solution into an integer buffer for MPI broadcast.
 *
 *  sol: source solution
 *  n: number of customers
 *  buf: destination integer buffer with size solution_pack_size_ints(K, n)
 *
 *  returns: nothing
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
 * deserializes a packed solution buffer into an existing Solution object.
 *
 *  sol: destination solution (already allocated with matching K/capacities)
 *  n: number of customers
 *  buf: serialized integer buffer
 *
 *  returns: true on successful unpack
 *           false if packed metadata is inconsistent
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
#endif

/*
 * Function:  aco_vrp
 * ------------------
 * runs the hybrid OpenMP+MPI ACO solver for VRP.
 * algorithm outline:
 * 1) split ants across MPI ranks
 * 2) initialize shared eta/tau matrices
 * 3) for each iteration, evaluate local ants (OpenMP in-rank parallelism)
 * 4) reduce local iteration-best to global best with MPI collectives
 * 5) synchronize winning solution across ranks
 * 6) update pheromone matrix from global iteration-best solution
 *
 *  n: number of customers (1..n), with depot at 0
 *  K: number of routes/vehicles
 *  m: total ants per iteration across all ranks
 *  T: number of iterations
 *  c: cost matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  rho: evaporation factor
 *  tau0: initial pheromone value
 *  Q: deposit scale
 *  seed: base random seed
 *  best_solution: output container for best route set
 *  best_cost: output best objective value
 *
 *  returns: nothing; on allocation or synchronization failure prints an error
 *           and returns early
 */
void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost) {
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
  int *packed_solution =
      mpi_enabled ? calloc((size_t)packed_len, sizeof(int)) : NULL;
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
    fprintf(stderr, "allocation failure in aco_vrp\n");
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
        eta[i][j] = 1.0 / (c[i][j] + ACO_EPS);
        tau[i][j] = tau0;
      }
    }
  }

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  int vehicle_capacity_customers = (K > 0) ? ((n + K - 1) / K) : n;
  if (vehicle_capacity_customers < 1) {
    vehicle_capacity_customers = 1;
  }

#ifdef _OPENMP
  bool omp_enabled = (omp_get_max_threads() > 1 && local_m > 1);
#else
  bool omp_enabled = false;
#endif

  AcoScoreCache *serial_cache = NULL;
  if (!omp_enabled) {
    serial_cache =
        aco_score_cache_create(n, choose_l1_lines(n), choose_l2_lines(n));
  }

  size_t scratch_len = (n > 0) ? (size_t)n : 1u;

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
        AcoScoreCache *score_cache =
            aco_score_cache_create(n, choose_l1_lines(n), choose_l2_lines(n));
        int *unvisited_nodes = malloc(scratch_len * sizeof(int));
        double *candidate_scores = malloc(scratch_len * sizeof(double));
        double *random_draws = malloc(scratch_len * sizeof(double));

        if (!sol || !thread_best || !unvisited_nodes ||
            !candidate_scores || !random_draws) {
#pragma omp critical
          { iter_failed = 1; }
        } else {
#pragma omp for schedule(static)
          for (int ant = 0; ant < local_m; ++ant) {
            int global_ant = ant_offset + ant;
            unsigned int rng_state = aco_make_ant_seed(seed, iter, global_ant);

            aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                                   vehicle_capacity_customers, score_cache,
                                   &rng_state, unvisited_nodes,
                                   candidate_scores, random_draws);
            double cost = solution_cost(sol, c);

            if (cost < thread_best_cost ||
                (fabs(cost - thread_best_cost) <= ACO_EPS &&
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
                  (fabs(thread_best_cost - iter_best_cost) <= ACO_EPS &&
                   thread_best_ant < iter_best_ant)) {
                iter_best_cost = thread_best_cost;
                iter_best_ant = thread_best_ant;
                solution_copy(iter_best, thread_best);
              }
            }
          }
        }

        free(random_draws);
        free(candidate_scores);
        free(unvisited_nodes);
        aco_score_cache_free(score_cache);
        solution_free(thread_best);
        solution_free(sol);
      }
#endif
    } else {
      aco_score_cache_invalidate(serial_cache);
      Solution *sol = solution_create(K, n);
      int *unvisited_nodes = malloc(scratch_len * sizeof(int));
      double *candidate_scores = malloc(scratch_len * sizeof(double));
      double *random_draws = malloc(scratch_len * sizeof(double));

      if (!sol || !unvisited_nodes || !candidate_scores ||
          !random_draws) {
        iter_failed = 1;
      } else {
        for (int ant = 0; ant < local_m; ++ant) {
          int global_ant = ant_offset + ant;
          unsigned int rng_state = aco_make_ant_seed(seed, iter, global_ant);

          aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                                 vehicle_capacity_customers, serial_cache,
                                 &rng_state, unvisited_nodes,
                                 candidate_scores, random_draws);
          double cost = solution_cost(sol, c);

          if (cost < iter_best_cost ||
              (fabs(cost - iter_best_cost) <= ACO_EPS &&
               global_ant < iter_best_ant)) {
            iter_best_cost = cost;
            iter_best_ant = global_ant;
            solution_copy(iter_best, sol);
          }
        }
      }

      free(random_draws);
      free(candidate_scores);
      free(unvisited_nodes);
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
        aco_score_cache_free(serial_cache);
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
      if (local_has_solution && fabs(iter_best_cost - global_best_cost) <= ACO_EPS) {
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

      if (mpi_rank != winner_rank &&
          !solution_unpack(iter_best, n, packed_solution)) {
        fprintf(stderr, "solution synchronization failure\n");
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
        aco_score_cache_free(serial_cache);
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
      aco_score_cache_free(serial_cache);
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
  aco_score_cache_free(serial_cache);
#ifdef USE_MPI
  free(packed_solution);
#endif
}
