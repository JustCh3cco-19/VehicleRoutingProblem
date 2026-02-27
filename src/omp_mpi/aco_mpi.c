#include "aco.h"

#include "matrix.h"

#include "aco_common.h"
#include "aco_mpi_utils.h"

#include <float.h>
#include <math.h>
#include <limits.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MPI_CHECK(call)                                                          \
  do {                                                                           \
    int mpi_err__ = (call);                                                      \
    if (mpi_err__ != MPI_SUCCESS) {                                              \
      char mpi_err_str__[MPI_MAX_ERROR_STRING];                                  \
      int mpi_err_len__ = 0;                                                     \
      MPI_Error_string(mpi_err__, mpi_err_str__, &mpi_err_len__);                \
      fprintf(stderr, "MPI error %s:%d (%s): %.*s\\n", __FILE__, __LINE__,       \
              #call, mpi_err_len__, mpi_err_str__);                              \
      rc = -1;                                                                   \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

int aco_vrp_mpi_openmp(int n, int K, int m, int T, double **c,
                       double alpha, double beta, double rho,
                       double tau0, double Q, unsigned int seed,
                       int num_threads, int sync_every,
                       Solution *best_solution, double *best_cost) {
  int rc = 0;
  int mpi_initialized = 0;
  int rank = 0;
  int world_size = 1;

  int local_ant_offset = 0;
  int local_ant_count = 0;

  double **eta = NULL;
  double **tau = NULL;
  Solution *iter_best_global = NULL;
  Solution *iter_best_local = NULL;
  Solution *window_best_local = NULL;
  int *iter_lens = NULL;
  int *iter_nodes = NULL;
  
  Solution **thread_ant_sols = NULL;
  Solution **thread_best_sols = NULL;
  bool **thread_visited = NULL;

  MPI_CHECK(MPI_Initialized(&mpi_initialized));
  if (!mpi_initialized) {
    fprintf(stderr, "aco_vrp_mpi_openmp requires MPI_Init before invocation\n");
    return -1;
  }

  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  if (n <= 0 || K <= 0 || m <= 0 || T <= 0) {
    if (rank == 0) {
      fprintf(stderr, "invalid input sizes (n, K, m, T must be > 0)\n");
    }
    rc = -1;
    goto cleanup;
  }

  if (sync_every <= 0) {
    sync_every = 1;
  }

  if (num_threads <= 0) {
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#else
    num_threads = 1;
#endif
  }

  aco_mpi_ant_range_for_rank(m, rank, world_size,
                             &local_ant_offset, &local_ant_count);

  eta = matrix_alloc(n);
  tau = matrix_alloc(n);
  iter_best_global = solution_create(K, n);
  iter_best_local = solution_create(K, n);
  window_best_local = solution_create(K, n);

  int max_route_len = n + 2;
  iter_lens = malloc((size_t)K * sizeof(int));
  iter_nodes = malloc((size_t)K * (size_t)max_route_len * sizeof(int));

  /* Allocazione strutture dati thread-local al di fuori del ciclo */
  thread_ant_sols = malloc(num_threads * sizeof(Solution *));
  thread_best_sols = malloc(num_threads * sizeof(Solution *));
  thread_visited = malloc(num_threads * sizeof(bool *));

  int local_ok = (eta && tau && iter_best_global && iter_best_local &&
                  window_best_local && iter_lens && iter_nodes &&
                  thread_ant_sols && thread_best_sols && thread_visited)
                     ? 1
                     : 0;
                     
  if (local_ok) {
    for (int i = 0; i < num_threads; ++i) {
      thread_ant_sols[i] = solution_create(K, n);
      thread_best_sols[i] = solution_create(K, n);
      thread_visited[i] = calloc((size_t)(n + 1), sizeof(bool));
      if (!thread_ant_sols[i] || !thread_best_sols[i] || !thread_visited[i]) {
        local_ok = 0;
        break;
      }
    }
  }

  int global_ok = 0;
  /* Unica chiamata MPI di verifica, effettuata una volta sola nel setup iniziale */
  MPI_CHECK(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                          MPI_COMM_WORLD));

  if (!global_ok) {
    if (rank == 0) {
      fprintf(stderr, "allocation failure in aco_vrp_mpi_openmp setup\n");
    }
    rc = -1;
    goto cleanup;
  }

  aco_init_eta_tau(n, c, eta, tau, tau0);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;
  int best_ant_id = INT_MAX;
  double window_best_cost = DBL_MAX;
  int window_best_ant_id = INT_MAX;

  for (int iter = 0; iter < T; ++iter) {
    double local_best_cost = DBL_MAX;
    int local_best_ant_id = INT_MAX;
    solution_reset(iter_best_local);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads) default(none)                     \
    shared(n, K, local_ant_count, local_ant_offset, seed, iter, tau, eta,       \
           alpha, beta, c, iter_best_local, local_best_cost,                    \
           local_best_ant_id, thread_ant_sols, thread_best_sols, thread_visited)
#endif
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      Solution *thread_ant_sol = thread_ant_sols[tid];
      Solution *thread_best_sol = thread_best_sols[tid];
      bool *visited = thread_visited[tid];

      double thread_best_cost = DBL_MAX;
      int thread_best_ant_id = INT_MAX;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int local_ant = 0; local_ant < local_ant_count; ++local_ant) {
        int global_ant = local_ant_offset + local_ant;
        unsigned int local_seed = aco_ant_seed(seed, iter, global_ant);

        aco_construct_ant_solution(thread_ant_sol, visited, n, K,
                                   tau, eta, alpha, beta, local_seed);
        double cost = solution_cost(thread_ant_sol, c);

        if (aco_better_candidate(cost, global_ant,
                                 thread_best_cost, thread_best_ant_id)) {
          thread_best_cost = cost;
          thread_best_ant_id = global_ant;
          solution_copy(thread_best_sol, thread_ant_sol);
        }
      }

#ifdef _OPENMP
#pragma omp critical
#endif
      {
        if (aco_better_candidate(thread_best_cost, thread_best_ant_id,
                                 local_best_cost, local_best_ant_id)) {
          local_best_cost = thread_best_cost;
          local_best_ant_id = thread_best_ant_id;
          solution_copy(iter_best_local, thread_best_sol);
        }
      }
    }

    /* Rimossa la chiamata MPI_Allreduce() in questo punto */

    if (isfinite(local_best_cost) &&
        aco_better_candidate(local_best_cost, local_best_ant_id,
                             window_best_cost, window_best_ant_id)) {
      window_best_cost = local_best_cost;
      window_best_ant_id = local_best_ant_id;
      solution_copy(window_best_local, iter_best_local);
    }

    if (isfinite(local_best_cost)) {
      aco_update_pheromones(tau, iter_best_local, local_best_cost,
                            n, K, rho, Q);
    }

    if (((iter + 1) % sync_every) == 0 || (iter + 1) == T) {
      struct {
        double value;
        int index;
      } local_pair, global_pair;

      local_pair.value = window_best_cost;
      local_pair.index = window_best_ant_id;
      MPI_CHECK(MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT,
                              MPI_MINLOC, MPI_COMM_WORLD));

      int owner_rank = aco_mpi_rank_for_ant(m, world_size, global_pair.index);
      if (owner_rank < world_size) {
        int iter_nodes_count = 0;
        if (rank == owner_rank) {
          iter_nodes_count =
              aco_mpi_solution_to_flat_packed(window_best_local, K,
                                              iter_lens, iter_nodes);
        }
        MPI_CHECK(MPI_Bcast(iter_lens, K, MPI_INT, owner_rank, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(&iter_nodes_count, 1, MPI_INT, owner_rank,
                            MPI_COMM_WORLD));
        MPI_CHECK(MPI_Bcast(iter_nodes, iter_nodes_count, MPI_INT,
                            owner_rank, MPI_COMM_WORLD));
        aco_mpi_flat_packed_to_solution(iter_best_global, K, iter_lens,
                                        iter_nodes_count, iter_nodes);
      } else {
        solution_reset(iter_best_global);
      }

      if (aco_better_candidate(global_pair.value, global_pair.index,
                               *best_cost, best_ant_id)) {
        *best_cost = global_pair.value;
        best_ant_id = global_pair.index;
        solution_copy(best_solution, iter_best_global);
      }

      window_best_cost = DBL_MAX;
      window_best_ant_id = INT_MAX;

      int matrix_size = (n + 1) * (n + 1);
      MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, tau[0], matrix_size,
                              MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      double scale = 1.0 / (double)world_size;
      for (int idx = 0; idx < matrix_size; ++idx) {
        tau[0][idx] *= scale;
      }
    }
  }

cleanup:
  matrix_free(eta);
  matrix_free(tau);
  solution_free(iter_best_global);
  solution_free(iter_best_local);
  solution_free(window_best_local);
  free(iter_lens);
  free(iter_nodes);
  
  if (thread_ant_sols) {
    for (int i = 0; i < num_threads; ++i) {
      if (thread_ant_sols[i]) solution_free(thread_ant_sols[i]);
    }
    free(thread_ant_sols);
  }
  if (thread_best_sols) {
    for (int i = 0; i < num_threads; ++i) {
      if (thread_best_sols[i]) solution_free(thread_best_sols[i]);
    }
    free(thread_best_sols);
  }
  if (thread_visited) {
    for (int i = 0; i < num_threads; ++i) {
      if (thread_visited[i]) free(thread_visited[i]);
    }
    free(thread_visited);
  }

  return rc;
}