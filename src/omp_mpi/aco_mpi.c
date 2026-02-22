#include "aco.h"

#include "matrix.h"

#include "aco_common.h"
#include "aco_mpi_utils.h"

#include <float.h>
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
      rc = -1;                                                                    \
      goto cleanup;                                                               \
    }                                                                             \
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
  int *iter_lens = NULL;
  int *iter_nodes = NULL;

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

  int max_route_len = n + 2;
  iter_lens = malloc((size_t)K * sizeof(int));
  iter_nodes = malloc((size_t)K * (size_t)max_route_len * sizeof(int));

  int local_ok = (eta && tau && iter_best_global && iter_best_local &&
                  iter_lens && iter_nodes)
                     ? 1
                     : 0;
  int global_ok = 0;
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

  for (int iter = 0; iter < T; ++iter) {
    double local_best_cost = DBL_MAX;
    int local_best_ant_id = INT_MAX;
    int worker_failed = 0;

#ifdef _OPENMP
/* default(none) prevents implicit sharing mistakes in hybrid regions. */
#pragma omp parallel num_threads(num_threads) default(none)                     \
    shared(n, K, local_ant_count, local_ant_offset, seed, iter, tau, eta,      \
           alpha, beta, c, iter_best_local, local_best_cost,                    \
           local_best_ant_id, stderr)                                            \
    reduction(max:worker_failed)
#endif
    {
      Solution *thread_ant_sol = solution_create(K, n);
      Solution *thread_best_sol = solution_create(K, n);
      bool *visited = calloc((size_t)(n + 1), sizeof(bool));
      double thread_best_cost = DBL_MAX;
      int thread_best_ant_id = INT_MAX;

      if (!thread_ant_sol || !thread_best_sol || !visited) {
        worker_failed = 1;
#ifdef _OPENMP
/* Serialize only the merge; ant evaluation stays fully parallel. */
#pragma omp critical
#endif
        {
          fprintf(stderr, "allocation failure in MPI/OpenMP worker\n");
        }
      } else {
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

      free(visited);
      solution_free(thread_ant_sol);
      solution_free(thread_best_sol);
    }

    int local_iter_ok = worker_failed ? 0 : 1;
    int global_iter_ok = 0;
    /* Every rank participates in the same collective order to avoid deadlock. */
    MPI_CHECK(MPI_Allreduce(&local_iter_ok, &global_iter_ok, 1, MPI_INT,
                            MPI_MIN, MPI_COMM_WORLD));
    if (!global_iter_ok) {
      rc = -1;
      goto cleanup;
    }

    struct {
      double value;
      int index;
    } local_pair, global_pair;

    local_pair.value = local_best_cost;
    local_pair.index = local_best_ant_id;

    MPI_CHECK(MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT,
                            MPI_MINLOC, MPI_COMM_WORLD));

    int owner_candidate =
        (local_best_ant_id == global_pair.index) ? rank : world_size;
    int owner_rank = world_size;
    MPI_CHECK(MPI_Allreduce(&owner_candidate, &owner_rank, 1, MPI_INT,
                            MPI_MIN, MPI_COMM_WORLD));

    if (owner_rank < world_size) {
      if (rank == owner_rank) {
        aco_mpi_solution_to_flat(iter_best_local, K, n, iter_lens, iter_nodes);
      }
      /* Broadcast route shape + payload from owner to all ranks. */
      MPI_CHECK(MPI_Bcast(iter_lens, K, MPI_INT, owner_rank, MPI_COMM_WORLD));
      MPI_CHECK(MPI_Bcast(iter_nodes, K * max_route_len, MPI_INT,
                          owner_rank, MPI_COMM_WORLD));
      aco_mpi_flat_to_solution(iter_best_global, K, n, iter_lens, iter_nodes);
    } else {
      solution_reset(iter_best_global);
    }

    if (aco_better_candidate(global_pair.value, global_pair.index,
                             *best_cost, best_ant_id)) {
      *best_cost = global_pair.value;
      best_ant_id = global_pair.index;
      solution_copy(best_solution, iter_best_global);
    }

    aco_update_pheromones(tau, iter_best_global, global_pair.value,
                          n, K, rho, Q);

    if (((iter + 1) % sync_every) == 0 || (iter + 1) == T) {
      int matrix_size = (n + 1) * (n + 1);
      /* Average all local tau matrices into a consistent global view. */
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
  free(iter_lens);
  free(iter_nodes);
  return rc;
}
