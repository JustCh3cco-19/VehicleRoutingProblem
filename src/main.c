#include "aco.h"
#include "cli_common.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

/**
 * @brief Fills an example distance matrix used in local/demo execution mode.
 * @param c Distance matrix.
 * @param n Number of customers.
 */
static void fill_example_costs(double **c, int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        c[i][j] = 1.0 + fabs((double)i - (double)j);
      }
    }
  }
}

/**
 * @brief CLI entrypoint for sequential and MPI/OpenMP backends.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, non-zero on error.
 */
int main(int argc, char **argv) {
  int status = 0;
  unsigned int seed = 1234u;
  int use_instance_file = 0;
  const char *instance_path = NULL;
  VrpInstanceMeta instance_meta = {0};

#ifdef USE_MPI
  int mpi_rank = 0;
  int provided = 0;
  if (MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init_thread failed\n");
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  int n = 5, K = 2, m = 10;
  if (argc == 5 && cli_parse_int_arg(argv[1], &n)) {
    int ok = 1;
    ok = ok && cli_parse_int_arg(argv[2], &K);
    ok = ok && cli_parse_int_arg(argv[3], &m);
    seed = cli_parse_uint_arg(argv[4], &ok);
    if (!ok || n <= 0 || K <= 0 || m < 0) {
#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
        fprintf(stderr, "usage: %s [n K m seed]\n", argv[0]);
      status = 1; goto cleanup_mpi;
    }
  } else if ((argc == 4 || argc == 5) && argv[1] != NULL) {
    int ok = 1; instance_path = argv[1];
    ok = ok && cli_parse_int_arg(argv[2], &K);
    ok = ok && cli_parse_int_arg(argv[3], &m);
    if (argc == 5) seed = cli_parse_uint_arg(argv[4], &ok);
    if (!ok || K <= 0 || m < 0) {
#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
        fprintf(stderr, "usage: %s <instance.vrp> <K> <m> [seed]\n", argv[0]);
      status = 1; goto cleanup_mpi;
    }
    use_instance_file = 1;
  } else if (argc != 1) {
#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
      fprintf(stderr, "usage: %s [options]\n", argv[0]);
    status = 1; goto cleanup_mpi;
  }

  double alpha = 1.0, beta = 2.0, rho = 0.5, tau0 = 1.0, Q = 1.0;
  double **c = NULL;
  if (use_instance_file) {
    if (vrp_load_tsplib_euc2d_matrix_ex(instance_path, &n, &c, &instance_meta) != 0) {
#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
        fprintf(stderr, "failed to load instance: %s\n", instance_path);
      status = 1; goto cleanup_mpi;
    }
  } else {
    c = matrix_alloc(n);
    if (!c) { fprintf(stderr, "failed to allocate cost matrix\n"); status = 1; goto cleanup_mpi; }
    fill_example_costs(c, n);
  }

  Solution *best = solution_create(K, n);
  if (!best) { fprintf(stderr, "failed to allocate solution\n"); status = 1; matrix_free(c); goto cleanup_mpi; }

  double best_cost = 0.0;
  AcoStatus solver_status;
  if (use_instance_file) {
    if (instance_meta.vehicles > 0 && instance_meta.vehicles != K) {
      fprintf(stderr, "instance VEHICLES mismatch: CLI K=%d, file VEHICLES=%d\n", K, instance_meta.vehicles);
      status = 1; solution_free(best); matrix_free(c); goto cleanup_mpi;
    }
    solver_status = aco_vrp_with_capacity(n, K, instance_meta.capacity, m, c,
                                          alpha, beta, rho, tau0, Q, seed,
                                          best, &best_cost);
  } else {
    solver_status = aco_vrp(n, K, m, c, alpha, beta, rho, tau0, Q, seed, best,
                            &best_cost);
  }

#ifdef USE_MPI
  if (mpi_rank == 0)
#endif
  {
    if (solver_status != ACO_OK) {
      fprintf(stderr, "solver failed: %s\n", aco_status_string(solver_status));
      status = 1;
    } else if (!cli_validate_solution_or_report(best, n, K, best_cost)) {
      status = 1;
    } else {
      cli_print_solution_routes(best, K);
      cli_print_solution_cost(best_cost);
    }
  }

  solution_free(best);
  matrix_free(c);

cleanup_mpi:
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return status;
}
