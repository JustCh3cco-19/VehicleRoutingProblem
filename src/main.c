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
  AcoCliOptions options;
  VrpInstance instance;
  double **c = NULL;
  Solution *best = NULL;
  double best_cost = 0.0;
  AcoStatus solver_status = ACO_OK;
  vrp_instance_init(&instance);
  cli_options_defaults(&options);

#ifdef USE_MPI
  int mpi_rank = 0;
  int provided = 0;
  if (MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init_thread failed\n");
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  if (!cli_parse_solver_options(argc, argv, &options)) {
#ifdef USE_MPI
    if (mpi_rank == 0)
#endif
    {
      cli_print_usage(argv[0]);
    }
    status = 1;
    goto cleanup;
  }

  if (options.mode == ACO_CLI_MODE_INSTANCE) {
    if (vrp_load_tsplib_instance(options.instance_path, &instance) != 0 ||
        vrp_instance_create_euc2d_matrix(&instance, &c) != 0) {
#ifdef USE_MPI
      if (mpi_rank == 0)
#endif
        fprintf(stderr, "failed to load instance: %s\n", options.instance_path);
      status = 1;
      goto cleanup;
    }
    options.n = instance.n;
  } else {
    c = matrix_alloc(options.n);
    if (!c) {
      fprintf(stderr, "failed to allocate cost matrix\n");
      status = 1;
      goto cleanup;
    }
    fill_example_costs(c, options.n);
  }

  best = solution_create(options.K, options.n);
  if (!best) {
    fprintf(stderr, "failed to allocate solution\n");
    status = 1;
    goto cleanup;
  }

  if (options.mode == ACO_CLI_MODE_INSTANCE) {
    if (instance.vehicles > 0 && instance.vehicles != options.K) {
      fprintf(stderr, "instance VEHICLES mismatch: CLI K=%d, file VEHICLES=%d\n",
              options.K, instance.vehicles);
      status = 1;
      goto cleanup;
    }
    solver_status = aco_vrp_with_capacity(
        options.n, options.K, instance.capacity, options.m, c, options.alpha,
        options.beta, options.rho, options.tau0, options.Q, options.seed, best,
        &best_cost);
  } else {
    solver_status = aco_vrp(options.n, options.K, options.m, c, options.alpha,
                            options.beta, options.rho, options.tau0, options.Q,
                            options.seed, best, &best_cost);
  }

#ifdef USE_MPI
  if (mpi_rank == 0)
#endif
  {
    if (solver_status != ACO_OK) {
      fprintf(stderr, "solver failed: %s\n", aco_status_string(solver_status));
      status = 1;
    } else if (!cli_validate_solution_or_report(
                   best, options.n, options.K,
                   options.mode == ACO_CLI_MODE_INSTANCE ? instance.demands
                                                         : NULL,
                   options.mode == ACO_CLI_MODE_INSTANCE ? instance.capacity
                                                         : 0,
                   best_cost)) {
      status = 1;
    } else {
      cli_print_solution_routes(best, options.K);
      cli_print_solution_cost(best_cost);
    }
  }

cleanup:
  solution_free(best);
  matrix_free(c);
  vrp_instance_free(&instance);

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return status;
}
