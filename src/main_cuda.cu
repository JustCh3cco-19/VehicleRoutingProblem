extern "C" {
# include "solver.h"
#include "cli_common.h"
#include "instance_parser.h"
#include "solution.h"
}

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Runs the CUDA ACO backend with explicit vehicle capacity.
 * @param n Number of customers.
 * @param k Number of vehicles.
 * @param vehicle_capacity_customers Per-vehicle customer capacity.
 * @param m Number of ants.
 * @param coords_x X coordinates.
 * @param coords_y Y coordinates.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param q Deposit factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 * @return 0 on success, non-zero on failure.
 */
t_status aco_vrp_cuda_with_capacity(int n, int k,
                                     int vehicle_capacity_customers, int m,
                                     float *coords_x, float *coords_y,
                                     double alpha, double beta, double rho,
                                     double tau0, double q, unsigned int seed,
                                     t_solution *best_solution,
                                     double *best_cost);

/**
 * @brief CLI entrypoint for the CUDA backend.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, non-zero on error.
 */
int main(int argc, char **argv) {
  int status = 0;
  t_cli_options options;
  t_vrp_instance instance;
  float *coords_x = NULL;
  float *coords_y = NULL;
  t_solution *best = NULL;
  double best_cost = 0.0;
  t_status solver_status = SOLVER_OK;
  t_cli_validation validation;
  vrp_instance_init(&instance);
  cli_options_defaults(&options);

  if (!cli_parse_solver_options(argc, argv, &options) ||
      options.mode != CLI_MODE_INSTANCE) {
    cli_print_usage(argv[0]);
    status = 1;
    goto cleanup;
  }

  if (vrp_load_tsplib_instance(options.instance_path, &instance) != 0 ||
      vrp_instance_create_float_coords(&instance, &coords_x, &coords_y) != 0) {
    status = 1;
    goto cleanup;
  }
  options.n = instance.n;

  if (instance.vehicles > 0 && instance.vehicles != options.k) {
    fprintf(stderr,
            "instance VEHICLES mismatch: CLI k=%d, file VEHICLES=%d\n",
            options.k, instance.vehicles);
    status = 1;
    goto cleanup;
  }

  best = solution_create(options.k, options.n);
  if (!best) {
    fprintf(stderr, "failed to allocate solution\n");
    status = 1;
    goto cleanup;
  }

  solver_status = aco_vrp_cuda_with_capacity(
      options.n, options.k, instance.capacity, options.m, coords_x, coords_y,
      options.alpha, options.beta, options.rho, options.tau0, options.q,
      options.seed, best, &best_cost);
  if (solver_status != SOLVER_OK) {
    fprintf(stderr, "CUDA solver failed: %s\n",
            status_string(solver_status));
    status = 1;
    goto cleanup;
  }

  validation.best = best;
  validation.n = options.n;
  validation.k = options.k;
  validation.demands = instance.demands;
  validation.vehicle_capacity = instance.capacity;
  validation.best_cost = best_cost;
  if (!cli_validate_solution_or_report(&validation)) {
    status = 1;
    goto cleanup;
  }

  cli_print_solution_routes(best, options.k);
  cli_print_solution_cost(best_cost);

cleanup:
  solution_free(best);
  vrp_instance_free(&instance);
  free(coords_x);
  free(coords_y);
  return (status);
}
