extern "C" {
#include "aco.h"
#include "cli_common.h"
#include "instance_parser.h"
#include "solution.h"
}

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Runs the CUDA ACO backend with explicit vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param vehicle_capacity_customers Per-vehicle customer capacity.
 * @param m Number of ants.
 * @param coords_x X coordinates.
 * @param coords_y Y coordinates.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 * @return 0 on success, non-zero on failure.
 */
AcoStatus aco_vrp_cuda_with_capacity(int n, int K,
                                     int vehicle_capacity_customers, int m,
                                     float *coords_x, float *coords_y,
                                     double alpha, double beta, double rho,
                                     double tau0, double Q, unsigned int seed,
                                     Solution *best_solution,
                                     double *best_cost);

/**
 * @brief CLI entrypoint for the CUDA backend.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return 0 on success, non-zero on error.
 */
int main(int argc, char **argv) {
    const char *path;
    int K = 0;
    int m = 0;
    int ok = 1;
    unsigned int seed = 1234u;
    int n = 0;
    VrpInstance instance;
    float *coords_x = NULL;
    float *coords_y = NULL;
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.5;
    double tau0 = 1.0;
    double Q = 1.0;
    Solution *best;
    double best_cost = 0.0;
    vrp_instance_init(&instance);

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <instance.vrp> <K> <m> [seed]\n", argv[0]);
        fprintf(stderr, "example: %s instances/generated_benchmark/n500_k8_s19000.vrp 8 0 1234\n", argv[0]);
        return 1;
    }

    path = argv[1];
    ok = ok && cli_parse_int_arg(argv[2], &K);
    ok = ok && cli_parse_int_arg(argv[3], &m);
    if (argc == 5) {
        seed = cli_parse_uint_arg(argv[4], &ok);
    }

    if (!ok || K <= 0 || m < 0) {
        fprintf(stderr, "usage: %s <instance.vrp> <K> <m> [seed]\n", argv[0]);
        return 1;
    }

    if (vrp_load_tsplib_instance(path, &instance) != 0 ||
        vrp_instance_create_float_coords(&instance, &coords_x, &coords_y) != 0) {
        vrp_instance_free(&instance);
        return 1;
    }
    n = instance.n;

    if (instance.vehicles > 0 && instance.vehicles != K) {
        fprintf(stderr, "instance VEHICLES mismatch: CLI K=%d, file VEHICLES=%d\n",
                K, instance.vehicles);
        vrp_instance_free(&instance);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    best = solution_create(K, n);
    if (!best) {
        fprintf(stderr, "failed to allocate solution\n");
        vrp_instance_free(&instance);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    AcoStatus solver_status = aco_vrp_cuda_with_capacity(n, K, instance.capacity, m, coords_x, coords_y, alpha,
                                      beta, rho, tau0, Q, seed, best,
                                      &best_cost);
    if (solver_status != ACO_OK) {
        fprintf(stderr, "CUDA solver failed: %s\n", aco_status_string(solver_status));
        solution_free(best);
        vrp_instance_free(&instance);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    if (!cli_validate_solution_or_report(best, n, K, instance.demands,
                                         instance.capacity, best_cost)) {
        solution_free(best);
        vrp_instance_free(&instance);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    cli_print_solution_routes(best, K);
    cli_print_solution_cost(best_cost);

    solution_free(best);
    vrp_instance_free(&instance);
    free(coords_x);
    free(coords_y);
    return 0;
}
