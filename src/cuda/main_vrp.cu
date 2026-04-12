extern "C" {
#include "aco.h"
#include "instance_parser.h"
#include "solution.h"
}

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

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
int aco_vrp_cuda_with_capacity(int n, int K, int vehicle_capacity_customers,
                               int m, float *coords_x, float *coords_y,
                               double alpha, double beta, double rho,
                               double tau0, double Q, unsigned int seed,
                               Solution *best_solution, double *best_cost);

/**
 * @brief Parses a positive integer argument.
 * @param s Input string.
 * @param out Output parsed value.
 * @return 1 on success, 0 on parse/range error.
 */
static int parse_int_arg(const char *s, int *out) {
    char *end = NULL;
    long v;
    errno = 0;
    v = strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return 0;
    }
    if (v < 0 || v > 100000000L) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

/**
 * @brief Parses an unsigned integer argument.
 * @param s Input string.
 * @param ok Output parse status flag.
 * @return Parsed value if valid, 0 otherwise.
 */
static unsigned int parse_uint_arg(const char *s, int *ok) {
    char *end = NULL;
    unsigned long v;
    errno = 0;
    v = strtoul(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFUL) {
        *ok = 0;
        return 0u;
    }
    *ok = 1;
    return (unsigned int)v;
}

/**
 * @brief Prints routes from a solution in human-readable format.
 * @param best Best solution to print.
 * @param K Number of routes.
 */
static void print_solution_routes(const Solution *best, int K) {
    int i;
    for (i = 0; i < K; ++i) {
        const Route *r = &best->routes[i];
        int t;
        int printed = 0;
        printf("Route %d:", i + 1);
        for (t = 0; t < r->len; ++t) {
            int node = r->nodes[t];
            if (node != 0) {
                printf("%s%d", printed ? " " : " ", node);
                printed = 1;
            }
        }
        printf("\n");
    }
}

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
    VrpInstanceMeta instance_meta = {0};
    float *coords_x = NULL;
    float *coords_y = NULL;
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.5;
    double tau0 = 1.0;
    double Q = 1.0;
    Solution *best;
    double best_cost = 0.0;

    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s <instance.vrp> <K> <m> [seed]\n", argv[0]);
        fprintf(stderr, "example: %s instances/test_aligned/n500_k8_s19000.vrp 8 0 1234\n", argv[0]);
        return 1;
    }

    path = argv[1];
    ok = ok && parse_int_arg(argv[2], &K);
    ok = ok && parse_int_arg(argv[3], &m);
    if (argc == 5) {
        seed = parse_uint_arg(argv[4], &ok);
    }

    if (!ok || K <= 0 || m < 0) {
        fprintf(stderr, "usage: %s <instance.vrp> <K> <m> [seed]\n", argv[0]);
        return 1;
    }

    if (vrp_load_tsplib_euc2d_coords(path, &n, &coords_x, &coords_y, &instance_meta) != 0) {
        return 1;
    }

    if (instance_meta.vehicles > 0 && instance_meta.vehicles != K) {
        fprintf(stderr, "instance VEHICLES mismatch: CLI K=%d, file VEHICLES=%d\n",
                K, instance_meta.vehicles);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    best = solution_create(K, n);
    if (!best) {
        fprintf(stderr, "failed to allocate solution\n");
        free(coords_x);
        free(coords_y);
        return 1;
    }

    if (aco_vrp_cuda_with_capacity(n, K, instance_meta.capacity, m, coords_x, coords_y, alpha,
                                      beta, rho, tau0, Q, seed, best,
                                      &best_cost) != 0) {
        fprintf(stderr, "CUDA solver failed\n");
        solution_free(best);
        free(coords_x);
        free(coords_y);
        return 1;
    }

    print_solution_routes(best, K);
    printf("Cost: %.3f\n", best_cost);
    printf("best cost: %.3f\n", best_cost);

    solution_free(best);
    free(coords_x);
    free(coords_y);
    return 0;
}
