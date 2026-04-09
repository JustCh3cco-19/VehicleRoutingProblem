extern "C" {
#include "aco.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"
}

#include <stdio.h>
#include <stdlib.h>

// Forward declaration del solver CUDA (C++ name mangling tra file .cu)
int aco_vrp_cuda(int n, int K, int m, double timeout_minutes, double improvement_threshold,
                 double **c, double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost);

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <instance.vrp> <K> <m> <timeout_minutes> [improvement_threshold] [seed]\n", argv[0]);
        fprintf(stderr, "Example: %s instances/benchmark/eil33.vrp 4 64 2.0 0.001 1234\n", argv[0]);
        fprintf(stderr, "Note: m=0 uses optimized default (64 ants)\n");
        return 1;
    }

    const char *path = argv[1];
    int K = atoi(argv[2]);
    int m = atoi(argv[3]);
    double timeout_minutes = atof(argv[4]);
    double improvement_threshold = (argc > 5) ? atof(argv[5]) : 0.001; // 0.1% default
    unsigned int seed = (argc > 6) ? (unsigned int)atoi(argv[6]) : 1234;

    int n = 0;
    double **c = NULL;
    if (vrp_load_tsplib_euc2d_matrix(path, &n, &c) != 0) {
        return 1;
    }

    // Parametri ACO standard (ora interni al main o configurabili via macro)
    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.1;
    double tau0 = 1.0;
    double Q = 100.0;

    Solution *best = solution_create(K, n);
    double best_cost = 0.0;

    printf("Starting Hardware-Aware CUDA Solver (Timer: %.2f min, Early Stoppage: %.4f)\n", timeout_minutes, improvement_threshold);
    
    if (aco_vrp_cuda(n, K, m, timeout_minutes, improvement_threshold, c, alpha, beta, rho, tau0, Q, seed, best, &best_cost) != 0) {
        fprintf(stderr, "CUDA solver failed\n");
        solution_free(best);
        matrix_free(c);
        return 1;
    }

    printf("\n--- Best Solution Found ---\n");
    printf("Final Best Cost: %.3f\n", best_cost);
    for (int i = 0; i < K; ++i) {
        Route *r = &best->routes[i];
        if (r->len > 2) {
            printf("Route %d: ", i + 1);
            for (int t = 0; t < r->len; ++t) {
                printf("%d ", r->nodes[t]);
            }
            printf("\n");
        }
    }

    solution_free(best);
    matrix_free(c);
    return 0;
}
