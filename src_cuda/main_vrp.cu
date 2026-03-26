#include "aco.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <instance.vrp> <K> <m> <T> [seed] [convergence_iter] [epsilon_rel]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int K = atoi(argv[2]);
    int m = atoi(argv[3]);
    int T = atoi(argv[4]);
    unsigned int seed = (argc > 5) ? (unsigned int)atoi(argv[5]) : 1234;
    int convergence_iter = (argc > 6) ? atoi(argv[6]) : 1000;
    double epsilon = (argc > 7) ? atof(argv[7]) : 0.01;

    int n = 0;
    double **c = NULL;
    if (vrp_load_tsplib_euc2d_matrix(path, &n, &c) != 0) {
        return 1;
    }

    // printf("Loaded instance %s, n = %d, K = %d, m = %d, T = %d, conv = %d, eps = %e\n", path, n, K, m, T, convergence_iter, epsilon);

    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.5;
    double tau0 = 1.0;
    double Q = 100.0;

    Solution *best = solution_create(K, n);
    double best_cost = 0.0;

    if (aco_vrp_cuda(n, K, m, T, c, alpha, beta, rho, tau0, Q, seed, false, convergence_iter, epsilon, best, &best_cost) != 0) {
        fprintf(stderr, "CUDA solver failed\n");
        solution_free(best);
        matrix_free(c);
        return 1;
    }

    for (int i = 0; i < K; ++i) {
        printf("Route %d: ", i + 1);
        Route *r = &best->routes[i];
        bool first = true;
        for (int t = 0; t < r->len; ++t) {
            int node = r->nodes[t];
            if (node != 0) { 
                if (!first) printf(" ");
                printf("%d", node);
                first = false;
            }
        }
        printf("\n");
    }
    printf("Cost: %.3f\n", best_cost);

    solution_free(best);
    matrix_free(c);
    return 0;
}
