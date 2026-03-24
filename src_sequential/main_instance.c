#include "aco.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <instance.vrp> <K> <m> <T> [seed]\n", argv[0]);
        return 1;
    }

    const char *path = argv[1];
    int K = atoi(argv[2]);
    int m = atoi(argv[3]);
    int T = atoi(argv[4]);
    unsigned int seed = (argc > 5) ? (unsigned int)atoi(argv[5]) : 1234;

    int n = 0;
    double **c = NULL;
    if (vrp_load_tsplib_euc2d_matrix(path, &n, &c) != 0) {
        return 1;
    }

    printf("Loaded instance %s, n = %d, K = %d, m = %d, T = %d\n", path, n, K, m, T);

    double alpha = 1.0;
    double beta = 2.0;
    double rho = 0.5;
    double tau0 = 1.0;
    double Q = 100.0;

    Solution *best = solution_create(K, n);
    double best_cost = 0.0;

    aco_vrp_sequential(n, K, m, T, c, alpha, beta, rho, tau0, Q, seed, false, best, &best_cost);

    for (int i = 0; i < K; ++i) {
        printf("Route %d: ", i + 1);
        Route *r = &best->routes[i];
        for (int t = 0; t < r->len; ++t) {
            int node = r->nodes[t];
            if (node != 0) {
                printf("%d ", node);
            }
        }
        printf("\n");
    }
    printf("Cost: %.3f\n", best_cost);

    solution_free(best);
    matrix_free(c);
    return 0;
}
