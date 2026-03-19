#include "aco_cuda_host_utils.h"
#include <stdlib.h>

double* flatten_matrix(double **matrix, int n) {
    int size = n + 1;
    double *flat = (double *)malloc(size * size * sizeof(double));
    if (!flat) return NULL;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flat[i * size + j] = matrix[i][j];
        }
    }
    return flat;
}

void deposit_pheromones_host(double *tau_host, const Solution *best_solution, double deposit, int n) {
    int size = n + 1;
    for (int i = 0; i < best_solution->K; ++i) {
        const Route *r = &best_solution->routes[i];
        for (int t = 0; t + 1 < r->len; ++t) {
            int u = r->nodes[t];
            int v = r->nodes[t + 1];
            tau_host[u * size + v] += deposit;
            tau_host[v * size + u] += deposit;
        }
    }
}
