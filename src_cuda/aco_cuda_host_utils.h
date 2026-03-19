#ifndef ACO_CUDA_HOST_UTILS_H
#define ACO_CUDA_HOST_UTILS_H

#include "solution.h"

#ifdef __cplusplus
extern "C" {
#endif

double* flatten_matrix(double **matrix, int n);
void deposit_pheromones_host(double *tau_host, const Solution *best_solution, double deposit, int n);

#ifdef __cplusplus
}
#endif

#endif
