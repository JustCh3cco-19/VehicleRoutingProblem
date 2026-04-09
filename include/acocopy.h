#ifndef ACO_H
#define ACO_H

#include "solution.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Nuovo solver hardware-aware con timeout e early stoppage
int aco_vrp_cuda(int n, int K, double timeout_minutes, double improvement_threshold,
                 double **c, double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost);

#ifdef __cplusplus
}
#endif

#endif // ACO_H
