#ifndef ACO_H
#define ACO_H

#include "solution.h"

void aco_vrp_sequential(int n, int K, int m, int T, double **c,
                        double alpha, double beta, double rho,
                        double tau0, double Q, unsigned int seed,
                        Solution *best_solution, double *best_cost);

#endif
