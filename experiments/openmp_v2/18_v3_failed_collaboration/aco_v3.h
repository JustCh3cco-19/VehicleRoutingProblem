#ifndef ACO_V3_H
#define ACO_V3_H

#include "solution.h"

/**
 * @brief Versione 3 del solutore ACO VRP con Collaborative Ant Teams.
 * 
 * I thread OpenMP sono divisi in team che collaborano alla costruzione
 * di una singola formica, parallelizzando la scansione del Fallback
 * per massimizzare la banda di memoria e il prefetching.
 */
void aco_vrp_v3(int n, int k, int m, double **c, double alpha,
                double beta, double rho, double tau0, double q,
                unsigned int seed, t_solution *best_solution, double *best_cost);

void aco_vrp_v3_with_capacity(int n, int k, int vehicle_capacity_customers, int m,
                              double **c, double alpha, double beta,
                              double rho, double tau0, double q,
                              unsigned int seed, t_solution *best_solution,
                              double *best_cost);

#endif
