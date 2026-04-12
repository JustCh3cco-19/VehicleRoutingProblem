#ifndef ACO_V2_H
#define ACO_V2_H

#include "solution.h"

/**
 * @brief Versione 2 del solutore ACO VRP ottimizzata per Strong e Weak Scaling.
 * 
 * Implementa un modello ibrido OpenMP-MPI con:
 * - Sincronizzazione MPI a bassa latenza (singola Allreduce).
 * - Scheduling dinamico OpenMP per bilanciamento stocastico.
 * - Parallelizzazione del deposito dei feromoni tramite atomics.
 */
void aco_vrp_v2(int n, int K, int m, double **c, double alpha,
                double beta, double rho, double tau0, double Q,
                unsigned int seed, Solution *best_solution, double *best_cost);

void aco_vrp_v2_with_capacity(int n, int K, int vehicle_capacity_customers, int m,
                              double **c, double alpha, double beta,
                              double rho, double tau0, double Q,
                              unsigned int seed, Solution *best_solution,
                              double *best_cost);

#endif
