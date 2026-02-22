#ifndef ACO_H
#define ACO_H

#include "solution.h"

/*
 * Sequential baseline: single process, single thread.
 */
void aco_vrp_sequential(int n, int K, int m, int T, double **c,
                        double alpha, double beta, double rho,
                        double tau0, double Q, unsigned int seed,
                        Solution *best_solution, double *best_cost);

/*
 * Shared-memory version:
 * - ant construction is parallelized with OpenMP
 * - pheromone update is performed once per iteration on host.
 */
void aco_vrp_openmp(int n, int K, int m, int T, double **c,
                    double alpha, double beta, double rho,
                    double tau0, double Q, unsigned int seed,
                    int num_threads,
                    Solution *best_solution, double *best_cost);

/*
 * Distributed-memory version:
 * - ants are split across MPI ranks
 * - each rank may internally use OpenMP threads
 * - pheromone matrices are synchronized every sync_every iterations.
 *
 * Returns 0 on success, non-zero on failure.
 */
int aco_vrp_mpi_openmp(int n, int K, int m, int T, double **c,
                       double alpha, double beta, double rho,
                       double tau0, double Q, unsigned int seed,
                       int num_threads, int sync_every,
                       Solution *best_solution, double *best_cost);

/*
 * CUDA version:
 * - ant construction and evaporation run on device
 * - best-ant selection and pheromone deposit run on host.
 *
 * Returns 0 on success, non-zero on failure.
 */
int aco_vrp_cuda(int n, int K, int m, int T, double **c,
                 double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost);

#endif
