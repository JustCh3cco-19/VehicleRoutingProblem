#ifndef ACO_COMMON_H
#define ACO_COMMON_H

#include "solution.h"

#include <stdbool.h>

#define ACO_EPS 1e-9

/* Deterministic per-ant seed derivation used across all backends. */
unsigned int aco_ant_seed(unsigned int seed, int iter, int ant);
/* Strict ordering with deterministic tie-break by ant index. */
int aco_better_candidate(double cost_a, int ant_a, double cost_b, int ant_b);

/* Initialize heuristic (eta) and pheromone (tau) matrices. */
void aco_init_eta_tau(int n, double **c, double **eta, double **tau,
                      double tau0);

/* Build one complete VRP solution for one ant. */
void aco_construct_ant_solution(Solution *sol, bool *visited,
                                int n, int K,
                                double **tau, double **eta,
                                double alpha, double beta,
                                unsigned int seed);

/* Evaporate and then deposit pheromone on iteration-best edges. */
void aco_update_pheromones(double **tau, const Solution *iter_best,
                           double iter_best_cost,
                           int n, int K, double rho, double Q);

#endif
