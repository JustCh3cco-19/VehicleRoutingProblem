#ifndef ACO_H
#define ACO_H

#include "solution.h"

#define ACO_EPS 1e-9

typedef enum {
  ACO_OK = 0,
  ACO_ERR_INVALID_INPUT = 1,
  ACO_ERR_ALLOCATION = 2,
  ACO_ERR_NO_SOLUTION = 3,
  ACO_ERR_BACKEND = 4
} AcoStatus;

const char *aco_status_string(AcoStatus status);

/**
 * @brief Runs the ACO solver with an auto-derived per-vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param m Number of ants (0 enables backend default tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp(int n, int K, int m, double **c, double alpha, double beta,
                  double rho, double tau0, double Q, unsigned int seed,
                  Solution *best_solution, double *best_cost);

/**
 * @brief Runs the ACO solver with an explicit per-vehicle customer capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param vehicle_capacity_customers Max customers served by each vehicle.
 * @param m Number of ants (0 enables backend default tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers,
                                int m, double **c, double alpha, double beta,
                                double rho, double tau0, double Q,
                                unsigned int seed, Solution *best_solution,
                                double *best_cost);

/**
 * @brief Returns a uniform random value in [0, 1) using the provided RNG state.
 * @param state Mutable RNG state.
 * @return Random value in [0, 1).
 */
double aco_rand01_state(unsigned int *state);

/**
 * @brief Builds a deterministic per-ant seed from base seed, iteration and ant id.
 * @param base_seed Global base seed.
 * @param iter Iteration index.
 * @param ant_index Ant index.
 * @return Derived seed for the ant.
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter, int ant_index);

#endif
