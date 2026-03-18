#ifndef ACO_H
#define ACO_H

#include "solution.h"

#define ACO_EPS 1e-9

/*
 * Function:  aco_vrp
 * ------------------
 * runs Ant Colony Optimization for the Vehicle Routing Problem and writes the
 * best solution found.
 *
 *  n: number of customers (customer ids are 1..n, depot is 0)
 *  K: number of vehicles/routes
 *  m: number of ants per iteration
 *  T: number of iterations
 *  c: cost matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  rho: pheromone evaporation rate
 *  tau0: initial pheromone value
 *  Q: pheromone deposit scaling factor
 *  seed: base random seed
 *  best_solution: output best solution container
 *  best_cost: output best cost
 *
 *  returns: nothing; on allocation/synchronization errors the function returns
 *           early and leaves partial progress only
 */
void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost);

/*
 * Function:  aco_rand01_state
 * ---------------------------
 * generates a random value in [0, 1] using either a local xorshift state or
 * the global rand() fallback when state is NULL.
 *
 *  state: optional pointer to per-thread/per-ant rng state
 *
 *  returns: pseudo-random double in [0, 1]
 */
double aco_rand01_state(unsigned int *state);

/*
 * Function:  aco_make_ant_seed
 * ----------------------------
 * derives a deterministic seed from base seed, iteration index, and ant index.
 *
 *  base_seed: user-provided seed
 *  iter: iteration index
 *  ant_index: ant index (local or global, depending on caller)
 *
 *  returns: non-zero mixed seed value
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter,
                               int ant_index);

/*
 * Function:  aco_select_next
 * --------------------------
 * selects the next customer from the unvisited list using roulette-wheel
 * sampling over tau^alpha * eta^beta scores.
 *
 *  current: current node id
 *  unvisited_nodes: array of unvisited customer ids
 *  unvisited_count: number of valid entries in unvisited_nodes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  roulette_r: random value in [0, 1] used for roulette threshold
 *  candidate_scores: scratch array sized at least unvisited_count
 *  selected_index: optional output index inside unvisited_nodes
 *
 *  returns: selected customer id on success
 *           0 when no unvisited nodes are available
 */
int aco_select_next(int current, const int *unvisited_nodes,
                    int unvisited_count, double **tau, double **eta,
                    double alpha, double beta, double roulette_r,
                    double *candidate_scores, int *selected_index);

/*
 * Function:  aco_build_ant_solution
 * ---------------------------------
 * constructs one complete ant solution by repeatedly selecting customers from
 * an in-place unvisited list and assigning them to routes.
 *
 *  sol: solution output container for the current ant
 *  n: number of customers
 *  K: number of routes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone influence exponent
 *  beta: heuristic influence exponent
 *  rng_state: rng state for deterministic random draws
 *  unvisited_nodes: scratch array of size at least n
 *  candidate_scores: scratch array of size at least n
 *  random_draws: scratch array of size at least n
 *
 *  returns: nothing
 */
void aco_build_ant_solution(Solution *sol, int n, int K, double **tau,
                            double **eta, double alpha, double beta,
                            unsigned int *rng_state, int *unvisited_nodes,
                            double *candidate_scores, double *random_draws);

#endif
