#include "aco.h"

#include "matrix.h"

#include "aco_common.h"

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void aco_vrp_sequential(int n, int K, int m, int T, double **c,
                        double alpha, double beta, double rho,
                        double tau0, double Q, unsigned int seed,
                        Solution *best_solution, double *best_cost) {
  /* Allocate once and reuse inside the full iteration loop. */
  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  Solution *ant_sol = solution_create(K, n);
  bool *visited = calloc((size_t)(n + 1), sizeof(bool));

  if (!eta || !tau || !iter_best || !ant_sol || !visited) {
    fprintf(stderr, "allocation failure in aco_vrp_sequential\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
    solution_free(ant_sol);
    free(visited);
    return;
  }

  aco_init_eta_tau(n, c, eta, tau, tau0);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  for (int iter = 0; iter < T; ++iter) {
    double iter_best_cost = DBL_MAX;
    int iter_best_ant = m;

    for (int ant = 0; ant < m; ++ant) {
      /* Seed is deterministic for reproducible results. */
      unsigned int local_seed = aco_ant_seed(seed, iter, ant);
      aco_construct_ant_solution(ant_sol, visited, n, K, tau, eta,
                                 alpha, beta, local_seed);

      double cost = solution_cost(ant_sol, c);
      if (aco_better_candidate(cost, ant, iter_best_cost, iter_best_ant)) {
        iter_best_cost = cost;
        iter_best_ant = ant;
        solution_copy(iter_best, ant_sol);
      }
      if (aco_better_candidate(cost, ant, *best_cost, m + 1)) {
        *best_cost = cost;
        solution_copy(best_solution, ant_sol);
      }
    }

    /* Keep pheromone update centralized after all ants of this iteration. */
    aco_update_pheromones(tau, iter_best, iter_best_cost, n, K, rho, Q);
  }

  free(visited);
  solution_free(ant_sol);
  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
}
