#include "aco.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*
 * Function:  clamp_int
 * --------------------
 * clamps an integer value within an inclusive [lo, hi] interval.
 *
 *  x: value to clamp
 *  lo: lower bound
 *  hi: upper bound
 *
 *  returns: clamped value
 */
static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/*
 * Function:  choose_auto_total_ants
 * ---------------------------------
 * selects an automatic ant count based on problem size and available workers.
 *
 *  n: number of customers
 *
 *  returns: total ants to use per iteration
 */
static int choose_auto_total_ants(int n) {
  int workers = 1;
#ifdef _OPENMP
  workers = omp_get_max_threads();
#endif
  if (workers < 1) {
    workers = 1;
  }

  int ants_per_worker;
  if (n <= 2000) {
    ants_per_worker = 4;
  } else if (n <= 8000) {
    ants_per_worker = 3;
  } else if (n <= 16000) {
    ants_per_worker = 2;
  } else {
    ants_per_worker = 1;
  }

  int total_ants = workers * ants_per_worker;
  total_ants = clamp_int(total_ants, workers, workers * 16);
  total_ants = clamp_int(total_ants, 8, n > 8 ? n : 8);
  return total_ants;
}

/*
 * Function:  aco_vrp
 * ------------------
 * runs the sequential ACO solver for VRP.
 * algorithm outline:
 * 1) initialize heuristic matrix eta and pheromone matrix tau
 * 2) for each iteration, build m ant solutions and keep the best one
 * 3) update global best solution if iteration best improves it
 * 4) evaporate pheromone and deposit on arcs of iteration-best solution
 *
 *  n: number of customers (1..n), with depot at 0
 *  K: number of routes/vehicles
 *  m: ants per iteration
 *  T: number of iterations
 *  c: cost matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  rho: evaporation factor
 *  tau0: initial pheromone value
 *  Q: deposit scale
 *  seed: base random seed
 *  best_solution: output container for best route set
 *  best_cost: output best objective value
 *
 *  returns: nothing; on allocation failure prints an error and returns early
 */
void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost) {
  int total_m = m;
  if (total_m <= 0) {
    total_m = choose_auto_total_ants(n);
  }

  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  AcoScoreCache *score_cache =
      aco_score_cache_create(n, (n + 1 < 8) ? (n + 1) : 8,
                             (n + 1 < 64) ? (n + 1) : 64);

  if (!eta || !tau || !iter_best) {
    fprintf(stderr, "allocation failure in aco_vrp\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
    aco_score_cache_free(score_cache);
    return;
  }

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        eta[i][j] = 0.0;
        tau[i][j] = 0.0;
      } else {
        eta[i][j] = 1.0 / (c[i][j] + ACO_EPS);
        tau[i][j] = tau0;
      }
    }
  }

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  int vehicle_capacity_customers = (K > 0) ? ((n + K - 1) / K) : n;
  if (vehicle_capacity_customers < 1) {
    vehicle_capacity_customers = 1;
  }

  size_t scratch_len = (n > 0) ? (size_t)n : 1u;

  for (int iter = 0; iter < T; ++iter) {
    aco_score_cache_invalidate(score_cache);
    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;
    int iter_failed = 0;

    Solution *sol = solution_create(K, n);
    int *unvisited_nodes = malloc(scratch_len * sizeof(int));
    double *candidate_scores = malloc(scratch_len * sizeof(double));
    double *random_draws = malloc(scratch_len * sizeof(double));

    if (!sol || !unvisited_nodes || !candidate_scores ||
        !random_draws) {
      iter_failed = 1;
    } else {
      for (int ant = 0; ant < total_m; ++ant) {
        unsigned int rng_state = aco_make_ant_seed(seed, iter, ant);

        aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                               vehicle_capacity_customers, score_cache,
                               &rng_state, unvisited_nodes, candidate_scores,
                               random_draws);
        double cost = solution_cost(sol, c);

        if (cost < iter_best_cost ||
            (fabs(cost - iter_best_cost) <= ACO_EPS &&
             ant < iter_best_ant)) {
          iter_best_cost = cost;
          iter_best_ant = ant;
          solution_copy(iter_best, sol);
        }
      }
    }

    free(random_draws);
    free(candidate_scores);
    free(unvisited_nodes);
    solution_free(sol);

    if (iter_failed) {
      fprintf(stderr, "allocation failure during iteration\n");
      solution_free(iter_best);
      matrix_free(eta);
      matrix_free(tau);
      aco_score_cache_free(score_cache);
      return;
    }

    if (iter_best_cost < *best_cost) {
      *best_cost = iter_best_cost;
      solution_copy(best_solution, iter_best);
    }

    if (iter_best_cost < DBL_MAX) {
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i != j) {
            tau[i][j] *= (1.0 - rho);
          }
        }
      }

      double deposit = Q / iter_best_cost;
      for (int i = 0; i < K; ++i) {
        Route *r = &iter_best->routes[i];
        for (int t = 0; t + 1 < r->len; ++t) {
          int u = r->nodes[t];
          int v = r->nodes[t + 1];
          tau[u][v] += deposit;
          tau[v][u] += deposit;
        }
      }
    }
  }

  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
  aco_score_cache_free(score_cache);
}
