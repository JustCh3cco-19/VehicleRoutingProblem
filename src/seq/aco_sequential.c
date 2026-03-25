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
 * Function:  choose_stagnation_level
 * ----------------------------------
 * maps the global stagnation counter to a discrete level:
 * 0 = low, 1 = medium, 2 = high.
 *
 *  stagnation_iters: consecutive iterations without global-best improvement
 *  stagnation_trigger: base trigger used for medium/high transitions
 *
 *  returns: stagnation level in [0, 2]
 */
static int choose_stagnation_level(int stagnation_iters,
                                   int stagnation_trigger) {
  if (stagnation_iters >= 2 * stagnation_trigger) {
    return 2;
  }
  if (stagnation_iters >= stagnation_trigger) {
    return 1;
  }
  return 0;
}

/*
 * Function:  choose_ant_batch_size
 * --------------------------------
 * picks a batch size for adaptive in-iteration ant evaluation. Higher
 * stagnation uses smaller batches to check marginal gains more frequently.
 *
 *  total_ants: upper bound ants for this iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: ants processed per batch (>= 1)
 */
static int choose_ant_batch_size(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 0;
  }

  int divisor = 8;
  if (stagnation_level <= 0) {
    divisor = 6;
  } else if (stagnation_level >= 2) {
    divisor = 12;
  }

  int batch = total_ants / divisor;
  return clamp_int(batch, 1, total_ants);
}

/*
 * Function:  choose_min_ants_before_stop
 * --------------------------------------
 * picks the minimum ants to evaluate before allowing early stop. Higher
 * stagnation enforces deeper per-iteration search.
 *
 *  total_ants: upper bound ants for this iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: minimum ants to evaluate (>= 1, <= total_ants)
 */
static int choose_min_ants_before_stop(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 0;
  }

  int min_ants = total_ants / 4;
  if (stagnation_level == 1) {
    min_ants = total_ants / 2;
  } else if (stagnation_level >= 2) {
    min_ants = (3 * total_ants) / 4;
  }
  return clamp_int(min_ants, 1, total_ants);
}

/*
 * Function:  choose_no_improve_patience
 * -------------------------------------
 * picks how many consecutive non-improving batches are tolerated before
 * stopping the current iteration early. Higher stagnation increases patience.
 *
 *  total_ants: upper bound ants for this iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: patience in number of batches (>= 1)
 */
static int choose_no_improve_patience(int total_ants, int stagnation_level) {
  if (total_ants <= 0) {
    return 1;
  }

  int patience = (total_ants >= 128) ? 3 : 2;
  if (stagnation_level == 1) {
    ++patience;
  } else if (stagnation_level >= 2) {
    patience += 2;
  }
  return clamp_int(patience, 1, 6);
}

/*
 * Function:  aco_vrp
 * ------------------
 * runs the sequential ACO solver for VRP.
 * algorithm outline:
 * 1) initialize heuristic matrix eta and pheromone matrix tau
 * 2) for each iteration, build up to m ant solutions in adaptive batches and
 *    keep the best one
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

  int stagnation_iters = 0;
  int stagnation_trigger = T / 4;
  if (stagnation_trigger < 4) {
    stagnation_trigger = 4;
  }

  size_t scratch_len = (n > 0) ? (size_t)n : 1u;

  for (int iter = 0; iter < T; ++iter) {
    int stagnation_level =
        choose_stagnation_level(stagnation_iters, stagnation_trigger);
    int batch_size = choose_ant_batch_size(total_m, stagnation_level);
    int min_ants_before_stop =
        choose_min_ants_before_stop(total_m, stagnation_level);
    int no_improve_patience =
        choose_no_improve_patience(total_m, stagnation_level);

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
      int ants_evaluated = 0;
      int consecutive_no_improve_batches = 0;

      for (int batch_start = 0; batch_start < total_m;
           batch_start += batch_size) {
        int batch_end = batch_start + batch_size;
        if (batch_end > total_m) {
          batch_end = total_m;
        }

        double prev_best_cost = iter_best_cost;
        int prev_best_ant = iter_best_ant;

        for (int ant = batch_start; ant < batch_end; ++ant) {
          unsigned int rng_state = aco_make_ant_seed(seed, iter, ant);

          aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                                 vehicle_capacity_customers, score_cache,
                                 &rng_state, unvisited_nodes,
                                 candidate_scores, random_draws);
          double cost = solution_cost(sol, c);

          if (cost < iter_best_cost ||
              (fabs(cost - iter_best_cost) <= ACO_EPS &&
               ant < iter_best_ant)) {
            iter_best_cost = cost;
            iter_best_ant = ant;
            solution_copy(iter_best, sol);
          }
        }

        ants_evaluated += (batch_end - batch_start);
        if (iter_best_cost < prev_best_cost - ACO_EPS ||
            (fabs(iter_best_cost - prev_best_cost) <= ACO_EPS &&
             iter_best_ant < prev_best_ant)) {
          consecutive_no_improve_batches = 0;
        } else {
          ++consecutive_no_improve_batches;
        }

        if (ants_evaluated >= min_ants_before_stop &&
            consecutive_no_improve_batches >= no_improve_patience) {
          break;
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
      stagnation_iters = 0;
    } else {
      ++stagnation_iters;
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
