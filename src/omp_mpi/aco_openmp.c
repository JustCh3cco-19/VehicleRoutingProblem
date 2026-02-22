#include "aco.h"

#include "matrix.h"

#include "aco_common.h"

#include <float.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void aco_vrp_openmp(int n, int K, int m, int T, double **c,
                    double alpha, double beta, double rho,
                    double tau0, double Q, unsigned int seed,
                    int num_threads,
                    Solution *best_solution, double *best_cost) {
  if (num_threads <= 0) {
#ifdef _OPENMP
    num_threads = omp_get_max_threads();
#else
    num_threads = 1;
#endif
  }

  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  if (!eta || !tau || !iter_best) {
    fprintf(stderr, "allocation failure in aco_vrp_openmp\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
    return;
  }

  aco_init_eta_tau(n, c, eta, tau, tau0);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  for (int iter = 0; iter < T; ++iter) {
    double iter_best_cost = DBL_MAX;
    int iter_best_ant = m;
    int worker_failed = 0;

#ifdef _OPENMP
/* Force explicit data-sharing clauses to reduce accidental races. */
#pragma omp parallel num_threads(num_threads) default(none)                     \
    shared(n, K, m, iter, seed, tau, eta, alpha, beta, c, iter_best,           \
           best_solution, best_cost, iter_best_cost, iter_best_ant, stderr)     \
    reduction(max:worker_failed)
#endif
    {
      /* Thread-local storage removes false sharing on ant construction. */
      Solution *thread_best_sol = solution_create(K, n);
      Solution *thread_ant_sol = solution_create(K, n);
      bool *visited = calloc((size_t)(n + 1), sizeof(bool));
      double thread_best_cost = DBL_MAX;
      int thread_best_ant = m;

      if (!thread_best_sol || !thread_ant_sol || !visited) {
        worker_failed = 1;
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          fprintf(stderr, "allocation failure in openmp worker\n");
        }
      } else {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int ant = 0; ant < m; ++ant) {
          unsigned int local_seed = aco_ant_seed(seed, iter, ant);
          aco_construct_ant_solution(thread_ant_sol, visited, n, K,
                                     tau, eta, alpha, beta, local_seed);
          double cost = solution_cost(thread_ant_sol, c);

          if (aco_better_candidate(cost, ant, thread_best_cost,
                                   thread_best_ant)) {
            thread_best_cost = cost;
            thread_best_ant = ant;
            solution_copy(thread_best_sol, thread_ant_sol);
          }
        }

#ifdef _OPENMP
/* Critical section only protects global-best merge. */
#pragma omp critical
#endif
        {
          if (aco_better_candidate(thread_best_cost, thread_best_ant,
                                   iter_best_cost, iter_best_ant)) {
            iter_best_cost = thread_best_cost;
            iter_best_ant = thread_best_ant;
            solution_copy(iter_best, thread_best_sol);
          }
          if (aco_better_candidate(thread_best_cost, thread_best_ant,
                                   *best_cost, m + 1)) {
            *best_cost = thread_best_cost;
            solution_copy(best_solution, thread_best_sol);
          }
        }
      }

      free(visited);
      solution_free(thread_ant_sol);
      solution_free(thread_best_sol);
    }

    /* Reduction above turns any worker allocation failure into a global fail. */
    if (worker_failed) {
      solution_free(iter_best);
      matrix_free(eta);
      matrix_free(tau);
      return;
    }

    aco_update_pheromones(tau, iter_best, iter_best_cost, n, K, rho, Q);
  }

  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
}
