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

  /* Allocazione strutture dati thread-local al di fuori del ciclo delle iterazioni */
  Solution **thread_ant_sols = malloc(num_threads * sizeof(Solution *));
  Solution **thread_best_sols = malloc(num_threads * sizeof(Solution *));
  bool **thread_visited = malloc(num_threads * sizeof(bool *));

  int thread_alloc_ok = (thread_ant_sols && thread_best_sols && thread_visited);
  if (thread_alloc_ok) {
    for (int i = 0; i < num_threads; ++i) {
      thread_ant_sols[i] = solution_create(K, n);
      thread_best_sols[i] = solution_create(K, n);
      thread_visited[i] = calloc((size_t)(n + 1), sizeof(bool));
      if (!thread_ant_sols[i] || !thread_best_sols[i] || !thread_visited[i]) {
        thread_alloc_ok = 0;
        break;
      }
    }
  }

  if (!thread_alloc_ok) {
    fprintf(stderr, "allocation failure in openmp workers setup\n");
    if (thread_ant_sols) {
      for (int i = 0; i < num_threads; ++i) if (thread_ant_sols[i]) solution_free(thread_ant_sols[i]);
      free(thread_ant_sols);
    }
    if (thread_best_sols) {
      for (int i = 0; i < num_threads; ++i) if (thread_best_sols[i]) solution_free(thread_best_sols[i]);
      free(thread_best_sols);
    }
    if (thread_visited) {
      for (int i = 0; i < num_threads; ++i) if (thread_visited[i]) free(thread_visited[i]);
      free(thread_visited);
    }
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

#ifdef _OPENMP
/* L'uso di array preallocati permette di accedere ai dati senza conflitti */
#pragma omp parallel num_threads(num_threads) default(none)                     \
    shared(n, K, m, iter, seed, tau, eta, alpha, beta, c, iter_best,            \
           best_solution, best_cost, iter_best_cost, iter_best_ant,             \
           thread_ant_sols, thread_best_sols, thread_visited)
#endif
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      Solution *thread_ant_sol = thread_ant_sols[tid];
      Solution *thread_best_sol = thread_best_sols[tid];
      bool *visited = thread_visited[tid];
      
      double thread_best_cost = DBL_MAX;
      int thread_best_ant = m;

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

    aco_update_pheromones(tau, iter_best, iter_best_cost, n, K, rho, Q);
  }

  /* Deallocazione a fine esecuzione per evitare l'overhead */
  for (int i = 0; i < num_threads; ++i) {
    if (thread_ant_sols[i]) solution_free(thread_ant_sols[i]);
    if (thread_best_sols[i]) solution_free(thread_best_sols[i]);
    if (thread_visited[i]) free(thread_visited[i]);
  }
  free(thread_ant_sols);
  free(thread_best_sols);
  free(thread_visited);

  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
}