#include "aco.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-9

static double rand01(void) {
  return (double)rand() / (double)RAND_MAX;
}

static int select_next(int current, const bool *visited, int n,
                       double **tau, double **eta,
                       double alpha, double beta) {
  double denom = 0.0;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      denom += score;
    }
  }
  if (denom <= 0.0) {
    for (int j = 1; j <= n; ++j) {
      if (!visited[j]) return j;
    }
    return 0;
  }

  double r = rand01();
  double cumulative = 0.0;
  int last = 1;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      double p = score / denom;
      cumulative += p;
      last = j;
      if (cumulative >= r) {
        return j;
      }
    }
  }
  return last;
}

void aco_vrp_sequential(int n, int K, int m, int T, double **c,
                        double alpha, double beta, double rho,
                        double tau0, double Q, unsigned int seed,
                        Solution *best_solution, double *best_cost) {
  srand(seed);

  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);
  if (!eta || !tau || !iter_best) {
    fprintf(stderr, "allocation failure in aco_vrp_sequential\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
    return;
  }

  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        eta[i][j] = 0.0;
        tau[i][j] = 0.0;
      } else {
        eta[i][j] = 1.0 / (c[i][j] + EPS);
        tau[i][j] = tau0;
      }
    }
  }

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  for (int iter = 0; iter < T; ++iter) {
    double iter_best_cost = DBL_MAX;

    for (int ant = 0; ant < m; ++ant) {
      Solution *sol = solution_create(K, n);
      bool *visited = calloc((size_t)(n + 1), sizeof(bool));
      if (!sol || !visited) {
        fprintf(stderr, "allocation failure during iteration\n");
        free(visited);
        solution_free(sol);
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
        return;
      }
      int unvisited_count = n;

      for (int vehicle = 1; vehicle <= K; ++vehicle) {
        Route *r = &sol->routes[vehicle - 1];
        route_append(r, 0);
        int current = 0;

        while (unvisited_count > 0 && unvisited_count > (K - vehicle)) {
          int next = select_next(current, visited, n, tau, eta, alpha, beta);
          route_append(r, next);
          visited[next] = true;
          --unvisited_count;
          current = next;
        }

        route_append(r, 0);
      }

      if (unvisited_count > 0) {
        Route *last = &sol->routes[K - 1];
        int end_idx = last->len - 1;
        for (int j = 1; j <= n; ++j) {
          if (!visited[j]) {
            last->nodes[end_idx++] = j;
            visited[j] = true;
          }
        }
        last->nodes[end_idx++] = 0;
        last->len = end_idx;
        unvisited_count = 0;
      }

      double cost = solution_cost(sol, c);

      if (cost < iter_best_cost) {
        iter_best_cost = cost;
        solution_copy(iter_best, sol);
      }
      if (cost < *best_cost) {
        *best_cost = cost;
        solution_copy(best_solution, sol);
      }

      free(visited);
      solution_free(sol);
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
}
