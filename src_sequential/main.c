#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_example_costs(double **c, int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        c[i][j] = 1.0 + fabs((double)i - (double)j);
      }
    }
  }
}

int main(void) {
  int n = 5;
  int K = 2;
  int m = 10;
  int T = 50;

  double alpha = 1.0;
  double beta = 2.0;
  double rho = 0.5;
  double tau0 = 1.0;
  double Q = 1.0;

  double **c = matrix_alloc(n);
  if (!c) {
    fprintf(stderr, "failed to allocate cost matrix\n");
    return 1;
  }
  fill_example_costs(c, n);

  Solution *best = solution_create(K, n);
  if (!best) {
    fprintf(stderr, "failed to allocate solution\n");
    matrix_free(c);
    return 1;
  }

  double best_cost = 0.0;
  aco_vrp_sequential(n, K, m, T, c, alpha, beta, rho, tau0, Q, 1234,
                     false, best, &best_cost);

  printf("best cost: %.3f\n", best_cost);

  solution_free(best);
  matrix_free(c);
  return 0;
}
