#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>

static void fill_costs(double **c, int n) {
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

static void assert_valid_solution(const Solution *s, int n, int K) {
  char err[128];
  bool ok = solution_validate(s, n, K, err, sizeof(err));
  if (!ok) {
    fprintf(stderr, "invalid CUDA solution: %s\n", err);
    assert(0);
  }
}

int main(void) {
  int n = 6;
  int K = 2;
  int m = 64;
  int T = 100;
  unsigned int seed = 2026;

  double **c = matrix_alloc(n);
  assert(c);
  fill_costs(c, n);

  Solution *best_cuda_a = solution_create(K, n);
  Solution *best_cuda_b = solution_create(K, n);
  assert(best_cuda_a && best_cuda_b);

  double cost_cuda_a = DBL_MAX;
  double cost_cuda_b = DBL_MAX;

  int rc = aco_vrp_cuda(n, K, m, T, c,
                        1.0, 2.0, 0.5, 1.0, 1.0, seed, false, 0,
                        best_cuda_a, &cost_cuda_a);
  assert(rc == 0);
  rc = aco_vrp_cuda(n, K, m, T, c,
                    1.0, 2.0, 0.5, 1.0, 1.0, seed, false, 0,
                    best_cuda_b, &cost_cuda_b);
  assert(rc == 0);
  assert_valid_solution(best_cuda_a, n, K);
  assert_valid_solution(best_cuda_b, n, K);
  assert(fabs(solution_cost(best_cuda_a, c) - cost_cuda_a) < 1e-6);
  assert(fabs(solution_cost(best_cuda_b, c) - cost_cuda_b) < 1e-6);

  /* Same seed should keep result stable across runs. */
  assert(fabs(cost_cuda_a - cost_cuda_b) < 1e-6);

  solution_free(best_cuda_a);
  solution_free(best_cuda_b);
  matrix_free(c);
  printf("CUDA tests passed.\n");
  return 0;
}
