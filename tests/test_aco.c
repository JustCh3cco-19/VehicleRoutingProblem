#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static void assert_close(double got, double expected, double tol) {
  if (fabs(got - expected) > tol) {
    fprintf(stderr, "expected %.6f, got %.6f\n", expected, got);
    assert(0);
  }
}

static void assert_valid_solution(const Solution *s, int n, int K) {
  char err[128];
  bool ok = solution_validate(s, n, K, err, sizeof(err));
  if (!ok) {
    fprintf(stderr, "invalid solution: %s\n", err);
    assert(0);
  }
}

static double route_segment_cost(const int *perm, int start, int end,
                                 double **c) {
  if (start == end) {
    return 0.0;
  }
  double cost = c[0][perm[start]];
  for (int i = start; i + 1 < end; ++i) {
    cost += c[perm[i]][perm[i + 1]];
  }
  cost += c[perm[end - 1]][0];
  return cost;
}

static void eval_splits(const int *perm, int n, int K, int route_idx,
                        int start, double **c, double cost_so_far,
                        double *best) {
  if (route_idx == K) {
    if (start == n && cost_so_far < *best) {
      *best = cost_so_far;
    }
    return;
  }

  for (int end = start; end <= n; ++end) {
    double add = route_segment_cost(perm, start, end, c);
    eval_splits(perm, n, K, route_idx + 1, end, c, cost_so_far + add, best);
  }
}

static void permute(int *perm, int l, int r, int n, int K, double **c,
                    double *best) {
  if (l == r) {
    eval_splits(perm, n, K, 0, 0, c, 0.0, best);
    return;
  }
  for (int i = l; i <= r; ++i) {
    int tmp = perm[l];
    perm[l] = perm[i];
    perm[i] = tmp;
    permute(perm, l + 1, r, n, K, c, best);
    tmp = perm[l];
    perm[l] = perm[i];
    perm[i] = tmp;
  }
}

static double exact_vrp_cost(int n, int K, double **c) {
  int *perm = malloc((size_t)n * sizeof(int));
  if (!perm) {
    return DBL_MAX;
  }
  for (int i = 0; i < n; ++i) {
    perm[i] = i + 1;
  }

  double best = DBL_MAX;
  permute(perm, 0, n - 1, n, K, c, &best);

  free(perm);
  return best;
}

static void test_single_customer(void) {
  int n = 1;
  int K = 1;
  int m = 1;
  int T = 1;

  double **c = matrix_alloc(n);
  assert(c);
  c[0][0] = 0.0;
  c[1][1] = 0.0;
  c[0][1] = 2.0;
  c[1][0] = 3.0;

  Solution *best = solution_create(K, n);
  assert(best);
  double best_cost = 0.0;

  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 1234,
                     best, &best_cost);

  assert_valid_solution(best, n, K);
  assert_close(best_cost, 5.0, 1e-9);
  assert_close(solution_cost(best, c), best_cost, 1e-9);
  assert(best->routes[0].len == 3);
  assert(best->routes[0].nodes[0] == 0);
  assert(best->routes[0].nodes[1] == 1);
  assert(best->routes[0].nodes[2] == 0);

  solution_free(best);
  matrix_free(c);
}

static void test_two_customers_two_vehicles(void) {
  int n = 2;
  int K = 2;
  int m = 1;
  int T = 1;

  double **c = matrix_alloc(n);
  assert(c);
  c[0][0] = 0.0;
  c[1][1] = 0.0;
  c[2][2] = 0.0;
  c[0][1] = 1.0;
  c[1][0] = 1.0;
  c[0][2] = 2.0;
  c[2][0] = 2.0;
  c[1][2] = 10.0;
  c[2][1] = 10.0;

  Solution *best = solution_create(K, n);
  assert(best);
  double best_cost = 0.0;

  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 5678,
                     best, &best_cost);

  assert_valid_solution(best, n, K);
  assert_close(best_cost, 6.0, 1e-9);
  assert_close(solution_cost(best, c), best_cost, 1e-9);

  solution_free(best);
  matrix_free(c);
}

static void test_solution_validation(void) {
  int n = 2;
  int K = 1;

  Solution *s = solution_create(K, n);
  assert(s);
  route_append(&s->routes[0], 0);
  route_append(&s->routes[0], 1);
  route_append(&s->routes[0], 1);
  route_append(&s->routes[0], 0);

  char err[128];
  bool ok = solution_validate(s, n, K, err, sizeof(err));
  assert(!ok);

  solution_free(s);
}

static void test_exact_solver_known(void) {
  int n = 3;
  int K = 1;

  double **c = matrix_alloc(n);
  assert(c);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      c[i][j] = 0.0;
    }
  }
  c[0][1] = 1.0;
  c[1][0] = 1.0;
  c[0][2] = 5.0;
  c[2][0] = 5.0;
  c[0][3] = 4.0;
  c[3][0] = 4.0;
  c[1][2] = 1.0;
  c[2][1] = 1.0;
  c[1][3] = 10.0;
  c[3][1] = 10.0;
  c[2][3] = 1.0;
  c[3][2] = 1.0;

  double opt = exact_vrp_cost(n, K, c);
  assert_close(opt, 7.0, 1e-9);

  matrix_free(c);
}

static void test_aco_vs_exact_small(void) {
  int n = 4;
  int K = 2;
  int m = 20;
  int T = 40;

  double **c = matrix_alloc(n);
  assert(c);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        c[i][j] = 1.0 + fabs((double)i - (double)j);
      }
    }
  }

  double opt = exact_vrp_cost(n, K, c);

  Solution *best = solution_create(K, n);
  assert(best);
  double best_cost = 0.0;

  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 42,
                     best, &best_cost);

  assert_valid_solution(best, n, K);
  assert_close(solution_cost(best, c), best_cost, 1e-9);
  assert(best_cost + 1e-9 >= opt);

  solution_free(best);
  matrix_free(c);
}

static void test_openmp_matches_sequential(void) {
  int n = 5;
  int K = 2;
  int m = 24;
  int T = 60;

  double **c = matrix_alloc(n);
  assert(c);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        c[i][j] = 1.0 + fabs((double)i - (double)j);
      }
    }
  }

  Solution *best_seq = solution_create(K, n);
  Solution *best_omp = solution_create(K, n);
  assert(best_seq && best_omp);

  double cost_seq = 0.0;
  double cost_omp = 0.0;

  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 2026,
                     best_seq, &cost_seq);
  aco_vrp_openmp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 2026,
                 4, best_omp, &cost_omp);

  assert_valid_solution(best_seq, n, K);
  assert_valid_solution(best_omp, n, K);
  assert_close(cost_seq, cost_omp, 1e-9);
  assert_close(solution_cost(best_omp, c), cost_omp, 1e-9);

  solution_free(best_seq);
  solution_free(best_omp);
  matrix_free(c);
}

static void test_openmp_consistent_across_threads(void) {
  int n = 6;
  int K = 2;
  int m = 40;
  int T = 70;

  double **c = matrix_alloc(n);
  assert(c);
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        c[i][j] = 1.0 + fabs((double)i - (double)j);
      }
    }
  }

  Solution *best_omp1 = solution_create(K, n);
  Solution *best_omp4 = solution_create(K, n);
  assert(best_omp1 && best_omp4);

  double cost_omp1 = 0.0;
  double cost_omp4 = 0.0;

  aco_vrp_openmp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 1337,
                 1, best_omp1, &cost_omp1);
  aco_vrp_openmp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 1337,
                 4, best_omp4, &cost_omp4);

  assert_valid_solution(best_omp1, n, K);
  assert_valid_solution(best_omp4, n, K);
  assert_close(cost_omp1, cost_omp4, 1e-9);
  assert_close(solution_cost(best_omp1, c), cost_omp1, 1e-9);
  assert_close(solution_cost(best_omp4, c), cost_omp4, 1e-9);

  solution_free(best_omp1);
  solution_free(best_omp4);
  matrix_free(c);
}

int main(void) {
  test_single_customer();
  test_two_customers_two_vehicles();
  test_solution_validation();
  test_exact_solver_known();
  test_aco_vs_exact_small();
  test_openmp_matches_sequential();
  test_openmp_consistent_across_threads();
  printf("All tests passed.\n");
  return 0;
}
