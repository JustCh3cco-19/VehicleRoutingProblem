#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Function:  assert_close
 * -----------------------
 * asserts that two doubles are within a tolerance; prints diagnostic on
 * failure.
 *
 *  got: computed value
 *  expected: reference value
 *  tol: absolute tolerance
 *
 *  returns: nothing; aborts via assert on failure
 */
static void assert_close(double got, double expected, double tol) {
  if (fabs(got - expected) > tol) {
    fprintf(stderr, "expected %.6f, got %.6f\n", expected, got);
    assert(0);
  }
}

/*
 * Function:  assert_valid_solution
 * --------------------------------
 * validates a solution and aborts with diagnostic message if invalid.
 *
 *  s: solution to check
 *  n: number of customers
 *  K: number of routes
 *
 *  returns: nothing; aborts via assert on validation failure
 */
static void assert_valid_solution(const Solution *s, int n, int K) {
  char err[128];
  bool ok = solution_validate(s, n, K, err, sizeof(err));
  if (!ok) {
    fprintf(stderr, "invalid solution: %s\n", err);
    assert(0);
  }
}

/*
 * Function:  route_segment_cost
 * -----------------------------
 * computes cost of one route segment defined by perm[start:end], including
 * depot departure and return.
 *
 *  perm: customer permutation
 *  start: first index in segment (inclusive)
 *  end: one-past-last index in segment (exclusive)
 *  c: cost matrix
 *
 *  returns: segment cost (0 when start == end)
 */
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

/*
 * Function:  eval_splits
 * ----------------------
 * recursively evaluates all ways to split a fixed customer order into K
 * routes, tracking the minimum total cost.
 *
 *  perm: customer permutation
 *  n: number of customers
 *  K: number of routes
 *  route_idx: current route index in recursion
 *  start: current start index in perm for this route
 *  c: cost matrix
 *  cost_so_far: accumulated cost before current recursion level
 *  best: pointer to global best cost accumulator
 *
 *  returns: nothing; updates *best in place
 */
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

/*
 * Function:  permute
 * ------------------
 * recursively enumerates all permutations of perm[l..r] and evaluates each via
 * eval_splits.
 *
 *  perm: mutable permutation array
 *  l: left index of active permutation window
 *  r: right index of active permutation window
 *  n: number of customers
 *  K: number of routes
 *  c: cost matrix
 *  best: pointer to global best cost accumulator
 *
 *  returns: nothing; updates *best in place
 */
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

/*
 * Function:  exact_vrp_cost
 * -------------------------
 * computes an exact optimum for small instances by full permutation + split
 * enumeration.
 *
 *  n: number of customers
 *  K: number of routes
 *  c: cost matrix
 *
 *  returns: optimal cost for the instance
 *           DBL_MAX on allocation failure
 */
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

/*
 * Function:  test_single_customer
 * -------------------------------
 * verifies correctness on a 1-customer, 1-vehicle instance with known cost.
 *
 *  returns: nothing; aborts via assert on failure
 */
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

  aco_vrp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 1234, best, &best_cost);

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

/*
 * Function:  test_two_customers_two_vehicles
 * ------------------------------------------
 * verifies route construction and objective value on a small handcrafted
 * 2-customer, 2-vehicle instance.
 *
 *  returns: nothing; aborts via assert on failure
 */
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

  aco_vrp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 5678, best, &best_cost);

  assert_valid_solution(best, n, K);
  assert_close(best_cost, 6.0, 1e-9);
  assert_close(solution_cost(best, c), best_cost, 1e-9);

  solution_free(best);
  matrix_free(c);
}

/*
 * Function:  test_solution_validation
 * -----------------------------------
 * checks that solution_validate rejects a solution where one customer is
 * visited multiple times.
 *
 *  returns: nothing; aborts via assert on failure
 */
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

/*
 * Function:  test_exact_solver_known
 * ----------------------------------
 * checks the brute-force exact solver on a known tiny instance with known
 * optimum.
 *
 *  returns: nothing; aborts via assert on failure
 */
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
  c[0][1] = 1.0; c[1][0] = 1.0;
  c[0][2] = 5.0; c[2][0] = 5.0;
  c[0][3] = 4.0; c[3][0] = 4.0;
  c[1][2] = 1.0; c[2][1] = 1.0;
  c[1][3] = 10.0; c[3][1] = 10.0;
  c[2][3] = 1.0; c[3][2] = 1.0;

  double opt = exact_vrp_cost(n, K, c);
  assert_close(opt, 7.0, 1e-9);

  matrix_free(c);
}

/*
 * Function:  test_aco_vs_exact_small
 * ----------------------------------
 * compares ACO output against exact optimum on a small instance to ensure ACO
 * never reports a cost better than the true optimum and returns a valid
 * solution.
 *
 *  returns: nothing; aborts via assert on failure
 */
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

  aco_vrp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 42, best, &best_cost);

  assert_valid_solution(best, n, K);
  assert_close(solution_cost(best, c), best_cost, 1e-9);
  assert(best_cost + 1e-9 >= opt);

  solution_free(best);
  matrix_free(c);
}

/*
 * Function:  main
 * --------------
 * runs all sequential unit tests.
 *
 *  returns: 0 when all tests pass
 *           non-zero if any assert triggers
 */
int main(void) {
  test_single_customer();
  test_two_customers_two_vehicles();
  test_solution_validation();
  test_exact_solver_known();
  test_aco_vs_exact_small();
  printf("All tests passed.\n");
  return 0;
}
