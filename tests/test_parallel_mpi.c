#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>

static void assert_close(double a, double b, double tol) {
  if (fabs(a - b) > tol) {
    fprintf(stderr, "assert_close failed: %.12f vs %.12f\n", a, b);
    assert(0);
  }
}

int main(void) {
  int provided = 0;
  int rc = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  assert(rc == MPI_SUCCESS);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = 1;
  int K = 1;
  int m = 6;
  int T = 8;

  double **c = matrix_alloc(n);
  assert(c);
  c[0][0] = 0.0;
  c[1][1] = 0.0;
  c[0][1] = 2.0;
  c[1][0] = 3.0;

  Solution *best = solution_create(K, n);
  assert(best);

  double best_cost = 0.0;
  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 1234, best,
                     &best_cost);

  char err[128];
  bool ok = solution_validate(best, n, K, err, sizeof(err));
  if (!ok) {
    fprintf(stderr, "rank %d invalid solution: %s\n", rank, err);
    assert(0);
  }

  double recomputed = solution_cost(best, c);
  assert_close(best_cost, recomputed, 1e-9);
  assert_close(best_cost, 5.0, 1e-9);

  double min_cost = 0.0;
  double max_cost = 0.0;
  MPI_Allreduce(&best_cost, &min_cost, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&best_cost, &max_cost, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  assert_close(min_cost, max_cost, 1e-9);

  solution_free(best);
  matrix_free(c);

  if (rank == 0) {
    printf("MPI/OpenMP test passed.\n");
  }

  MPI_Finalize();
  return 0;
}
