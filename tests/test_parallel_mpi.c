#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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
    fprintf(stderr, "invalid MPI solution: %s\n", err);
    assert(0);
  }
}

int main(int argc, char **argv) {
  int rc = MPI_Init(&argc, &argv);
  assert(rc == MPI_SUCCESS);

  int rank = 0;
  rc = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(rc == MPI_SUCCESS);

  int n = 6;
  int K = 2;
  int m = 24;
  int T = 60;

  double **c = matrix_alloc(n);
  assert(c);
  fill_costs(c, n);

  Solution *best_seq = solution_create(K, n);
  Solution *best_mpi_s1 = solution_create(K, n);
  Solution *best_mpi_s4 = solution_create(K, n);
  assert(best_seq && best_mpi_s1 && best_mpi_s4);

  double cost_seq = DBL_MAX;
  double cost_mpi_s1 = DBL_MAX;
  double cost_mpi_s4 = DBL_MAX;

  aco_vrp_sequential(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 2026, false,
                     best_seq, &cost_seq);
  assert_valid_solution(best_seq, n, K);

  rc = aco_vrp_mpi_openmp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 2026,
                          2, 1, best_mpi_s1, &cost_mpi_s1);
  assert(rc == 0);
  assert_valid_solution(best_mpi_s1, n, K);

  rc = aco_vrp_mpi_openmp(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0, 2026,
                          2, 4, best_mpi_s4, &cost_mpi_s4);
  assert(rc == 0);
  assert_valid_solution(best_mpi_s4, n, K);

  double min_cost = 0.0;
  double max_cost = 0.0;
  rc = MPI_Allreduce(&cost_mpi_s1, &min_cost, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  assert(rc == MPI_SUCCESS);
  rc = MPI_Allreduce(&cost_mpi_s1, &max_cost, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  assert(rc == MPI_SUCCESS);
  assert(fabs(max_cost - min_cost) < 1e-9);

  /* Regression guard: parallel variants should stay in a reasonable band. */
  assert(cost_mpi_s1 <= cost_seq * 1.20 + 1e-9);
  assert(cost_mpi_s4 <= cost_seq * 1.25 + 1e-9);

  solution_free(best_seq);
  solution_free(best_mpi_s1);
  solution_free(best_mpi_s4);
  matrix_free(c);

  rc = MPI_Finalize();
  assert(rc == MPI_SUCCESS);
  if (rank == 0) {
    printf("MPI parallel tests passed.\n");
  }
  return 0;
}
