#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

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

static int parse_int_arg(const char *s, int *out) {
  char *end = NULL;
  errno = 0;
  long v = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    return 0;
  }
  if (v < 0 || v > 100000000L) {
    return 0;
  }
  *out = (int)v;
  return 1;
}

static unsigned int parse_uint_arg(const char *s, int *ok) {
  char *end = NULL;
  errno = 0;
  unsigned long v = strtoul(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFUL) {
    *ok = 0;
    return 0u;
  }
  *ok = 1;
  return (unsigned int)v;
}

int main(int argc, char **argv) {
  int status = 0;
  unsigned int seed = 1234u;

#ifdef USE_MPI
  int mpi_rank = 0;
  int provided = 0;
  if (MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided) !=
      MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init_thread failed\n");
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif

  int n = 5;
  int K = 2;
  int m = 10;
  int T = 50;

  if (argc == 6) {
    int ok = 1;
    ok = ok && parse_int_arg(argv[1], &n);
    ok = ok && parse_int_arg(argv[2], &K);
    ok = ok && parse_int_arg(argv[3], &m);
    ok = ok && parse_int_arg(argv[4], &T);
    seed = parse_uint_arg(argv[5], &ok);

    if (!ok || n <= 0 || K <= 0 || m <= 0 || T <= 0) {
      if (
#ifdef USE_MPI
          mpi_rank == 0 &&
#endif
          1) {
        fprintf(stderr,
                "usage: %s [n K m T seed]\n"
                "example: %s 200 16 1024 100 1234\n",
                argv[0], argv[0]);
      }
      status = 1;
      goto cleanup_mpi;
    }
  } else if (argc != 1) {
    if (
#ifdef USE_MPI
        mpi_rank == 0 &&
#endif
        1) {
      fprintf(stderr,
              "usage: %s [n K m T seed]\n"
              "example: %s 200 16 1024 100 1234\n",
              argv[0], argv[0]);
    }
    status = 1;
    goto cleanup_mpi;
  }

  double alpha = 1.0;
  double beta = 2.0;
  double rho = 0.5;
  double tau0 = 1.0;
  double Q = 1.0;

  double **c = matrix_alloc(n);
  if (!c) {
    fprintf(stderr, "failed to allocate cost matrix\n");
    status = 1;
    goto cleanup_mpi;
  }
  fill_example_costs(c, n);

  Solution *best = solution_create(K, n);
  if (!best) {
    fprintf(stderr, "failed to allocate solution\n");
    status = 1;
    matrix_free(c);
    goto cleanup_mpi;
  }

  double best_cost = 0.0;
  aco_vrp_sequential(n, K, m, T, c, alpha, beta, rho, tau0, Q, seed, best,
                     &best_cost);

#ifdef USE_MPI
  if (mpi_rank == 0) {
    printf("best cost: %.3f\n", best_cost);
  }
#else
  printf("best cost: %.3f\n", best_cost);
#endif

  solution_free(best);
  matrix_free(c);

cleanup_mpi:
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return status;
}
