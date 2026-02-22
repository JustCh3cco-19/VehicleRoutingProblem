#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <errno.h>
#include <math.h>
#include <mpi.h>
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

static void print_mpi_error(const char *what, int err) {
  char err_str[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(err, err_str, &len);
  fprintf(stderr, "%s failed: %.*s\n", what, len, err_str);
}

static int parse_positive_int(const char *text, int *out,
                              const char *name) {
  char *end = NULL;
  long value = 0;

  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 ||
      value > 1000000L) {
    fprintf(stderr, "invalid %s: '%s'\n", name, text);
    return -1;
  }
  *out = (int)value;
  return 0;
}

static int parse_non_negative_int(const char *text, int *out,
                                  const char *name) {
  char *end = NULL;
  long value = 0;

  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 ||
      value > 1000000L) {
    fprintf(stderr, "invalid %s: '%s'\n", name, text);
    return -1;
  }
  *out = (int)value;
  return 0;
}

int main(int argc, char **argv) {
  int rank = 0;
  int world = 1;
  int exit_code = 0;

  int n = 5;
  int K = 2;
  int m = 10;
  int T = 50;

  double alpha = 1.0;
  double beta = 2.0;
  double rho = 0.5;
  double tau0 = 1.0;
  double Q = 1.0;
  unsigned int seed = 1234;

  int threads_per_rank = 0;
  int sync_every = 1;

  double **c = NULL;
  Solution *best = NULL;

  int mpi_err = MPI_Init(&argc, &argv);
  if (mpi_err != MPI_SUCCESS) {
    print_mpi_error("MPI_Init", mpi_err);
    return 1;
  }

  mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (mpi_err != MPI_SUCCESS) {
    print_mpi_error("MPI_Comm_rank", mpi_err);
    exit_code = 1;
    goto finalize;
  }

  mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &world);
  if (mpi_err != MPI_SUCCESS) {
    print_mpi_error("MPI_Comm_size", mpi_err);
    exit_code = 1;
    goto finalize;
  }

  if (argc != 1 && argc != 3 && argc != 5 && argc != 7) {
    if (rank == 0) {
      fprintf(stderr,
              "usage: %s [threads_per_rank sync_every] [n K m T]\n"
              "   or: %s [n K m T]\n",
              argv[0], argv[0]);
    }
    exit_code = 1;
    goto finalize;
  }

  if (argc >= 3) {
    if (argc == 3 || argc == 7) {
      if (parse_non_negative_int(argv[1], &threads_per_rank,
                                 "threads_per_rank") != 0 ||
          parse_positive_int(argv[2], &sync_every, "sync_every") != 0) {
        exit_code = 1;
        goto finalize;
      }
    }
  }
  if (argc == 5 || argc == 7) {
    int arg_n = (argc == 5) ? 1 : 3;
    int arg_k = (argc == 5) ? 2 : 4;
    int arg_m = (argc == 5) ? 3 : 5;
    int arg_t = (argc == 5) ? 4 : 6;
    if (parse_positive_int(argv[arg_n], &n, "n") != 0 ||
        parse_positive_int(argv[arg_k], &K, "K") != 0 ||
        parse_positive_int(argv[arg_m], &m, "m") != 0 ||
        parse_positive_int(argv[arg_t], &T, "T") != 0) {
      exit_code = 1;
      goto finalize;
    }
  }

  c = matrix_alloc(n);
  if (!c) {
    if (rank == 0) {
      fprintf(stderr, "failed to allocate cost matrix\n");
    }
    exit_code = 1;
    goto finalize;
  }
  fill_example_costs(c, n);

  best = solution_create(K, n);
  if (!best) {
    if (rank == 0) {
      fprintf(stderr, "failed to allocate solution\n");
    }
    exit_code = 1;
    goto finalize;
  }

  double best_cost = 0.0;

  double t0 = MPI_Wtime();
  int rc = aco_vrp_mpi_openmp(n, K, m, T, c,
                              alpha, beta, rho, tau0, Q, seed,
                              threads_per_rank, sync_every,
                              best, &best_cost);
  double t1 = MPI_Wtime();

  if (rc != 0) {
    if (rank == 0) {
      fprintf(stderr, "aco_vrp_mpi_openmp failed\n");
    }
    exit_code = 2;
    goto finalize;
  }

  if (rank == 0) {
    printf("mode: mpi+omp\n");
    printf("ranks: %d\n", world);
    printf("threads per rank: %d (0 = default OpenMP)\n", threads_per_rank);
    printf("sync_every: %d\n", sync_every);
    printf("n=%d K=%d m=%d T=%d\n", n, K, m, T);
    printf("best cost: %.3f\n", best_cost);
    printf("elapsed: %.6f s\n", t1 - t0);
  }

finalize:
  solution_free(best);
  matrix_free(c);

  mpi_err = MPI_Finalize();
  if (mpi_err != MPI_SUCCESS) {
    print_mpi_error("MPI_Finalize", mpi_err);
    if (exit_code == 0) {
      exit_code = 1;
    }
  }

  return exit_code;
}
