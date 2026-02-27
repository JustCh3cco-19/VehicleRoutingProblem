#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_example_costs(double **c, int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) c[i][j] = 0.0;
      else c[i][j] = 1.0 + fabs((double)i - (double)j);
    }
  }
}

static int parse_positive_int(const char *text, int *out, const char *name) {
  char *end = NULL;
  long value = 0;
  errno = 0;
  value = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value <= 0 || value > 1000000L) {
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

  int n = 100;
  int K = 8;
  int m = 256;
  int T = 120;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world);

  if (argc != 1 && argc != 5) {
    if (rank == 0) {
      fprintf(stderr, "usage: %s [n K m T]\n", argv[0]);
    }
    exit_code = 1;
    goto finalize;
  }
  if (argc == 5) {
    if (parse_positive_int(argv[1], &n, "n") != 0 ||
        parse_positive_int(argv[2], &K, "K") != 0 ||
        parse_positive_int(argv[3], &m, "m") != 0 ||
        parse_positive_int(argv[4], &T, "T") != 0) {
      exit_code = 1;
      goto finalize;
    }
  }

  int dev_count = 0;
  cudaError_t cerr = cudaGetDeviceCount(&dev_count);
  if (cerr != cudaSuccess || dev_count <= 0) {
    if (rank == 0) {
      fprintf(stderr, "no CUDA devices available\n");
    }
    exit_code = 2;
    goto finalize;
  }
  int dev = rank % dev_count;
  if (cudaSetDevice(dev) != cudaSuccess) {
    fprintf(stderr, "[rank %d] cudaSetDevice(%d) failed\n", rank, dev);
    exit_code = 2;
    goto finalize;
  }

  double **c = matrix_alloc(n);
  Solution *best = solution_create(K, n);
  if (!c || !best) {
    fprintf(stderr, "[rank %d] allocation failure\n", rank);
    matrix_free(c);
    solution_free(best);
    exit_code = 3;
    goto finalize;
  }
  fill_example_costs(c, n);

  double local_best = INFINITY;
  double global_best = INFINITY;
  int local_owner = rank;
  int global_owner = -1;

  double t0 = MPI_Wtime();
  if (aco_vrp_cuda(n, K, m, T, c, 1.0, 2.0, 0.5, 1.0, 1.0,
                   1234u + (unsigned int)rank, best, &local_best) != 0) {
    fprintf(stderr, "[rank %d] aco_vrp_cuda failed\n", rank);
    matrix_free(c);
    solution_free(best);
    exit_code = 4;
    goto finalize;
  }
  double t1 = MPI_Wtime();

  struct {
    double value;
    int rank;
  } in = {local_best, local_owner}, out = {INFINITY, -1};
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  global_best = out.value;
  global_owner = out.rank;

  if (rank == 0) {
    printf("mode: mpi+cuda\n");
    printf("ranks: %d\n", world);
    printf("n=%d K=%d m=%d T=%d\n", n, K, m, T);
    printf("best cost: %.3f (owner rank %d)\n", global_best, global_owner);
    printf("elapsed: %.6f s\n", t1 - t0);
  }

  matrix_free(c);
  solution_free(best);

finalize:
  MPI_Finalize();
  return exit_code;
}
