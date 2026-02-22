#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
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

static int parse_positive_int(const char *text, int *out,
                              const char *name) {
  if (parse_non_negative_int(text, out, name) != 0) {
    return -1;
  }
  if (*out <= 0) {
    fprintf(stderr, "invalid %s: '%s' (must be > 0)\n", name, text);
    return -1;
  }
  return 0;
}

static double now_seconds(void) {
#ifdef _OPENMP
  return omp_get_wtime();
#else
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

int main(int argc, char **argv) {
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

  int num_threads = 0;
  if (argc != 1 && argc != 2 && argc != 5 && argc != 6) {
    fprintf(stderr, "usage: %s [threads] [n K m T]\n", argv[0]);
    return 1;
  }

  if (argc == 2 || argc == 6) {
    if (parse_non_negative_int(argv[1], &num_threads, "threads") != 0) {
      return 1;
    }
  }

  if (argc == 5) {
    if (parse_positive_int(argv[1], &n, "n") != 0 ||
        parse_positive_int(argv[2], &K, "K") != 0 ||
        parse_positive_int(argv[3], &m, "m") != 0 ||
        parse_positive_int(argv[4], &T, "T") != 0) {
      return 1;
    }
  } else if (argc == 6) {
    if (parse_positive_int(argv[2], &n, "n") != 0 ||
        parse_positive_int(argv[3], &K, "K") != 0 ||
        parse_positive_int(argv[4], &m, "m") != 0 ||
        parse_positive_int(argv[5], &T, "T") != 0) {
      return 1;
    }
  }

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
  double t0 = now_seconds();
  aco_vrp_openmp(n, K, m, T, c, alpha, beta, rho, tau0, Q, seed,
                 num_threads, best, &best_cost);
  double t1 = now_seconds();

  printf("mode: omp\n");
  printf("threads: %d (0 = default)\n", num_threads);
  printf("n=%d K=%d m=%d T=%d\n", n, K, m, T);
  printf("best cost: %.3f\n", best_cost);
  printf("elapsed: %.6f s\n", t1 - t0);

  solution_free(best);
  matrix_free(c);
  return 0;
}
