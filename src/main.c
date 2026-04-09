#include "aco.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

/*
 * Function:  fill_example_costs
 * -----------------------------
 * fills the cost matrix with a simple symmetric example:
 * c[i][j] = 0 when i == j, otherwise 1 + |i-j|.
 *
 *  c: matrix to fill
 *  n: number of customers (matrix size is n+1)
 *
 *  returns: nothing
 */
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

static void print_solution_routes(const Solution *best, int K) {
  for (int i = 0; i < K; ++i) {
    const Route *r = &best->routes[i];
    int printed = 0;
    printf("Route %d:", i + 1);
    for (int t = 0; t < r->len; ++t) {
      int node = r->nodes[t];
      if (node != 0) {
        printf("%s%d", printed ? " " : " ", node);
        printed = 1;
      }
    }
    printf("\n");
  }
}

/*
 * Function:  parse_int_arg
 * ------------------------
 * parses one non-negative integer argument with bounds checking.
 *
 *  s: input string
 *  out: output parsed integer when parsing succeeds
 *
 *  returns: 1 on success
 *           0 on parsing/conversion/range error
 */
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

/*
 * Function:  parse_uint_arg
 * -------------------------
 * parses one unsigned 32-bit integer argument.
 *
 *  s: input string
 *  ok: output success flag (set to 1 on success, 0 on error)
 *
 *  returns: parsed unsigned value on success
 *           0 on error (with *ok set to 0)
 */
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

/*
 * Function:  main
 * --------------
 * parses CLI arguments, initializes optional MPI runtime, builds a demo cost
 * matrix, runs aco_vrp, and prints the best cost.
 *
 *  argc: number of command-line arguments
 *  argv: command-line argument array
 *
 *  returns: 0 on success
 *           1 on usage/initialization/allocation failures
 */
int main(int argc, char **argv) {
  int status = 0;
  unsigned int seed = 1234u;
  int use_instance_file = 0;
  const char *instance_path = NULL;

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

  if (argc == 6 && parse_int_arg(argv[1], &n)) {
    int ok = 1;
    ok = ok && parse_int_arg(argv[2], &K);
    ok = ok && parse_int_arg(argv[3], &m);
    ok = ok && parse_int_arg(argv[4], &T);
    seed = parse_uint_arg(argv[5], &ok);

    if (!ok || n <= 0 || K <= 0 || m < 0 || T <= 0) {
      if (
#ifdef USE_MPI
          mpi_rank == 0 &&
#endif
          1) {
        fprintf(stderr,
                "usage: %s [n K m T seed]\n"
                "example: %s 200 16 0 100 1234   (m=0 => auto ants)\n",
                argv[0], argv[0]);
      }
      status = 1;
      goto cleanup_mpi;
    }
  } else if ((argc == 5 || argc == 6) && argv[1] != NULL) {
    int ok = 1;
    instance_path = argv[1];
    ok = ok && parse_int_arg(argv[2], &K);
    ok = ok && parse_int_arg(argv[3], &m);
    ok = ok && parse_int_arg(argv[4], &T);
    if (argc == 6) {
      seed = parse_uint_arg(argv[5], &ok);
    }

    if (!ok || K <= 0 || m < 0 || T <= 0) {
      if (
#ifdef USE_MPI
          mpi_rank == 0 &&
#endif
          1) {
        fprintf(stderr,
                "usage: %s [n K m T seed]\n"
                "   or: %s <instance.vrp> <K> <m> <T> [seed]\n"
                "example: %s instances/test_aligned/n500_k8_s19000.vrp 8 32 20 9000\n",
                argv[0], argv[0], argv[0]);
      }
      status = 1;
      goto cleanup_mpi;
    }
    use_instance_file = 1;
  } else if (argc != 1) {
    if (
#ifdef USE_MPI
        mpi_rank == 0 &&
#endif
        1) {
      fprintf(stderr,
              "usage: %s [n K m T seed]\n"
              "   or: %s <instance.vrp> <K> <m> <T> [seed]\n"
              "example: %s 200 16 0 100 1234   (m=0 => auto ants)\n",
              argv[0], argv[0], argv[0]);
    }
    status = 1;
    goto cleanup_mpi;
  }

  double alpha = 1.0;
  double beta = 2.0;
  double rho = 0.5;
  double tau0 = 1.0;
  double Q = 1.0;

  double **c = NULL;
  if (use_instance_file) {
    if (vrp_load_tsplib_euc2d_matrix(instance_path, &n, &c) != 0) {
      if (
#ifdef USE_MPI
          mpi_rank == 0 &&
#endif
          1) {
        fprintf(stderr, "failed to load instance: %s\n", instance_path);
      }
      status = 1;
      goto cleanup_mpi;
    }
  } else {
    c = matrix_alloc(n);
    if (!c) {
      fprintf(stderr, "failed to allocate cost matrix\n");
      status = 1;
      goto cleanup_mpi;
    }
    fill_example_costs(c, n);
  }

  Solution *best = solution_create(K, n);
  if (!best) {
    fprintf(stderr, "failed to allocate solution\n");
    status = 1;
    matrix_free(c);
    goto cleanup_mpi;
  }

  double best_cost = 0.0;
  aco_vrp(n, K, m, T, c, alpha, beta, rho, tau0, Q, seed, best, &best_cost);

#ifdef USE_MPI
  if (mpi_rank == 0) {
    print_solution_routes(best, K);
    printf("Cost: %.3f\n", best_cost);
    printf("best cost: %.3f\n", best_cost);
  }
#else
  print_solution_routes(best, K);
  printf("Cost: %.3f\n", best_cost);
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
