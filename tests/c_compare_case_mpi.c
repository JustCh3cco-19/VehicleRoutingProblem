#include "aco.h"
#include "matrix.h"
#include "solution.h"
#include "test_types.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#else
#error "c_compare_case_mpi.c must be compiled with -DUSE_MPI"
#endif

/*
 * Function:  lcg_next
 * ------------------------
 * advances the local LCG state and returns the next value.
 *
 *  state: input parameter
 *
 *  returns: value as defined by the function contract
 */
static unsigned int lcg_next(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

/*
 * Function:  rand01_lcg
 * ------------------------
 * maps one LCG step to a floating-point value in [0, 1].
 *
 *  state: input parameter
 *
 *  returns: value as defined by the function contract
 */
static double rand01_lcg(unsigned int *state) {
  return (double)lcg_next(state) / (double)0xFFFFFFFFu;
}

/*
 * Function:  clamp
 * ------------------------
 * clamps a scalar value inside the inclusive [lo, hi] interval.
 *
 *  x: input parameter
 *  lo: input parameter
 *  hi: input parameter
 *
 *  returns: value as defined by the function contract
 */
static double clamp(double x, double lo, double hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/*
 * Function:  generate_points
 * ------------------------
 * generates deterministic synthetic points for one layout.
 *
 *  pts: input parameter
 *  n: input parameter
 *  seed: input parameter
 *  layout: input parameter
 *
 *  returns: value as defined by the function contract
 */
static void generate_points(Point *pts, int n, unsigned int seed, int layout) {
  const double scale = 1000.0;
  unsigned int st = seed ? seed : 1u;

  pts[0].x = 0.5 * scale;
  pts[0].y = 0.5 * scale;

  for (int i = 1; i <= n; ++i) {
    double r1 = rand01_lcg(&st);
    double r2 = rand01_lcg(&st);

    if (layout == 1) {
      int cluster = i % 3;
      static const double cx[3] = {0.2, 0.8, 0.5};
      static const double cy[3] = {0.2, 0.2, 0.8};
      pts[i].x = clamp((cx[cluster] + 0.25 * (r1 - 0.5)) * scale, 0.0, scale);
      pts[i].y = clamp((cy[cluster] + 0.25 * (r2 - 0.5)) * scale, 0.0, scale);
    } else if (layout == 2) {
      int left = ((i % 2) == 0);
      double xbase = left ? 0.04 : 0.84;
      pts[i].x = (xbase + 0.12 * r1) * scale;
      pts[i].y = r2 * scale;
    } else if (layout == 3) {
      pts[i].x = ((double)i / (double)(n + 1)) * scale;
      pts[i].y = (((i % 2) == 0) ? 0.35 : 0.65) * scale +
                 0.03 * (r2 - 0.5) * scale;
    } else {
      pts[i].x = r1 * scale;
      pts[i].y = r2 * scale;
    }
  }
}

/*
 * Function:  fill_cost_matrix
 * ------------------------
 * fills the rounded Euclidean distance matrix.
 *
 *  c: input parameter
 *  pts: input parameter
 *  n: input parameter
 *
 *  returns: value as defined by the function contract
 */
static void fill_cost_matrix(double **c, const Point *pts, int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        double xi = round(pts[i].x);
        double yi = round(pts[i].y);
        double xj = round(pts[j].x);
        double yj = round(pts[j].y);
        double dx = xi - xj;
        double dy = yi - yj;
        c[i][j] = sqrt(dx * dx + dy * dy);
      }
    }
  }
}

/*
 * Function:  parse_int
 * ------------------------
 * parses a signed integer argument.
 *
 *  s: input parameter
 *  out: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int parse_int(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (!s || end == s || *end != '\0') {
    return 0;
  }
  *out = (int)v;
  return 1;
}

/*
 * Function:  parse_uint
 * ------------------------
 * parses an unsigned integer argument.
 *
 *  s: input parameter
 *  out: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int parse_uint(const char *s, unsigned int *out) {
  char *end = NULL;
  unsigned long v = strtoul(s, &end, 10);
  if (!s || end == s || *end != '\0' || v > 0xFFFFFFFFUL) {
    return 0;
  }
  *out = (unsigned int)v;
  return 1;
}

/*
 * Function:  parse_double
 * ------------------------
 * parses a floating-point argument.
 *
 *  s: input parameter
 *  out: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int parse_double(const char *s, double *out) {
  char *end = NULL;
  double v = strtod(s, &end);
  if (!s || end == s || *end != '\0') {
    return 0;
  }
  *out = v;
  return 1;
}

/*
 * Function:  main
 * ------------------------
 * parses CLI options, executes scenarios, and writes outputs.
 *
 *  argc: input parameter
 *  argv: input parameter
 *
 *  returns: value as defined by the function contract
 */
int main(int argc, char **argv) {
  int mpi_rank = 0;
  int mpi_size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc != 13) {
    if (mpi_rank == 0) {
      fprintf(stderr,
              "usage: %s n K m T solver_seed instance_seed layout_id alpha beta rho tau0 Q\n",
              argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  int n = 0, K = 0, m = 0, T = 0, layout_id = 0;
  unsigned int solver_seed = 0u, instance_seed = 0u;
  double alpha = 1.0, beta = 3.0, rho = 0.3, tau0 = 1.0, Q = 1.0;

  if (!parse_int(argv[1], &n) || !parse_int(argv[2], &K) ||
      !parse_int(argv[3], &m) || !parse_int(argv[4], &T) ||
      !parse_uint(argv[5], &solver_seed) || !parse_uint(argv[6], &instance_seed) ||
      !parse_int(argv[7], &layout_id) || !parse_double(argv[8], &alpha) ||
      !parse_double(argv[9], &beta) || !parse_double(argv[10], &rho) ||
      !parse_double(argv[11], &tau0) || !parse_double(argv[12], &Q)) {
    if (mpi_rank == 0) {
      fprintf(stderr, "invalid arguments\n");
    }
    MPI_Finalize();
    return 1;
  }

  if (n <= 0 || K <= 0 || m < 0 || T <= 0 || layout_id < 0 || layout_id > 3 ||
      alpha <= 0.0 || beta <= 0.0 || rho <= 0.0 || rho >= 1.0 || tau0 <= 0.0 ||
      Q <= 0.0) {
    if (mpi_rank == 0) {
      fprintf(stderr, "arguments out of range\n");
    }
    MPI_Finalize();
    return 1;
  }

  Point *pts = malloc((size_t)(n + 1) * sizeof(*pts));
  double **c = matrix_alloc(n);
  Solution *best = solution_create(K, n);
  if (!pts || !c || !best) {
    if (mpi_rank == 0) {
      fprintf(stderr, "allocation failure\n");
    }
    solution_free(best);
    matrix_free(c);
    free(pts);
    MPI_Finalize();
    return 1;
  }

  generate_points(pts, n, instance_seed, layout_id);
  fill_cost_matrix(c, pts, n);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_s = MPI_Wtime();

  double best_cost = DBL_MAX;
  aco_vrp(n, K, m, T, c, alpha, beta, rho, tau0, Q, solver_seed, best,
          &best_cost);

  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed_s = MPI_Wtime() - start_s;

  double global_best_cost = 0.0;
  MPI_Allreduce(&best_cost, &global_best_cost, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);

  double max_elapsed_s = 0.0;
  MPI_Allreduce(&elapsed_s, &max_elapsed_s, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  if (mpi_rank == 0) {
    if (global_best_cost >= DBL_MAX / 2.0) {
      fprintf(stderr, "solver returned non-finite objective\n");
      solution_free(best);
      matrix_free(c);
      free(pts);
      MPI_Finalize();
      return 2;
    }
    printf("best_cost=%.6f\n", global_best_cost);
    printf("elapsed_s=%.6f\n", max_elapsed_s);
    printf("mpi_ranks=%d\n", mpi_size);
  }

  solution_free(best);
  matrix_free(c);
  free(pts);
  MPI_Finalize();
  return 0;
}
