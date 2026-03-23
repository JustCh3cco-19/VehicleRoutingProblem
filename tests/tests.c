#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Struct:  Point
 * --------------
 * stores one 2D coordinate used to build synthetic Euclidean instances.
 *
 *  x: x-axis coordinate
 *  y: y-axis coordinate
 */
typedef struct {
  double x;
  double y;
} Point;

/*
 * Struct:  GoldenCase
 * -------------------
 * describes one offline PyVRP baseline scenario loaded from CSV.
 *
 *  n: number of customers
 *  K: number of routes/vehicles
 *  m: number of ants
 *  T: number of ACO iterations
 *  solver_seed: deterministic seed for ACO
 *  instance_seed: deterministic seed for instance generation
 *  layout_id: synthetic geometry selector
 *  pyvrp_cost: reference objective produced by PyVRP
 *  max_gap_pct: maximum allowed positive gap vs PyVRP (percentage)
 */
typedef struct {
  int n;
  int K;
  int m;
  int T;
  unsigned int solver_seed;
  unsigned int instance_seed;
  int layout_id;
  double pyvrp_cost;
  double max_gap_pct;
} GoldenCase;

/*
 * Function:  lcg_next
 * -------------------
 * advances one step of a deterministic 32-bit linear congruential generator.
 * used to keep synthetic instance generation reproducible.
 *
 *  state: rng state updated in place
 *
 *  returns: next pseudo-random 32-bit value
 */
static unsigned int lcg_next(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

/*
 * Function:  rand01_lcg
 * ---------------------
 * maps the LCG state to a double in [0, 1].
 *
 *  state: rng state updated in place
 *
 *  returns: pseudo-random double in [0, 1]
 */
static double rand01_lcg(unsigned int *state) {
  return (double)lcg_next(state) / (double)0xFFFFFFFFu;
}

/*
 * Function:  clamp
 * ----------------
 * clamps one value inside a closed interval [lo, hi].
 *
 *  x: input value
 *  lo: lower bound
 *  hi: upper bound
 *
 *  returns: clamped value
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
 * --------------------------
 * builds deterministic depot+customer coordinates according to layout id:
 * 0 uniform, 1 clustered, 2 border-heavy, 3 parity/line pattern.
 *
 *  pts: output array of n+1 points (index 0 is depot)
 *  n: number of customers
 *  seed: deterministic instance seed
 *  layout: layout selector in [0, 3]
 *
 *  returns: nothing
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
      bool left = ((i % 2) == 0);
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
 * ---------------------------
 * converts coordinates to a full symmetric Euclidean cost matrix:
 * c[i][j] = 0 if i == j, else 1 + euclidean_distance(i, j).
 *
 *  c: matrix to fill
 *  pts: node coordinates
 *  n: number of customers (matrix size is n+1)
 *
 *  returns: nothing
 */
static void fill_cost_matrix(double **c, const Point *pts, int n) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        c[i][j] = 0.0;
      } else {
        double dx = pts[i].x - pts[j].x;
        double dy = pts[i].y - pts[j].y;
        c[i][j] = 1.0 + sqrt(dx * dx + dy * dy);
      }
    }
  }
}

/*
 * Function:  parse_golden_line
 * ----------------------------
 * parses one CSV row from the offline PyVRP golden file.
 * expected format:
 * n,K,m,T,solver_seed,instance_seed,layout_id,pyvrp_cost,max_gap_pct
 *
 *  line: input CSV row
 *  gc: output parsed case
 *
 *  returns: 1 on success
 *           0 on parse error
 */
static int parse_golden_line(const char *line, GoldenCase *gc) {
  int read = sscanf(line, "%d,%d,%d,%d,%u,%u,%d,%lf,%lf", &gc->n, &gc->K,
                    &gc->m, &gc->T, &gc->solver_seed, &gc->instance_seed,
                    &gc->layout_id, &gc->pyvrp_cost, &gc->max_gap_pct);
  return (read == 9);
}

/*
 * Function:  assert_valid_solution
 * --------------------------------
 * validates VRP structural constraints and aborts if invalid.
 *
 *  s: solution to validate
 *  n: number of customers
 *  K: number of routes
 *
 *  returns: nothing
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
 * Function:  run_case
 * -------------------
 * executes one golden scenario end-to-end:
 * - generate deterministic instance
 * - run ACO solver
 * - validate structure and objective coherence
 * - compare gap against offline PyVRP baseline threshold
 *
 *  gc: golden scenario with expected PyVRP cost and max allowed gap
 *
 *  returns: nothing (aborts on first failed check)
 */
static void run_case(const GoldenCase *gc) {
  Point *pts = malloc((size_t)(gc->n + 1) * sizeof(*pts));
  assert(pts);
  double **c = matrix_alloc(gc->n);
  assert(c);
  Solution *best = solution_create(gc->K, gc->n);
  assert(best);

  generate_points(pts, gc->n, gc->instance_seed, gc->layout_id);
  fill_cost_matrix(c, pts, gc->n);

  double best_cost = 0.0;
  aco_vrp(gc->n, gc->K, gc->m, gc->T, c, 1.0, 2.0, 0.5, 1.0, 1.0,
          gc->solver_seed, best, &best_cost);

  assert_valid_solution(best, gc->n, gc->K);
  double recomputed = solution_cost(best, c);
  double cost_tol = 1e-7;
  if (fabs(best_cost - recomputed) > cost_tol) {
    fprintf(stderr, "cost mismatch: best=%.9f recomputed=%.9f\n", best_cost,
            recomputed);
    assert(0);
  }

  double gap_pct = ((best_cost - gc->pyvrp_cost) / gc->pyvrp_cost) * 100.0;
  if (gap_pct > gc->max_gap_pct + 1e-9) {
    fprintf(stderr,
            "gap too large for n=%d: c=%.3f pyvrp=%.3f gap=%.3f%% max=%.3f%%\n",
            gc->n, best_cost, gc->pyvrp_cost, gap_pct, gc->max_gap_pct);
    assert(0);
  }

  printf("[OK] n=%d K=%d m=%d T=%d c=%.3f pyvrp=%.3f gap=%.3f%% (max=%.3f%%)\n",
         gc->n, gc->K, gc->m, gc->T, best_cost, gc->pyvrp_cost, gap_pct,
         gc->max_gap_pct);

  solution_free(best);
  matrix_free(c);
  free(pts);
}

/*
 * Function:  main
 * --------------
 * loads the offline golden CSV and runs all listed scenarios.
 *
 * usage:
 *   ./tests/test.out [golden_csv_path]
 *
 *  argc: number of command-line arguments
 *  argv: command-line argument array
 *
 *  returns: 0 on success
 *           1 on usage/file/parse failures
 */
int main(int argc, char **argv) {
  const char *path = "tests/files/golden_pyvrp.csv";
  if (argc == 2) {
    path = argv[1];
  } else if (argc > 2) {
    fprintf(stderr, "usage: %s [golden_csv_path]\n", argv[0]);
    return 1;
  }

  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "failed to open golden file: %s\n", path);
    return 1;
  }

  char line[512];
  int cases = 0;
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n') {
      continue;
    }
    if (!strncmp(line, "n,", 2)) {
      continue;
    }

    GoldenCase gc;
    if (!parse_golden_line(line, &gc)) {
      fprintf(stderr, "invalid golden row: %s", line);
      fclose(fp);
      return 1;
    }

    run_case(&gc);
    ++cases;
  }
  fclose(fp);

  if (cases == 0) {
    fprintf(stderr, "no golden cases found in %s\n", path);
    return 1;
  }

  printf("All tests passed (%d case(s)).\n", cases);
  return 0;
}
