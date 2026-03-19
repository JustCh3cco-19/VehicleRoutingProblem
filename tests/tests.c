#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

/*
 * Default directory that contains all scenario files.
 */
#define DEFAULT_CASE_DIR "tests/files"
#define PATH_BUF_LEN 512

typedef struct {
  double x;
  double y;
} Point;

typedef struct {
  int n;
  int K;
  int m;
  int T;
  unsigned int solver_seed;
  unsigned int instance_seed;
  int layout_id;
} Scenario;

/*
 * Ordered list of all scenario files that compose the full regression suite.
 */
static const char *kCaseFiles[] = {
    "test_01_a35_p5_w3",   "test_01_a35_p7_w2",   "test_01_a35_p8_w1",
    "test_01_a35_p8_w4",   "test_02_a30k_p20k_w1", "test_02_a30k_p20k_w2",
    "test_02_a30k_p20k_w3", "test_02_a30k_p20k_w4", "test_02_a30k_p20k_w5",
    "test_02_a30k_p20k_w6", "test_03_a20_p4_w1",   "test_04_a20_p4_w1",
    "test_05_a20_p4_w1",   "test_06_a20_p4_w1",   "test_07_a1M_p5k_w1",
    "test_07_a1M_p5k_w2",  "test_07_a1M_p5k_w3",  "test_07_a1M_p5k_w4",
    "test_08_a100M_p1_w1", "test_08_a100M_p1_w2", "test_08_a100M_p1_w3",
    "test_09_a16-17_p3_w1",
};

/*
 * Asserts that two floating-point values are close within an absolute
 * tolerance.
 */
static void assert_close(double a, double b, double tol) {
  if (fabs(a - b) > tol) {
    fprintf(stderr, "assert_close failed: %.12f vs %.12f\n", a, b);
    assert(0);
  }
}

/*
 * Validates VRP structural constraints and aborts on the first invalid
 * solution.
 */
static void assert_valid_solution(const Solution *s, int n, int K, int rank) {
  char err[128];
  bool ok = solution_validate(s, n, K, err, sizeof(err));
  if (!ok) {
    fprintf(stderr, "rank %d invalid solution: %s\n", rank, err);
    assert(0);
  }
}

/*
 * Simple deterministic LCG used to generate synthetic but reproducible
 * instances from seeds in the scenario files.
 */
static unsigned int lcg_next(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

/*
 * Generates pseudo-random doubles in [0,1] using the deterministic LCG.
 */
static double rand01_lcg(unsigned int *state) {
  return (double)lcg_next(state) / (double)0xFFFFFFFFu;
}

/*
 * Clamps one value inside a closed interval.
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
 * Builds depot + customer coordinates according to the selected layout_id:
 * 0 uniform, 1 clustered, 2 border-heavy, 3 parity/line pattern.
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
 * Converts coordinates into a full symmetric Euclidean cost matrix.
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
 * Parses one scenario line from a case file and validates basic ranges.
 */
static bool parse_scenario(FILE *fp, Scenario *sc) {
  int read = fscanf(fp, "%d %d %d %d %u %u %d", &sc->n, &sc->K, &sc->m,
                    &sc->T, &sc->solver_seed, &sc->instance_seed,
                    &sc->layout_id);
  if (read != 7) {
    return false;
  }

  if (sc->n <= 0 || sc->K <= 0 || sc->m <= 0 || sc->T <= 0) {
    return false;
  }
  if (sc->layout_id < 0 || sc->layout_id > 3) {
    return false;
  }
  return true;
}

/*
 * Runs one scenario end-to-end:
 * - generate deterministic instance
 * - execute ACO
 * - validate route structure and objective coherence
 * - in MPI mode, ensure all ranks agree on best cost
 */
static void run_single_scenario(const char *case_name, int scenario_index,
                                const Scenario *sc, int rank) {
  Point *pts = malloc((size_t)(sc->n + 1) * sizeof(Point));
  assert(pts);

  double **c = matrix_alloc(sc->n);
  assert(c);

  generate_points(pts, sc->n, sc->instance_seed, sc->layout_id);
  fill_cost_matrix(c, pts, sc->n);

  int repeats = (sc->n <= 300) ? 2 : 1;
  double first_cost = -1.0;

  for (int run = 0; run < repeats; ++run) {
    Solution *best = solution_create(sc->K, sc->n);
    assert(best);

    double best_cost = 0.0;
    aco_vrp(sc->n, sc->K, sc->m, sc->T, c, 1.0, 2.0, 0.5, 1.0, 1.0,
            sc->solver_seed, best, &best_cost);

    assert_valid_solution(best, sc->n, sc->K, rank);
    assert(isfinite(best_cost));

    double recomputed = solution_cost(best, c);
    assert_close(best_cost, recomputed, 1e-7);

    if (run == 0) {
      first_cost = best_cost;
    } else {
      assert_close(first_cost, best_cost, 1e-9);
    }

#ifdef USE_MPI
    double min_cost = 0.0;
    double max_cost = 0.0;
    MPI_Allreduce(&best_cost, &min_cost, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&best_cost, &max_cost, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    assert_close(min_cost, max_cost, 1e-9);
#endif

    solution_free(best);
  }

#ifdef USE_MPI
  if (rank == 0) {
#endif
    printf("[OK] %s scenario %d (n=%d K=%d m=%d T=%d)\n", case_name,
           scenario_index + 1, sc->n, sc->K, sc->m, sc->T);
#ifdef USE_MPI
  }
#endif

  matrix_free(c);
  free(pts);
}

/*
 * Loads one case file and executes all scenarios contained in it.
 */
static int run_case_file(const char *case_dir, const char *case_name,
                         int rank) {
  char path[PATH_BUF_LEN];
  int written = snprintf(path, sizeof(path), "%s/%s", case_dir, case_name);
  assert(written > 0 && written < (int)sizeof(path));

  FILE *fp = fopen(path, "r");
  if (!fp) {
    fprintf(stderr, "failed to open case file: %s\n", path);
    assert(0);
  }

  int scenario_count = 0;
  int read = fscanf(fp, "%d", &scenario_count);
  if (read != 1 || scenario_count <= 0) {
    fprintf(stderr, "invalid scenario count in: %s\n", path);
    fclose(fp);
    assert(0);
  }

  for (int i = 0; i < scenario_count; ++i) {
    Scenario sc;
    if (!parse_scenario(fp, &sc)) {
      fprintf(stderr, "invalid scenario line %d in: %s\n", i + 1, path);
      fclose(fp);
      assert(0);
    }
    run_single_scenario(case_name, i, &sc, rank);
  }

  fclose(fp);
  return scenario_count;
}

/*
 * Test entrypoint.
 * Usage:
 *   ./tests            -> run full suite
 *   ./tests <dir>      -> run full suite from custom directory
 *   ./tests <dir> <file> -> run one case file
 * In MPI builds this initializes/finalizes MPI and keeps output on rank 0.
 */
int main(int argc, char **argv) {
  int rank = 0;

#ifdef USE_MPI
  int provided = 0;
  int rc = MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  if (rc != MPI_SUCCESS) {
    fprintf(stderr, "MPI_Init_thread failed\n");
    return 1;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  if (argc > 3) {
    if (rank == 0) {
      fprintf(stderr, "usage: %s [case_dir] [case_name]\n", argv[0]);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 1;
  }

  const char *case_dir = (argc >= 2) ? argv[1] : DEFAULT_CASE_DIR;
  const char *case_name = (argc == 3) ? argv[2] : NULL;

  int total_scenarios = 0;

  if (case_name) {
    total_scenarios = run_case_file(case_dir, case_name, rank);
    if (rank == 0) {
      printf("Final test passed (%s, %d scenario(s)).\n", case_name,
             total_scenarios);
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
  }

  int total_files = (int)(sizeof(kCaseFiles) / sizeof(kCaseFiles[0]));
  for (int i = 0; i < total_files; ++i) {
    total_scenarios += run_case_file(case_dir, kCaseFiles[i], rank);
  }

  if (rank == 0) {
    printf("All final tests passed (%d files, %d scenarios).\n", total_files,
           total_scenarios);
  }

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
