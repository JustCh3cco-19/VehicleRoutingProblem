#define _POSIX_C_SOURCE 200809L

#include "aco.h"
#include "matrix.h"
#include "solution.h"
#include "test_types.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

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

static double bytes_to_gib(size_t x) { return (double)x / (1024.0 * 1024.0 * 1024.0); }

/*
 * Function:  read_available_memory_bytes
 * ------------------------
 * reads available memory from /proc/meminfo.
 *
 *  returns: value as defined by the function contract
 */
static size_t read_available_memory_bytes(void) {
  FILE *fp = fopen("/proc/meminfo", "r");
  if (!fp) {
    return 0;
  }

  char key[64];
  unsigned long long value = 0;
  char unit[32];
  size_t out = 0;

  while (fscanf(fp, "%63s %llu %31s", key, &value, unit) == 3) {
    if (strcmp(key, "MemAvailable:") == 0) {
      out = (size_t)value * 1024u;
      break;
    }
    if (strcmp(key, "MemTotal:") == 0 && out == 0) {
      out = (size_t)value * 1024u;
    }
  }

  fclose(fp);
  return out;
}

/*
 * Function:  align_up_size
 * ------------------------
 * rounds a size up to the requested alignment.
 *
 *  value: input parameter
 *  alignment: input parameter
 *
 *  returns: value as defined by the function contract
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}

/*
 * Function:  choose_candidate_count_for_log
 * ------------------------
 * selects candidate-list size used for memory/report estimates.
 *
 *  n: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int choose_candidate_count_for_log(int n) {
  if (n <= 8) {
    return n;
  }
  if (n <= 256) {
    return 16;
  }
  if (n <= 4096) {
    return 24;
  }
  return 32;
}

/*
 * Function:  estimate_c_memory_bytes
 * ------------------------
 * estimates memory footprint for one scenario.
 *
 *  n: input parameter
 *
 *  returns: value as defined by the function contract
 */
static size_t estimate_c_memory_bytes(int n) {
  size_t side = (size_t)n + 1u;
  size_t dense_cost = side * side * sizeof(double); /* c */
  size_t dense_tau = side * side * sizeof(double);  /* tau */

  int cand_k = choose_candidate_count_for_log(n);
  size_t row_bytes = (size_t)cand_k * sizeof(float);
  size_t padded_row_bytes = align_up_size(row_bytes, 64u);
  size_t stride = padded_row_bytes / sizeof(float);
  size_t candidate_rows = side * stride;

  size_t candidate_idx = candidate_rows * sizeof(int);
  size_t eta_beta = candidate_rows * sizeof(float);
  size_t score = candidate_rows * sizeof(float);
  size_t base = dense_cost + dense_tau + candidate_idx + eta_beta + score;
  return (size_t)((double)base * 1.2);
}

/*
 * Function:  clamp_int
 * ------------------------
 * clamps an integer value between lower and upper bounds.
 *
 *  x: input parameter
 *  lo: input parameter
 *  hi: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/* Mirrors solver auto-ant policy for reporting in this runner. */
/*
 * Function:  estimate_auto_ants_for_log
 * ------------------------
 * mirrors solver auto-ant policy for user-facing logs.
 *
 *  n: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int estimate_auto_ants_for_log(int n) {
  int workers = 1;
  int ants_per_worker = 6;
  if (n <= 2000) {
    ants_per_worker = 8;
  } else if (n > 16000) {
    ants_per_worker = 4;
  }

  int total_ants = workers * ants_per_worker;
  total_ants = clamp_int(total_ants, workers * 4, workers * 8);
  if (total_ants < 8) {
    total_ants = 8;
  }
  return total_ants;
}

/*
 * Function:  build_scenarios
 * ------------------------
 * builds the deterministic scaling scenario list.
 *
 *  out: input parameter
 *  count: input parameter
 *
 *  returns: value as defined by the function contract
 */
static void build_scenarios(Scenario *out, int *count) {
  static const int levels[] = {500, 1000, 2000, 4000, 8000,
                               12000, 16000, 24000, 32000, 40000,
                               48000, 56000, 64000, 72000, 80000,
                               90000, 100000};
  const int total = (int)(sizeof(levels) / sizeof(levels[0]));

  for (int idx = 0; idx < total; ++idx) {
    int n = levels[idx];
    int m, T;

    if (n <= 2000) {
      m = 32;
      T = 20;
    } else if (n <= 8000) {
      m = 16;
      T = 10;
    } else if (n <= 16000) {
      m = 8;
      T = 6;
    } else if (n <= 32000) {
      m = 4;
      T = 4;
    } else {
      m = 3;
      T = 3;
    }

    out[idx].n = n;
    out[idx].K = n / 1000;
    if (out[idx].K < 8) {
      out[idx].K = 8;
    }
    if (out[idx].K > 128) {
      out[idx].K = 128;
    }
    out[idx].m = m;
    out[idx].T = T;
    out[idx].solver_seed = 9000u + (unsigned int)idx;
    out[idx].instance_seed = 19000u + (unsigned int)idx;
    out[idx].layout_id = idx % 4;
  }

  *count = total;
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
        c[i][j] = round(sqrt(dx * dx + dy * dy));
      }
    }
  }
}

/*
 * Function:  ensure_parent_dir
 * ------------------------
 * creates the parent directory of a file path if needed.
 *
 *  path: input parameter
 *
 *  returns: value as defined by the function contract
 */
static int ensure_parent_dir(const char *path) {
  const char *slash = strrchr(path, '/');
  if (!slash) {
    return 1;
  }

  size_t len = (size_t)(slash - path);
  if (len == 0u) {
    return 1;
  }

  char dir[1024];
  if (len >= sizeof(dir)) {
    return 0;
  }

  memcpy(dir, path, len);
  dir[len] = '\0';

  if (mkdir(dir, 0775) == 0 || errno == EEXIST) {
    return 1;
  }

  return 0;
}

static double elapsed_seconds(const struct timespec *start,
                              const struct timespec *end) {
  time_t ds = end->tv_sec - start->tv_sec;
  long dns = end->tv_nsec - start->tv_nsec;
  return (double)ds + (double)dns / 1e9;
}

/*
 * Function:  print_command_line
 * ------------------------
 * prints the current command line to the selected stream.
 *
 *  out: input parameter
 *  argc: input parameter
 *  argv: input parameter
 *
 *  returns: value as defined by the function contract
 */
static void print_command_line(FILE *out, int argc, char **argv) {
  if (!out || argc <= 0 || !argv) {
    return;
  }
  for (int i = 0; i < argc; ++i) {
    fprintf(out, "%s", argv[i]);
    if (i + 1 < argc) {
      fputc(' ', out);
    }
  }
  fputc('\n', out);
}

static void apply_solver_timer_env(double timeout_s, double stagnation_s) {
  char buf[64];
  if (timeout_s > 0.0) {
    snprintf(buf, sizeof(buf), "%.6f", timeout_s);
    setenv("ACO_SOLVER_TIMEOUT_SECONDS", buf, 1);
  } else {
    unsetenv("ACO_SOLVER_TIMEOUT_SECONDS");
  }

  if (stagnation_s > 0.0) {
    snprintf(buf, sizeof(buf), "%.6f", stagnation_s);
    setenv("ACO_SOLVER_STAGNATION_SECONDS", buf, 1);
  } else {
    unsetenv("ACO_SOLVER_STAGNATION_SECONDS");
  }
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
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);

  const char *csv_path = "results/scaling_progressive_c.csv";
  const char *input_log_path = "results/sequential_test_inputs.log";
  double memory_utilization = 0.70;
  int c_max_n = 100000;
  int force = 0;
  int auto_ants = 1;
  double solver_timeout_seconds = 0.0;
  double solver_stagnation_seconds = 0.0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (strcmp(argv[i], "--input-log") == 0 && i + 1 < argc) {
      input_log_path = argv[++i];
    } else if (strcmp(argv[i], "--memory-utilization") == 0 && i + 1 < argc) {
      memory_utilization = atof(argv[++i]);
      if (memory_utilization <= 0.0 || memory_utilization > 1.0) {
        fprintf(stderr, "invalid --memory-utilization, expected (0, 1].\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--c-max-n") == 0 && i + 1 < argc) {
      c_max_n = atoi(argv[++i]);
      if (c_max_n <= 0) {
        fprintf(stderr, "invalid --c-max-n, expected positive integer.\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--force") == 0) {
      force = 1;
    } else if (strcmp(argv[i], "--auto-ants") == 0) {
      auto_ants = 1;
    } else if (strcmp(argv[i], "--fixed-ants") == 0) {
      auto_ants = 0;
    } else if (strcmp(argv[i], "--solver-timeout-seconds") == 0 &&
               i + 1 < argc) {
      solver_timeout_seconds = atof(argv[++i]);
      if (solver_timeout_seconds <= 0.0) {
        fprintf(stderr,
                "invalid --solver-timeout-seconds, expected positive number.\n");
        return 1;
      }
    } else if (strcmp(argv[i], "--solver-stagnation-seconds") == 0 &&
               i + 1 < argc) {
      solver_stagnation_seconds = atof(argv[++i]);
      if (solver_stagnation_seconds <= 0.0) {
        fprintf(stderr,
                "invalid --solver-stagnation-seconds, expected positive number.\n");
        return 1;
      }
    } else {
      fprintf(stderr,
              "usage: %s [--csv PATH] [--input-log PATH] [--memory-utilization X] [--c-max-n N] [--force] [--auto-ants|--fixed-ants] [--solver-timeout-seconds X] [--solver-stagnation-seconds X]\n",
              argv[0]);
      return 1;
    }
  }

  if (!ensure_parent_dir(csv_path)) {
    fprintf(stderr, "failed to create parent directory for CSV: %s\n", csv_path);
    return 1;
  }
  if (!ensure_parent_dir(input_log_path)) {
    fprintf(stderr, "failed to create parent directory for input log: %s\n",
            input_log_path);
    return 1;
  }

  FILE *csv = fopen(csv_path, "w");
  if (!csv) {
    fprintf(stderr, "failed to open output CSV: %s\n", csv_path);
    return 1;
  }
  FILE *input_log = fopen(input_log_path, "w");
  if (!input_log) {
    fprintf(stderr, "failed to open input log: %s\n", input_log_path);
    fclose(csv);
    return 1;
  }
  setvbuf(csv, NULL, _IOLBF, 0);
  setvbuf(input_log, NULL, _IOLBF, 0);

  fprintf(csv,
          "mode,n,K,m,T,estimated_mem_gib,status,elapsed_s,c_cost,error\n");
  fflush(csv);

  size_t available = read_available_memory_bytes();
  size_t threshold = (size_t)((double)available * memory_utilization);

  printf("========================================================================\n");
  printf("C Progressive Scaling\n");
  printf("========================================================================\n");
  printf("[INFO] available_mem_gib    : %.2f\n", bytes_to_gib(available));
  printf("[INFO] threshold_gib        : %.2f\n", bytes_to_gib(threshold));
  printf("[INFO] memory_utilization   : %.2f\n", memory_utilization);
  printf("[INFO] c_max_n              : %d\n\n", c_max_n);
  printf("[INFO] ants_mode            : %s\n\n",
         auto_ants ? "auto(solver,m=0)" : "fixed(scenario)");
  printf("[INFO] solver_timeout_s     : %.2f\n", solver_timeout_seconds);
  printf("[INFO] solver_stagnation_s  : %.2f\n\n",
         solver_stagnation_seconds);
  printf("[INFO] csv_path             : %s\n", csv_path);
  printf("[INFO] input_log_path       : %s\n", input_log_path);
  printf("[INFO] command              : ");
  print_command_line(stdout, argc, argv);
  printf("\n");

  fprintf(input_log, "# sequential_tests input log\n");
  fprintf(input_log, "command: ");
  print_command_line(input_log, argc, argv);
  fprintf(input_log, "csv_path=%s\n", csv_path);
  fprintf(input_log, "memory_utilization=%.6f\n", memory_utilization);
  fprintf(input_log, "c_max_n=%d\n", c_max_n);
  fprintf(input_log, "force=%d\n", force);
  fprintf(input_log, "auto_ants=%d\n", auto_ants);
  fprintf(input_log, "solver_timeout_seconds=%.6f\n", solver_timeout_seconds);
  fprintf(input_log, "solver_stagnation_seconds=%.6f\n",
          solver_stagnation_seconds);
  fprintf(input_log, "available_mem_gib=%.4f\n", bytes_to_gib(available));
  fprintf(input_log, "threshold_gib=%.4f\n\n", bytes_to_gib(threshold));
  fflush(input_log);
  apply_solver_timer_env(solver_timeout_seconds, solver_stagnation_seconds);

  Scenario scenarios[24];
  int total = 0;
  build_scenarios(scenarios, &total);

  int ok_count = 0;
  int skipped_count = 0;
  int failed_count = 0;
  double total_elapsed = 0.0;

  for (int i = 0; i < total; ++i) {
    const Scenario *sc = &scenarios[i];
    int run_m = auto_ants ? 0 : sc->m;
    int run_m_effective = auto_ants ? estimate_auto_ants_for_log(sc->n) : sc->m;
    size_t est_mem = estimate_c_memory_bytes(sc->n);
    double est_gib = bytes_to_gib(est_mem);

    printf("[RUN  %02d/%02d] customers=%-6d vehicles=%-4d ants=%-3d iterations=%-3d est_mem=%6.2f GiB\n",
           i + 1, total, sc->n, sc->K, run_m_effective, sc->T, est_gib);
    printf("  [INPUT] solver   : n=%d K=%d m=%d T=%d alpha=1.0 beta=2.0 rho=0.5 tau0=1.0 Q=1.0 seed=%u\n",
           sc->n, sc->K, run_m, sc->T, sc->solver_seed);
    printf("  [INPUT] instance : seed=%u layout_id=%d\n",
           sc->instance_seed, sc->layout_id);

    fprintf(input_log,
            "[RUN %02d/%02d] n=%d K=%d m_passed=%d m_effective=%d T=%d alpha=1.0 beta=2.0 rho=0.5 tau0=1.0 Q=1.0 solver_seed=%u instance_seed=%u layout_id=%d est_mem_gib=%.4f\n",
            i + 1, total, sc->n, sc->K, run_m, run_m_effective, sc->T,
            sc->solver_seed, sc->instance_seed, sc->layout_id, est_gib);

    if (sc->n > c_max_n) {
      ++skipped_count;
      printf("  [SKIP] n=%d supera c-max-n=%d\n", sc->n, c_max_n);
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,skipped_c_max_n,,,\n", sc->n, sc->K,
              run_m_effective, sc->T, est_gib);
      fprintf(input_log, "  result=skipped_c_max_n c_max_n=%d\n\n", c_max_n);
      fflush(csv);
      fflush(input_log);
      continue;
    }

    if (!force && threshold > 0 && est_mem > threshold) {
      ++skipped_count;
      printf("  [SKIP] memoria stimata %.2f GiB > soglia %.2f GiB\n",
             est_gib, bytes_to_gib(threshold));
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,skipped_memory,,,\n", sc->n, sc->K,
              run_m_effective, sc->T, est_gib);
      fprintf(input_log,
              "  result=skipped_memory est_mem_gib=%.4f threshold_gib=%.4f\n\n",
              est_gib, bytes_to_gib(threshold));
      fflush(csv);
      fflush(input_log);
      continue;
    }

    Point *pts = malloc((size_t)(sc->n + 1) * sizeof(*pts));
    double **c = matrix_alloc(sc->n);
    Solution *best = solution_create(sc->K, sc->n);
    double best_cost = DBL_MAX;

    if (!pts || !c || !best) {
      ++failed_count;
      printf("  [FAIL] allocation failure before solve\n");
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,failed_alloc,,,allocation_failure\n",
              sc->n, sc->K, run_m_effective, sc->T, est_gib);
      fprintf(input_log, "  result=failed_alloc error=allocation_failure\n\n");
      fflush(csv);
      fflush(input_log);
      solution_free(best);
      matrix_free(c);
      free(pts);
      continue;
    }

    generate_points(pts, sc->n, sc->instance_seed, sc->layout_id);
    fill_cost_matrix(c, pts, sc->n);

    struct timespec start_ts;
    struct timespec end_ts;
    timespec_get(&start_ts, TIME_UTC);
    aco_vrp(sc->n, sc->K, run_m, sc->T, c, 1.0, 2.0, 0.5, 1.0, 1.0,
            sc->solver_seed, best, &best_cost);
    timespec_get(&end_ts, TIME_UTC);

    double elapsed = elapsed_seconds(&start_ts, &end_ts);

    if (best_cost >= DBL_MAX / 2.0) {
      ++failed_count;
      printf("  [FAIL] solver did not produce a finite objective\n");
      fprintf(csv,
              "c,%d,%d,%d,%d,%.4f,failed_solver,%.6f,,non_finite_objective\n",
              sc->n, sc->K, run_m_effective, sc->T, est_gib, elapsed);
      fprintf(input_log,
              "  result=failed_solver elapsed_s=%.6f error=non_finite_objective\n\n",
              elapsed);
      fflush(csv);
      fflush(input_log);
    } else {
      ++ok_count;
      total_elapsed += elapsed;
      printf("  [OK] c solver completed in %7.2fs, objective=%.3f\n", elapsed,
             best_cost);
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,ok,%.6f,%.6f,\n", sc->n, sc->K,
              run_m_effective, sc->T, est_gib, elapsed, best_cost);
      fprintf(input_log, "  result=ok elapsed_s=%.6f best_cost=%.6f\n\n", elapsed,
              best_cost);
      fflush(csv);
      fflush(input_log);
    }

    solution_free(best);
    matrix_free(c);
    free(pts);
  }

  fclose(csv);
  fclose(input_log);

  printf("\n========================================================================\n");
  printf("C Scaling Summary\n");
  printf("========================================================================\n");
  printf("[SUMMARY] scenarios_total   : %d\n", total);
  printf("[SUMMARY] ok                : %d\n", ok_count);
  printf("[SUMMARY] failed            : %d\n", failed_count);
  printf("[SUMMARY] skipped           : %d\n", skipped_count);
  printf("[SUMMARY] total_solve_s     : %.2f\n", total_elapsed);
  printf("[DONE] wrote %s\n", csv_path);
  printf("[DONE] wrote %s\n", input_log_path);

  return 0;
}
