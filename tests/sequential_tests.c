#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>

typedef struct {
  int n;
  int K;
  int m;
  int T;
  unsigned int solver_seed;
  unsigned int instance_seed;
  int layout_id;
} Scenario;

typedef struct {
  double x;
  double y;
} Point;

static unsigned int lcg_next(unsigned int *state) {
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

static double rand01_lcg(unsigned int *state) {
  return (double)lcg_next(state) / (double)0xFFFFFFFFu;
}

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

static size_t estimate_c_memory_bytes(int n) {
  size_t side = (size_t)n + 1u;
  size_t one_dense = side * side * sizeof(double);
  size_t dense_three = one_dense * 3u; /* c + eta + tau */
  size_t layered_cache = side * (size_t)(8 + 64) * sizeof(double); /* L1 + L2 */
  size_t base = dense_three + layered_cache;
  return (size_t)((double)base * 1.2); /* safety overhead */
}

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

int main(int argc, char **argv) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);

  const char *csv_path = "results/scaling_progressive_c.csv";
  double memory_utilization = 0.70;
  int c_max_n = 100000;
  int force = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
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
    } else {
      fprintf(stderr,
              "usage: %s [--csv PATH] [--memory-utilization X] [--c-max-n N] [--force]\n",
              argv[0]);
      return 1;
    }
  }

  if (!ensure_parent_dir(csv_path)) {
    fprintf(stderr, "failed to create parent directory for CSV: %s\n", csv_path);
    return 1;
  }

  FILE *csv = fopen(csv_path, "w");
  if (!csv) {
    fprintf(stderr, "failed to open output CSV: %s\n", csv_path);
    return 1;
  }
  setvbuf(csv, NULL, _IOLBF, 0);

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

  Scenario scenarios[24];
  int total = 0;
  build_scenarios(scenarios, &total);

  int ok_count = 0;
  int skipped_count = 0;
  int failed_count = 0;
  double total_elapsed = 0.0;

  for (int i = 0; i < total; ++i) {
    const Scenario *sc = &scenarios[i];
    size_t est_mem = estimate_c_memory_bytes(sc->n);
    double est_gib = bytes_to_gib(est_mem);

    printf("[RUN  %02d/%02d] customers=%-6d vehicles=%-4d ants=%-3d iterations=%-3d est_mem=%6.2f GiB\n",
           i + 1, total, sc->n, sc->K, sc->m, sc->T, est_gib);

    if (sc->n > c_max_n) {
      ++skipped_count;
      printf("  [SKIP] customers > c-max-n (%d)\n", c_max_n);
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,skipped_c_max_n,,,\n", sc->n, sc->K,
              sc->m, sc->T, est_gib);
      fflush(csv);
      continue;
    }

    if (!force && threshold > 0 && est_mem > threshold) {
      ++skipped_count;
      printf("  [SKIP] estimated memory above threshold\n");
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,skipped_memory,,,\n", sc->n, sc->K,
              sc->m, sc->T, est_gib);
      fflush(csv);
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
              sc->n, sc->K, sc->m, sc->T, est_gib);
      fflush(csv);
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
    aco_vrp(sc->n, sc->K, sc->m, sc->T, c, 1.0, 2.0, 0.5, 1.0, 1.0,
            sc->solver_seed, best, &best_cost);
    timespec_get(&end_ts, TIME_UTC);

    double elapsed = elapsed_seconds(&start_ts, &end_ts);

    if (best_cost >= DBL_MAX / 2.0) {
      ++failed_count;
      printf("  [FAIL] solver did not produce a finite objective\n");
      fprintf(csv,
              "c,%d,%d,%d,%d,%.4f,failed_solver,%.6f,,non_finite_objective\n",
              sc->n, sc->K, sc->m, sc->T, est_gib, elapsed);
      fflush(csv);
    } else {
      ++ok_count;
      total_elapsed += elapsed;
      printf("  [OK] c solver completed in %7.2fs, objective=%.3f\n", elapsed,
             best_cost);
      fprintf(csv, "c,%d,%d,%d,%d,%.4f,ok,%.6f,%.6f,\n", sc->n, sc->K, sc->m,
              sc->T, est_gib, elapsed, best_cost);
      fflush(csv);
    }

    solution_free(best);
    matrix_free(c);
    free(pts);
  }

  fclose(csv);

  printf("\n========================================================================\n");
  printf("C Scaling Summary\n");
  printf("========================================================================\n");
  printf("[SUMMARY] scenarios_total   : %d\n", total);
  printf("[SUMMARY] ok                : %d\n", ok_count);
  printf("[SUMMARY] failed            : %d\n", failed_count);
  printf("[SUMMARY] skipped           : %d\n", skipped_count);
  printf("[SUMMARY] total_solve_s     : %.2f\n", total_elapsed);
  printf("[DONE] wrote %s\n", csv_path);

  return 0;
}
