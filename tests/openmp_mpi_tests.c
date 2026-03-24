#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef USE_MPI
#include <mpi.h>
#else
#error "openmp_mpi_tests.c must be compiled with -DUSE_MPI"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

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

static double bytes_to_gib(size_t x) {
  return (double)x / (1024.0 * 1024.0 * 1024.0);
}

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
  size_t dense_three = one_dense * 3u;
  size_t layered_cache = side * (size_t)(8 + 64) * sizeof(double); /* L1 + L2 */
  size_t base = dense_three + layered_cache;
  return (size_t)((double)base * 1.2);
}

static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/* Mirrors solver auto-ant policy only for user-facing reporting in this runner. */
static int estimate_auto_ants_for_log(int n, int mpi_ranks) {
  int omp_threads = 1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif
  if (omp_threads < 1) {
    omp_threads = 1;
  }

  int workers = mpi_ranks * omp_threads;
  if (workers < 1) {
    workers = 1;
  }

  int ants_per_worker;
  if (n <= 2000) {
    ants_per_worker = 4;
  } else if (n <= 8000) {
    ants_per_worker = 3;
  } else if (n <= 16000) {
    ants_per_worker = 2;
  } else {
    ants_per_worker = 1;
  }

  int total_ants = workers * ants_per_worker;
  total_ants = clamp_int(total_ants, workers, workers * 16);
  total_ants = clamp_int(total_ants, 8, n > 8 ? n : 8);
  return total_ants;
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
      m = 64;
      T = 40;
    } else if (n <= 8000) {
      m = 32;
      T = 20;
    } else if (n <= 16000) {
      m = 16;
      T = 12;
    } else if (n <= 32000) {
      m = 8;
      T = 8;
    } else {
      m = 4;
      T = 6;
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

int main(int argc, char **argv) {
  int rank = 0;
  int size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char *csv_path = "results/scaling_progressive_openmp_mpi.csv";
  double memory_utilization = 0.70;
  int c_max_n = 100000;
  int enforce_c_max_n = 0;
  int force = 0;
  int auto_ants = 1;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (strcmp(argv[i], "--memory-utilization") == 0 && i + 1 < argc) {
      memory_utilization = atof(argv[++i]);
      if (memory_utilization <= 0.0 || memory_utilization > 1.0) {
        if (rank == 0) {
          fprintf(stderr, "invalid --memory-utilization, expected (0, 1].\n");
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--c-max-n") == 0 && i + 1 < argc) {
      c_max_n = atoi(argv[++i]);
      if (c_max_n <= 0) {
        if (rank == 0) {
          fprintf(stderr, "invalid --c-max-n, expected positive integer.\n");
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--enforce-c-max-n") == 0) {
      enforce_c_max_n = 1;
    } else if (strcmp(argv[i], "--force") == 0) {
      force = 1;
    } else if (strcmp(argv[i], "--auto-ants") == 0) {
      auto_ants = 1;
    } else if (strcmp(argv[i], "--fixed-ants") == 0) {
      auto_ants = 0;
    } else {
      if (rank == 0) {
        fprintf(stderr,
          "usage: %s [--csv PATH] [--memory-utilization X] [--c-max-n N] [--enforce-c-max-n] [--force] [--auto-ants|--fixed-ants]\n",
                argv[0]);
      }
      MPI_Finalize();
      return 1;
    }
  }

  FILE *csv = NULL;
  if (rank == 0) {
    if (!ensure_parent_dir(csv_path)) {
      fprintf(stderr, "failed to create parent directory for CSV: %s\n", csv_path);
      MPI_Finalize();
      return 1;
    }

    csv = fopen(csv_path, "w");
    if (!csv) {
      fprintf(stderr, "failed to open output CSV: %s\n", csv_path);
      MPI_Finalize();
      return 1;
    }

    fprintf(csv,
            "mode,mpi_ranks,n,K,m,T,estimated_mem_gib,status,elapsed_s,cost,error\n");
  }

  size_t available = read_available_memory_bytes();
  size_t threshold = (size_t)((double)available * memory_utilization);

  if (rank == 0) {
    int omp_threads = 1;
  #ifdef _OPENMP
    omp_threads = omp_get_max_threads();
  #endif
    printf("========================================================================\n");
    printf("OpenMP+MPI Progressive Scaling\n");
    printf("========================================================================\n");
    printf("[INFO] mpi_ranks            : %d\n", size);
    printf("[INFO] omp_threads/rank     : %d\n", omp_threads);
        printf("[INFO] ants_mode            : %s\n",
          auto_ants ? "auto(solver,m=0)" : "fixed(scenario)");
    printf("[INFO] available_mem_gib    : %.2f\n", bytes_to_gib(available));
    printf("[INFO] threshold_gib        : %.2f\n", bytes_to_gib(threshold));
    printf("[INFO] memory_utilization   : %.2f\n", memory_utilization);
    printf("[INFO] c_max_n              : %d\n\n", c_max_n);
            printf("[INFO] enforce_c_max_n      : %s\n\n",
              enforce_c_max_n ? "yes" : "no");
  }

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
    int run_m_effective = auto_ants ? estimate_auto_ants_for_log(sc->n, size) : sc->m;
    size_t est_mem = estimate_c_memory_bytes(sc->n);
    double est_gib = bytes_to_gib(est_mem);

    if (rank == 0) {
      printf("[RUN  %02d/%02d] customers=%-6d vehicles=%-4d ants=%-3d iterations=%-3d est_mem=%6.2f GiB\n",
              i + 1, total, sc->n, sc->K, run_m_effective, sc->T, est_gib);
    }

    if (enforce_c_max_n && sc->n > c_max_n) {
      if (rank == 0) {
        ++skipped_count;
        printf("  [SKIP] customers > c-max-n (%d)\n", c_max_n);
        fprintf(csv,
                "openmp_mpi,%d,%d,%d,%d,%d,%.4f,skipped_c_max_n,,,\n",
          size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
      }
      continue;
    }

    if (!force && threshold > 0 && est_mem > threshold) {
      if (rank == 0) {
        ++skipped_count;
        printf("  [SKIP] estimated memory above threshold\n");
        fprintf(csv,
                "openmp_mpi,%d,%d,%d,%d,%d,%.4f,skipped_memory,,,\n",
          size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
      }
      continue;
    }

    Point *pts = malloc((size_t)(sc->n + 1) * sizeof(*pts));
    double **c = matrix_alloc(sc->n);
    Solution *best = solution_create(sc->K, sc->n);
    double best_cost = DBL_MAX;

    int local_alloc_fail = (!pts || !c || !best) ? 1 : 0;
    int global_alloc_fail = 0;
    MPI_Allreduce(&local_alloc_fail, &global_alloc_fail, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    if (global_alloc_fail) {
      if (rank == 0) {
        ++failed_count;
        printf("  [FAIL] allocation failure before solve\n");
        fprintf(csv,
                "openmp_mpi,%d,%d,%d,%d,%d,%.4f,failed_alloc,,,allocation_failure\n",
          size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
      }
      solution_free(best);
      matrix_free(c);
      free(pts);
      continue;
    }

    generate_points(pts, sc->n, sc->instance_seed, sc->layout_id);
    fill_cost_matrix(c, pts, sc->n);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_s = MPI_Wtime();
    aco_vrp(sc->n, sc->K, run_m, sc->T, c, 1.0, 3.0, 0.3, 1.0, 1.0,
            sc->solver_seed, best, &best_cost);
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start_s;

    int local_fail = (best_cost >= DBL_MAX / 2.0) ? 1 : 0;
    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);

    double global_best_cost = 0.0;
    MPI_Allreduce(&best_cost, &global_best_cost, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);

    if (rank == 0) {
      if (global_fail) {
        ++failed_count;
        printf("  [FAIL] solver did not produce a finite objective\n");
        fprintf(csv,
                "openmp_mpi,%d,%d,%d,%d,%d,%.4f,failed_solver,%.6f,,non_finite_objective\n",
          size, sc->n, sc->K, run_m_effective, sc->T, est_gib, elapsed);
      } else {
        ++ok_count;
        total_elapsed += elapsed;
        printf("  [OK] openmp+mpi solver completed in %7.2fs, objective=%.3f\n",
               elapsed, global_best_cost);
        fprintf(csv,
                "openmp_mpi,%d,%d,%d,%d,%d,%.4f,ok,%.6f,%.6f,\n",
          size, sc->n, sc->K, run_m_effective, sc->T, est_gib, elapsed,
                global_best_cost);
      }
    }

    solution_free(best);
    matrix_free(c);
    free(pts);
  }

  if (rank == 0) {
    fclose(csv);

    printf("\n========================================================================\n");
    printf("OpenMP+MPI Scaling Summary\n");
    printf("========================================================================\n");
    printf("[SUMMARY] scenarios_total   : %d\n", total);
    printf("[SUMMARY] ok                : %d\n", ok_count);
    printf("[SUMMARY] failed            : %d\n", failed_count);
    printf("[SUMMARY] skipped           : %d\n", skipped_count);
    printf("[SUMMARY] total_solve_s     : %.2f\n", total_elapsed);
    printf("[DONE] wrote %s\n", csv_path);
  }

  MPI_Finalize();
  return 0;
}
