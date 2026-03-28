#include "aco.h"
#include "matrix.h"
#include "solution.h"
#include "test_types.h"

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
 * Function:  bytes_to_gib
 * ------------------------
 * converts a byte count to gibibytes.
 *
 *  x: input parameter
 *
 *  returns: value as defined by the function contract
 */
static double bytes_to_gib(size_t x) {
  return (double)x / (1024.0 * 1024.0 * 1024.0);
}

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
 * Function:  estimate_runtime_seconds
 * ------------------------
 * rough runtime estimate for one scenario with safety factor.
 */
static double estimate_runtime_seconds(int n, int m_effective, int T,
                                       int mpi_ranks, double safety) {
  int omp_threads = 1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif
  int workers = mpi_ranks * (omp_threads > 0 ? omp_threads : 1);
  if (workers < 1) {
    workers = 1;
  }
  if (m_effective < 1) {
    m_effective = 1;
  }
  if (T < 1) {
    T = 1;
  }
  if (safety < 1.0) {
    safety = 1.0;
  }

  /* Conservative calibration for OpenMP+MPI ACO on cluster runs. */
  const double coeff = 4.5e-8;
  double est = coeff * (double)n * (double)n * (double)m_effective *
               (double)T / (double)workers;
  if (est < 1.0) {
    est = 1.0;
  }
  return est * safety;
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

/* Mirrors solver auto-ant policy only for user-facing reporting in this runner. */
/*
 * Function:  estimate_auto_ants_for_log
 * ------------------------
 * mirrors solver auto-ant policy for user-facing logs.
 *
 *  n: input parameter
 *  mpi_ranks: input parameter
 *
 *  returns: value as defined by the function contract
 */
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
  static const int levels[] = {24000, 32000, 40000, 48000, 56000,
                               64000, 72000, 80000, 90000, 100000};
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

static int file_exists_nonempty(const char *path) {
  struct stat st;
  if (!path) {
    return 0;
  }
  if (stat(path, &st) != 0) {
    return 0;
  }
  return st.st_size > 0 ? 1 : 0;
}

typedef struct {
  int next_index;
  int ok_count;
  int skipped_count;
  int failed_count;
  double total_elapsed;
} CheckpointState;

static int checkpoint_load(const char *path, CheckpointState *state, int total) {
  FILE *fp = fopen(path, "r");
  if (!fp) {
    return 0;
  }
  CheckpointState tmp = {0, 0, 0, 0, 0.0};
  if (fscanf(fp, "next_index=%d\n", &tmp.next_index) != 1 ||
      fscanf(fp, "ok_count=%d\n", &tmp.ok_count) != 1 ||
      fscanf(fp, "skipped_count=%d\n", &tmp.skipped_count) != 1 ||
      fscanf(fp, "failed_count=%d\n", &tmp.failed_count) != 1 ||
      fscanf(fp, "total_elapsed=%lf\n", &tmp.total_elapsed) != 1) {
    fclose(fp);
    return 0;
  }
  fclose(fp);

  tmp.next_index = clamp_int(tmp.next_index, 0, total);
  if (tmp.ok_count < 0) tmp.ok_count = 0;
  if (tmp.skipped_count < 0) tmp.skipped_count = 0;
  if (tmp.failed_count < 0) tmp.failed_count = 0;
  if (!(tmp.total_elapsed >= 0.0)) tmp.total_elapsed = 0.0;
  *state = tmp;
  return 1;
}

static void checkpoint_save(const char *path, const CheckpointState *state) {
  FILE *fp = fopen(path, "w");
  if (!fp) {
    return;
  }
  fprintf(fp, "next_index=%d\n", state->next_index);
  fprintf(fp, "ok_count=%d\n", state->ok_count);
  fprintf(fp, "skipped_count=%d\n", state->skipped_count);
  fprintf(fp, "failed_count=%d\n", state->failed_count);
  fprintf(fp, "total_elapsed=%.9f\n", state->total_elapsed);
  fclose(fp);
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

  int rank = 0;
  int size = 1;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const char *csv_path = "results/scaling_progressive_openmp_mpi_heavy.csv";
  const char *input_log_path = "results/openmp_mpi_test_inputs_heavy.log";
  const char *checkpoint_path = "results/openmp_mpi_tests_heavy.checkpoint";
  double memory_utilization = 0.70;
  double time_budget_minutes = 30.0;
  double estimate_safety = 1.25;
  int c_max_n = 100000;
  int enforce_c_max_n = 0;
  int force = 0;
  int auto_ants = 1;
  int resume = 0;
  int reset_checkpoint = 0;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csv_path = argv[++i];
    } else if (strcmp(argv[i], "--input-log") == 0 && i + 1 < argc) {
      input_log_path = argv[++i];
    } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
      checkpoint_path = argv[++i];
    } else if (strcmp(argv[i], "--resume") == 0) {
      resume = 1;
    } else if (strcmp(argv[i], "--reset-checkpoint") == 0) {
      reset_checkpoint = 1;
    } else if (strcmp(argv[i], "--memory-utilization") == 0 && i + 1 < argc) {
      memory_utilization = atof(argv[++i]);
      if (memory_utilization <= 0.0 || memory_utilization > 1.0) {
        if (rank == 0) {
          fprintf(stderr, "invalid --memory-utilization, expected (0, 1].\n");
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--time-budget-minutes") == 0 && i + 1 < argc) {
      time_budget_minutes = atof(argv[++i]);
      if (time_budget_minutes <= 0.0) {
        if (rank == 0) {
          fprintf(stderr,
                  "invalid --time-budget-minutes, expected positive number.\n");
        }
        MPI_Finalize();
        return 1;
      }
    } else if (strcmp(argv[i], "--estimate-safety") == 0 && i + 1 < argc) {
      estimate_safety = atof(argv[++i]);
      if (estimate_safety <= 0.0) {
        if (rank == 0) {
          fprintf(stderr, "invalid --estimate-safety, expected positive number.\n");
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
          "usage: %s [--csv PATH] [--input-log PATH] [--checkpoint PATH] [--resume] [--reset-checkpoint] [--memory-utilization X] [--time-budget-minutes X] [--estimate-safety X] [--c-max-n N] [--enforce-c-max-n] [--force] [--auto-ants|--fixed-ants]\n",
                argv[0]);
      }
      MPI_Finalize();
      return 1;
    }
  }

  FILE *csv = NULL;
  FILE *input_log = NULL;
  int append_mode = 0;
  if (reset_checkpoint) {
    resume = 0;
  }
  if (rank == 0) {
    if (reset_checkpoint) {
      remove(checkpoint_path);
    }
    if (!ensure_parent_dir(csv_path)) {
      fprintf(stderr, "failed to create parent directory for CSV: %s\n", csv_path);
      MPI_Finalize();
      return 1;
    }
    if (!ensure_parent_dir(input_log_path)) {
      fprintf(stderr, "failed to create parent directory for input log: %s\n",
              input_log_path);
      MPI_Finalize();
      return 1;
    }

    append_mode = resume && file_exists_nonempty(csv_path) ? 1 : 0;
    csv = fopen(csv_path, append_mode ? "a" : "w");
    if (!csv) {
      fprintf(stderr, "failed to open output CSV: %s\n", csv_path);
      MPI_Finalize();
      return 1;
    }
    input_log = fopen(input_log_path,
                      (resume && file_exists_nonempty(input_log_path)) ? "a"
                                                                       : "w");
    if (!input_log) {
      fprintf(stderr, "failed to open input log: %s\n", input_log_path);
      fclose(csv);
      MPI_Finalize();
      return 1;
    }
    setvbuf(csv, NULL, _IOLBF, 0);
    setvbuf(input_log, NULL, _IOLBF, 0);

    if (!append_mode) {
      fprintf(csv,
              "mode,mpi_ranks,n,K,m,T,estimated_mem_gib,status,elapsed_s,cost,error\n");
      fflush(csv);
    }
  }

  size_t available = read_available_memory_bytes();
  size_t threshold = (size_t)((double)available * memory_utilization);

  if (rank == 0) {
    int omp_threads = 1;
  #ifdef _OPENMP
    omp_threads = omp_get_max_threads();
  #endif
    printf("========================================================================\n");
    printf("OpenMP+MPI Progressive Scaling (heavy > 16k)\n");
    printf("========================================================================\n");
    printf("[INFO] mpi_ranks            : %d\n", size);
    printf("[INFO] omp_threads/rank     : %d\n", omp_threads);
        printf("[INFO] ants_mode            : %s\n",
          auto_ants ? "auto(solver,m=0)" : "fixed(scenario)");
    printf("[INFO] available_mem_gib    : %.2f\n", bytes_to_gib(available));
    printf("[INFO] threshold_gib        : %.2f\n", bytes_to_gib(threshold));
    printf("[INFO] memory_utilization   : %.2f\n", memory_utilization);
    printf("[INFO] c_max_n              : %d\n\n", c_max_n);
    printf("[INFO] time_budget_minutes  : %.2f\n", time_budget_minutes);
    printf("[INFO] estimate_safety      : %.2f\n\n", estimate_safety);
    printf("[INFO] checkpoint_path      : %s\n", checkpoint_path);
    printf("[INFO] resume               : %s\n\n", resume ? "yes" : "no");
    printf("[INFO] reset_checkpoint     : %s\n\n",
           reset_checkpoint ? "yes" : "no");
    printf("[INFO] enforce_c_max_n      : %s\n", enforce_c_max_n ? "yes" : "no");
    printf("[INFO] csv_path             : %s\n", csv_path);
    printf("[INFO] input_log_path       : %s\n", input_log_path);
    printf("[INFO] command              : ");
    print_command_line(stdout, argc, argv);
    printf("\n");

    if (!append_mode) {
      fprintf(input_log, "# openmp_mpi_tests input log\n");
    } else {
      fprintf(input_log, "\n# resume session\n");
    }
    fprintf(input_log, "command: ");
    print_command_line(input_log, argc, argv);
    fprintf(input_log, "mpi_ranks=%d\n", size);
    fprintf(input_log, "csv_path=%s\n", csv_path);
    fprintf(input_log, "memory_utilization=%.6f\n", memory_utilization);
    fprintf(input_log, "c_max_n=%d\n", c_max_n);
    fprintf(input_log, "enforce_c_max_n=%d\n", enforce_c_max_n);
    fprintf(input_log, "force=%d\n", force);
    fprintf(input_log, "auto_ants=%d\n", auto_ants);
    fprintf(input_log, "time_budget_minutes=%.6f\n", time_budget_minutes);
    fprintf(input_log, "estimate_safety=%.6f\n", estimate_safety);
    fprintf(input_log, "checkpoint_path=%s\n", checkpoint_path);
    fprintf(input_log, "resume=%d\n", resume);
    fprintf(input_log, "reset_checkpoint=%d\n", reset_checkpoint);
    fprintf(input_log, "available_mem_gib=%.4f\n", bytes_to_gib(available));
    fprintf(input_log, "threshold_gib=%.4f\n\n", bytes_to_gib(threshold));
    fflush(input_log);
  }

  Scenario scenarios[24];
  int total = 0;
  build_scenarios(scenarios, &total);
  CheckpointState cp = {0, 0, 0, 0, 0.0};
  if (rank == 0 && resume) {
    checkpoint_load(checkpoint_path, &cp, total);
  }
  int cp_i[4] = {0, 0, 0, 0};
  double cp_elapsed = 0.0;
  if (rank == 0) {
    cp_i[0] = cp.next_index;
    cp_i[1] = cp.ok_count;
    cp_i[2] = cp.skipped_count;
    cp_i[3] = cp.failed_count;
    cp_elapsed = cp.total_elapsed;
  }
  MPI_Bcast(cp_i, 4, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cp_elapsed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  cp.next_index = cp_i[0];
  cp.ok_count = cp_i[1];
  cp.skipped_count = cp_i[2];
  cp.failed_count = cp_i[3];
  cp.total_elapsed = cp_elapsed;

  if (rank == 0 && resume) {
    printf("[INFO] resume_next_index    : %d\n", cp.next_index);
    printf("[INFO] resume_ok/sk/fail    : %d/%d/%d\n",
           cp.ok_count, cp.skipped_count, cp.failed_count);
    printf("[INFO] resume_elapsed_s     : %.2f\n\n", cp.total_elapsed);
  }

  double estimated_total_s = 0.0;
  for (int i = cp.next_index; i < total; ++i) {
    int m_effective = auto_ants ? estimate_auto_ants_for_log(scenarios[i].n, size)
                                : scenarios[i].m;
    estimated_total_s += estimate_runtime_seconds(
      scenarios[i].n, m_effective, scenarios[i].T, size, estimate_safety);
  }
  if (rank == 0) {
    printf("[INFO] estimated_total_s    : %.1f\n", estimated_total_s);
    printf("[INFO] estimated_total_min  : %.1f\n", estimated_total_s / 60.0);
    if (estimated_total_s > time_budget_minutes * 60.0) {
      printf("[WARN] estimated total exceeds budget (%.1f > %.1f sec)\n",
             estimated_total_s, time_budget_minutes * 60.0);
    }
    printf("\n");
  }

  int ok_count = cp.ok_count;
  int skipped_count = cp.skipped_count;
  int failed_count = cp.failed_count;
  double total_elapsed = cp.total_elapsed;

  for (int i = cp.next_index; i < total; ++i) {
    const Scenario *sc = &scenarios[i];
    int run_m = auto_ants ? 0 : sc->m;
    int run_m_effective = auto_ants ? estimate_auto_ants_for_log(sc->n, size) : sc->m;
    size_t est_mem = estimate_c_memory_bytes(sc->n);
    double est_gib = bytes_to_gib(est_mem);
    double est_runtime_s = estimate_runtime_seconds(
      sc->n, run_m_effective, sc->T, size, estimate_safety);

    if (rank == 0) {
      printf("[RUN  %02d/%02d] customers=%-6d vehicles=%-4d ants=%-3d iterations=%-3d est_mem=%6.2f GiB\n",
              i + 1, total, sc->n, sc->K, run_m_effective, sc->T, est_gib);
      printf("  [TIME ] est_runtime=%.1fs\n", est_runtime_s);
      printf("  [INPUT] solver   : n=%d K=%d m=%d T=%d alpha=1.0 beta=3.0 rho=0.3 tau0=1.0 Q=1.0 seed=%u\n",
             sc->n, sc->K, run_m, sc->T, sc->solver_seed);
      printf("  [INPUT] instance : seed=%u layout_id=%d mpi_ranks=%d\n",
             sc->instance_seed, sc->layout_id, size);
      fprintf(input_log,
              "[RUN %02d/%02d] mpi_ranks=%d n=%d K=%d m_passed=%d m_effective=%d T=%d alpha=1.0 beta=3.0 rho=0.3 tau0=1.0 Q=1.0 solver_seed=%u instance_seed=%u layout_id=%d est_mem_gib=%.4f\n",
              i + 1, total, size, sc->n, sc->K, run_m, run_m_effective, sc->T,
              sc->solver_seed, sc->instance_seed, sc->layout_id, est_gib);
    }

    if (enforce_c_max_n && sc->n > c_max_n) {
      if (rank == 0) {
        ++skipped_count;
        printf("  [SKIP] n=%d supera c-max-n=%d\n", sc->n, c_max_n);
        fprintf(csv,
                "openmp_mpi_heavy,%d,%d,%d,%d,%d,%.4f,skipped_c_max_n,,,\n",
                size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
        fprintf(input_log, "  result=skipped_c_max_n c_max_n=%d\n\n", c_max_n);
        fflush(csv);
        fflush(input_log);
        cp.next_index = i + 1;
        cp.ok_count = ok_count;
        cp.skipped_count = skipped_count;
        cp.failed_count = failed_count;
        cp.total_elapsed = total_elapsed;
        checkpoint_save(checkpoint_path, &cp);
      }
      continue;
    }

    if (!force && threshold > 0 && est_mem > threshold) {
      if (rank == 0) {
        ++skipped_count;
        printf("  [SKIP] memoria stimata %.2f GiB > soglia %.2f GiB\n",
               est_gib, bytes_to_gib(threshold));
        fprintf(csv,
                "openmp_mpi_heavy,%d,%d,%d,%d,%d,%.4f,skipped_memory,,,\n",
                size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
        fprintf(input_log,
                "  result=skipped_memory est_mem_gib=%.4f threshold_gib=%.4f\n\n",
                est_gib, bytes_to_gib(threshold));
        fflush(csv);
        fflush(input_log);
        cp.next_index = i + 1;
        cp.ok_count = ok_count;
        cp.skipped_count = skipped_count;
        cp.failed_count = failed_count;
        cp.total_elapsed = total_elapsed;
        checkpoint_save(checkpoint_path, &cp);
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
                "openmp_mpi_heavy,%d,%d,%d,%d,%d,%.4f,failed_alloc,,,allocation_failure\n",
                size, sc->n, sc->K, run_m_effective, sc->T, est_gib);
        fprintf(input_log, "  result=failed_alloc error=allocation_failure\n\n");
        fflush(csv);
        fflush(input_log);
        cp.next_index = i + 1;
        cp.ok_count = ok_count;
        cp.skipped_count = skipped_count;
        cp.failed_count = failed_count;
        cp.total_elapsed = total_elapsed;
        checkpoint_save(checkpoint_path, &cp);
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
                "openmp_mpi_heavy,%d,%d,%d,%d,%d,%.4f,failed_solver,%.6f,,non_finite_objective\n",
                size, sc->n, sc->K, run_m_effective, sc->T, est_gib, elapsed);
        fprintf(input_log,
                "  result=failed_solver elapsed_s=%.6f error=non_finite_objective\n\n",
                elapsed);
        fflush(csv);
        fflush(input_log);
      } else {
        ++ok_count;
        total_elapsed += elapsed;
        printf("  [OK] openmp+mpi solver completed in %7.2fs, objective=%.3f\n",
               elapsed, global_best_cost);
        fprintf(csv,
                "openmp_mpi_heavy,%d,%d,%d,%d,%d,%.4f,ok,%.6f,%.6f,\n",
                size, sc->n, sc->K, run_m_effective, sc->T, est_gib, elapsed,
                global_best_cost);
        fprintf(input_log,
                "  result=ok elapsed_s=%.6f best_cost=%.6f\n\n", elapsed,
                global_best_cost);
        fflush(csv);
        fflush(input_log);
      }
      cp.next_index = i + 1;
      cp.ok_count = ok_count;
      cp.skipped_count = skipped_count;
      cp.failed_count = failed_count;
      cp.total_elapsed = total_elapsed;
      checkpoint_save(checkpoint_path, &cp);
    }

    solution_free(best);
    matrix_free(c);
    free(pts);
  }

  if (rank == 0) {
    fclose(csv);
    fclose(input_log);

    printf("\n========================================================================\n");
    printf("OpenMP+MPI Scaling Summary\n");
    printf("========================================================================\n");
    printf("[SUMMARY] scenarios_total   : %d\n", total);
    printf("[SUMMARY] ok                : %d\n", ok_count);
    printf("[SUMMARY] failed            : %d\n", failed_count);
    printf("[SUMMARY] skipped           : %d\n", skipped_count);
    printf("[SUMMARY] total_solve_s     : %.2f\n", total_elapsed);
    printf("[DONE] wrote %s\n", csv_path);
    printf("[DONE] wrote %s\n", input_log_path);
  }

  MPI_Finalize();
  return 0;
}
