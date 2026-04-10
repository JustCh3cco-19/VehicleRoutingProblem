extern "C" {
#include "aco.h"
#include "matrix.h"
#include "solution.h"
}

#include "aco_cuda_v4_kernels.h"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err__));                                    \
      return 1;                                                              \
    }                                                                        \
  } while (0)

static double wall_time_seconds(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void load_timer_directives(double *max_runtime_sec,
                                  int *max_stagnation_epochs,
                                  double *min_rel_improvement) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  const char *s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");

  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_epochs =
      (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = (s_rel && *s_rel) ? atof(s_rel) : 1e-3;

  if (*max_stagnation_epochs < 0) {
    *max_stagnation_epochs = 0;
  }
  if (*min_rel_improvement <= 0.0) {
    *min_rel_improvement = 1e-3;
  }
}

static int is_significant_improvement(double prev_best, double new_best,
                                      double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) {
    return (new_best < prev_best);
  }
  if (new_best >= prev_best - ACO_EPS) {
    return 0;
  }
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, ACO_EPS);
  return rel_gain + ACO_EPS >= min_rel_improvement;
}

static int choose_candidate_count(int n) {
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

static int choose_auto_total_ants(void) { return 256; }

static float *flatten_costs_float(double **c, int n) {
  int side = n + 1;
  size_t total = (size_t)side * (size_t)side;
  float *flat = (float *)malloc(total * sizeof(float));
  int i;
  int j;

  if (!flat) {
    return NULL;
  }

  for (i = 0; i <= n; ++i) {
    for (j = 0; j <= n; ++j) {
      flat[(size_t)i * (size_t)side + (size_t)j] = (float)c[i][j];
    }
  }

  return flat;
}

static int select_iter_best_host(const CudaV4AntSummary *summary, int m,
                                 int *best_idx, float *best_cost) {
  int found = 0;
  int idx = -1;
  float cost = FLT_MAX;
  int i;

  for (i = 0; i < m; ++i) {
    if (!summary[i].feasible) {
      continue;
    }
    if (!found || summary[i].cost < cost ||
        (fabsf(summary[i].cost - cost) <= (float)ACO_EPS && i < idx)) {
      found = 1;
      idx = i;
      cost = summary[i].cost;
    }
  }

  *best_idx = idx;
  *best_cost = cost;
  return found;
}

static int copy_ant_to_solution(const int *routes, const int *route_lengths,
                                int K, int route_stride, int ant_idx,
                                Solution *dst) {
  int v;
  solution_reset(dst);

  for (v = 0; v < K; ++v) {
    int len = route_lengths[ant_idx * K + v];
    int base = ((ant_idx * K) + v) * route_stride;
    Route *r = &dst->routes[v];

    if (len < 2 || len > r->cap) {
      return 0;
    }

    r->len = len;
    memcpy(r->nodes, routes + base, (size_t)len * sizeof(int));
  }

  return 1;
}

static int validate_cuda_solution(const Solution *sol, int n, int K, int cap,
                                  char *err, size_t err_len) {
  int *seen;
  int v;
  int p;

  if (!sol) {
    snprintf(err, err_len, "solution is NULL");
    return 0;
  }
  if (sol->K != K) {
    snprintf(err, err_len, "solution K mismatch");
    return 0;
  }

  seen = (int *)calloc((size_t)(n + 1), sizeof(int));
  if (!seen) {
    snprintf(err, err_len, "allocation failure");
    return 0;
  }

  for (v = 0; v < K; ++v) {
    const Route *r = &sol->routes[v];
    int customers = 0;

    if (r->len < 2) {
      free(seen);
      snprintf(err, err_len, "route len < 2");
      return 0;
    }
    if (r->nodes[0] != 0 || r->nodes[r->len - 1] != 0) {
      free(seen);
      snprintf(err, err_len, "route must start/end at depot");
      return 0;
    }

    for (p = 1; p + 1 < r->len; ++p) {
      int node = r->nodes[p];
      if (node <= 0 || node > n) {
        free(seen);
        snprintf(err, err_len, "invalid customer id");
        return 0;
      }
      if (seen[node]) {
        free(seen);
        snprintf(err, err_len, "duplicate customer");
        return 0;
      }
      seen[node] = 1;
      ++customers;
    }

    if (customers > cap) {
      free(seen);
      snprintf(err, err_len, "route capacity exceeded");
      return 0;
    }
  }

  for (p = 1; p <= n; ++p) {
    if (!seen[p]) {
      free(seen);
      snprintf(err, err_len, "customer not visited");
      return 0;
    }
  }

  free(seen);
  if (err && err_len > 0) {
    err[0] = '\0';
  }
  return 1;
}

typedef struct {
  double elapsed_s;
  float iter_best_cost;
  double global_best_cost;
  int improved_global;
  int feasible_ants;
  double mean_ant_cost;
  double std_ant_cost;
  CudaV4IterStats iter_stats;
  float tau_mean;
  float tau_std;
  float tau_max_value;
  float tau_max_share;
} CudaV4TelemetryRow;

static FILE *open_telemetry_csv(void) {
  const char *path = getenv("ACO_V4_TELEMETRY_CSV");
  FILE *f;

  if (!path || !*path) {
    return NULL;
  }
  f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "cuda v4: failed to open telemetry CSV: %s\n", path);
    return NULL;
  }
  fprintf(f,
          "iter,elapsed_s,iter_best_cost,global_best_cost,improved_global,"
          "feasible_ants,mean_ant_cost,std_ant_cost,candidate_moves,"
          "fallback_moves,depot_close_moves,customer_moves,nonempty_routes,"
          "tau_mean,tau_std,tau_max_value,tau_max_share\n");
  fflush(f);
  return f;
}

static void compute_ant_summary_stats(const CudaV4AntSummary *summary, int m,
                                      int *feasible_ants, double *mean_cost,
                                      double *std_cost) {
  int i;
  int feasible = 0;
  double sum = 0.0;
  double sum_sq = 0.0;

  for (i = 0; i < m; ++i) {
    if (!summary[i].feasible) {
      continue;
    }
    ++feasible;
    sum += (double)summary[i].cost;
    sum_sq += (double)summary[i].cost * (double)summary[i].cost;
  }

  *feasible_ants = feasible;
  if (feasible <= 0) {
    *mean_cost = 0.0;
    *std_cost = 0.0;
    return;
  }

  *mean_cost = sum / (double)feasible;
  {
    double var = (sum_sq / (double)feasible) - (*mean_cost * *mean_cost);
    if (var < 0.0) {
      var = 0.0;
    }
    *std_cost = sqrt(var);
  }
}

static void compute_tau_stats(const float *tau, int n, float *mean_out,
                              float *std_out, float *max_out,
                              float *max_share_out) {
  int side = n + 1;
  double sum = 0.0;
  double sum_sq = 0.0;
  float max_v = 0.0f;
  int count = 0;
  int i;
  int j;

  for (i = 0; i <= n; ++i) {
    for (j = 0; j <= n; ++j) {
      float v;
      if (i == j) {
        continue;
      }
      v = tau[(size_t)i * (size_t)side + (size_t)j];
      sum += (double)v;
      sum_sq += (double)v * (double)v;
      if (v > max_v) {
        max_v = v;
      }
      ++count;
    }
  }

  if (count <= 0 || sum <= 0.0) {
    *mean_out = 0.0f;
    *std_out = 0.0f;
    *max_out = 0.0f;
    *max_share_out = 0.0f;
    return;
  }

  *mean_out = (float)(sum / (double)count);
  {
    double var = (sum_sq / (double)count) - ((double)(*mean_out) * (double)(*mean_out));
    if (var < 0.0) {
      var = 0.0;
    }
    *std_out = (float)sqrt(var);
  }
  *max_out = max_v;
  *max_share_out = (float)((double)max_v / sum);
}

static void write_telemetry_row(FILE *f, int iter,
                                const CudaV4TelemetryRow *row) {
  if (!f || !row) {
    return;
  }
  fprintf(
      f,
      "%d,%.6f,%.6f,%.6f,%d,%d,%.6f,%.6f,%llu,%llu,%llu,%llu,%llu,%.6f,%.6f,%.6f,%.6f\n",
      iter, row->elapsed_s, row->iter_best_cost, row->global_best_cost,
      row->improved_global, row->feasible_ants, row->mean_ant_cost,
      row->std_ant_cost, row->iter_stats.candidate_moves,
      row->iter_stats.fallback_moves, row->iter_stats.depot_close_moves,
      row->iter_stats.customer_moves, row->iter_stats.nonempty_routes,
      row->tau_mean, row->tau_std, row->tau_max_value, row->tau_max_share);
  fflush(f);
}

int aco_vrp_cuda(int n, int K, int m, double **c, double alpha,
                 double beta, double rho, double tau0, double Q,
                 unsigned int seed, Solution *best_solution,
                 double *best_cost) {
  CudaV4Params params;
  double max_runtime_sec = 0.0;
  int max_stagnation_epochs = 0;
  double min_rel_improvement = 1e-3;
  double start_wall;
  int no_improve_epochs = 0;
  int stagnation_iters = 0;
  int total_m;
  int cand_k;
  int cap;
  int visited_words;
  int route_stride;
  int side;
  int stagnation_trigger;
  size_t matrix_elems;
  const double iter_deposit_weight = 0.3;
  const double global_deposit_weight = 0.7;
  float *h_costs = NULL;
  CudaV4AntSummary *h_ant_summary = NULL;
  int *h_routes = NULL;
  int *h_route_lengths = NULL;
  int *h_global_routes = NULL;
  int *h_global_route_lengths = NULL;
  float *h_tau = NULL;
  float *d_costs = NULL;
  float *d_tau = NULL;
  int *d_candidate_idx = NULL;
  float *d_eta_beta = NULL;
  int *d_routes = NULL;
  int *d_route_lengths = NULL;
  int *d_route_loads = NULL;
  int *d_curr_nodes = NULL;
  int *d_global_routes = NULL;
  int *d_global_route_lengths = NULL;
  uint64_t *d_visited = NULL;
  unsigned int *d_rng_states = NULL;
  CudaV4AntSummary *d_ant_summary = NULL;
  CudaV4IterStats *d_iter_stats = NULL;
  CudaV4IterStats h_iter_stats;
  Solution *iter_best_solution = NULL;
  FILE *telemetry_csv = NULL;
  int iter;
  int status = 1;
  int have_global_best = 0;

  load_timer_directives(&max_runtime_sec, &max_stagnation_epochs,
                        &min_rel_improvement);

  total_m = (m > 0) ? m : choose_auto_total_ants();
  cand_k = choose_candidate_count(n);
  cap = n - K + 3;
  if (cap < 1) {
    cap = 1;
  }
  visited_words = (n / 64) + 1;
  route_stride = cap + 2;
  side = n + 1;
  matrix_elems = (size_t)side * (size_t)side;
  stagnation_trigger =
      (max_stagnation_epochs > 0) ? (max_stagnation_epochs / 2) : 32;
  if (stagnation_trigger < 4) {
    stagnation_trigger = 4;
  }

  memset(&params, 0, sizeof(params));
  params.n = n;
  params.K = K;
  params.m = total_m;
  params.cap = cap;
  params.cand_k = cand_k;
  params.visited_words = visited_words;
  params.route_stride = route_stride;
  params.alpha = (float)alpha;
  params.beta = (float)beta;
  params.rho = (float)rho;
  params.tau0 = (float)tau0;
  params.Q = (float)Q;
  params.tau_max = (float)tau0;
  params.tau_min = (float)(tau0 * 0.05);
  params.depot_close_weight = 0.25f;

  h_costs = flatten_costs_float(c, n);
  h_ant_summary =
      (CudaV4AntSummary *)malloc((size_t)total_m * sizeof(*h_ant_summary));
  h_routes = (int *)malloc((size_t)total_m * (size_t)K * (size_t)route_stride *
                           sizeof(int));
  h_route_lengths =
      (int *)malloc((size_t)total_m * (size_t)K * sizeof(int));
  h_global_routes =
      (int *)malloc((size_t)K * (size_t)route_stride * sizeof(int));
  h_global_route_lengths = (int *)malloc((size_t)K * sizeof(int));
  h_tau = (float *)malloc(matrix_elems * sizeof(float));
  iter_best_solution = solution_create(K, n);
  telemetry_csv = open_telemetry_csv();

  if (!h_costs || !h_ant_summary || !h_routes || !h_route_lengths ||
      !h_global_routes || !h_global_route_lengths || !h_tau ||
      !iter_best_solution) {
    fprintf(stderr, "cuda v4: host allocation failure\n");
    goto cleanup;
  }

  if (cudaMalloc((void **)&d_costs, matrix_elems * sizeof(float)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_tau, matrix_elems * sizeof(float)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_candidate_idx,
                 (size_t)side * (size_t)cand_k * sizeof(int)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_eta_beta,
                 (size_t)side * (size_t)cand_k * sizeof(float)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_routes,
                 (size_t)total_m * (size_t)K * (size_t)route_stride *
                     sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&d_route_lengths,
                 (size_t)total_m * (size_t)K * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&d_route_loads,
                 (size_t)total_m * (size_t)K * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&d_curr_nodes,
                 (size_t)total_m * (size_t)K * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&d_global_routes,
                 (size_t)K * (size_t)route_stride * sizeof(int)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_global_route_lengths,
                 (size_t)K * sizeof(int)) != cudaSuccess ||
      cudaMalloc((void **)&d_visited,
                 (size_t)total_m * (size_t)visited_words * sizeof(uint64_t)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_rng_states,
                 (size_t)total_m * sizeof(unsigned int)) != cudaSuccess ||
      cudaMalloc((void **)&d_iter_stats, sizeof(CudaV4IterStats)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_ant_summary,
                 (size_t)total_m * sizeof(CudaV4AntSummary)) != cudaSuccess) {
    fprintf(stderr, "cuda v4: device allocation failure\n");
    goto cleanup;
  }

  CHECK_CUDA(cudaMemcpy(d_costs, h_costs, matrix_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  launch_build_candidate_lists_v4(d_costs, d_candidate_idx, d_eta_beta, n,
                                  cand_k, (float)beta);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_init_tau_v4(d_tau, n, (float)tau0);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  solution_reset(best_solution);
  *best_cost = DBL_MAX;
  start_wall = wall_time_seconds();

  for (iter = 0;; ++iter) {
    int iter_best_idx = -1;
    float iter_best_cost_f = FLT_MAX;
    int improved_global = 0;
    char errbuf[128];
    CudaV4TelemetryRow telemetry_row;

    memset(&telemetry_row, 0, sizeof(telemetry_row));

    if (max_runtime_sec > 0.0 &&
        (wall_time_seconds() - start_wall) >= max_runtime_sec) {
      break;
    }

    launch_reset_ant_state_v4(d_routes, d_route_lengths, d_route_loads,
                              d_curr_nodes, d_visited, d_rng_states,
                              d_ant_summary, params, seed + (unsigned int)iter);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemset(d_iter_stats, 0, sizeof(CudaV4IterStats)));

    launch_construct_solutions_v4(d_costs, d_tau, d_candidate_idx, d_eta_beta,
                                  d_routes, d_route_lengths, d_route_loads,
                                  d_curr_nodes, d_visited, d_rng_states,
                                  d_ant_summary, d_iter_stats, params);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ant_summary, d_ant_summary,
                          (size_t)total_m * sizeof(CudaV4AntSummary),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&h_iter_stats, d_iter_stats, sizeof(CudaV4IterStats),
                          cudaMemcpyDeviceToHost));
    telemetry_row.iter_stats = h_iter_stats;
    compute_ant_summary_stats(h_ant_summary, total_m, &telemetry_row.feasible_ants,
                              &telemetry_row.mean_ant_cost,
                              &telemetry_row.std_ant_cost);

    if (!select_iter_best_host(h_ant_summary, total_m, &iter_best_idx,
                               &iter_best_cost_f)) {
      ++stagnation_iters;
      ++no_improve_epochs;
      if (stagnation_iters >= stagnation_trigger) {
        launch_recenter_tau_v4(d_tau, n, params.tau0);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        stagnation_iters = 0;
      }
      if (max_stagnation_epochs > 0 &&
          no_improve_epochs >= max_stagnation_epochs) {
        break;
      }
      if (telemetry_csv) {
        telemetry_row.elapsed_s = wall_time_seconds() - start_wall;
        telemetry_row.iter_best_cost = -1.0f;
        telemetry_row.global_best_cost =
            (*best_cost < DBL_MAX / 2.0) ? *best_cost : -1.0;
        CHECK_CUDA(cudaMemcpy(h_tau, d_tau, matrix_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));
        compute_tau_stats(h_tau, n, &telemetry_row.tau_mean,
                          &telemetry_row.tau_std, &telemetry_row.tau_max_value,
                          &telemetry_row.tau_max_share);
        write_telemetry_row(telemetry_csv, iter, &telemetry_row);
      }
      continue;
    }

    CHECK_CUDA(cudaMemcpy(h_routes, d_routes,
                          (size_t)total_m * (size_t)K * (size_t)route_stride *
                              sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_route_lengths, d_route_lengths,
                          (size_t)total_m * (size_t)K * sizeof(int),
                          cudaMemcpyDeviceToHost));

    if (!copy_ant_to_solution(h_routes, h_route_lengths, K, route_stride,
                              iter_best_idx, iter_best_solution)) {
      fprintf(stderr, "cuda v4: failed to copy iter best solution\n");
      goto cleanup;
    }

    if (!validate_cuda_solution(iter_best_solution, n, K, cap, errbuf,
                                sizeof(errbuf))) {
      fprintf(stderr, "cuda v4: invalid iter best solution: %s\n", errbuf);
      goto cleanup;
    }

    if (is_significant_improvement(*best_cost, (double)iter_best_cost_f,
                                   min_rel_improvement)) {
      *best_cost = (double)iter_best_cost_f;
      solution_copy(best_solution, iter_best_solution);
      {
        size_t best_offset = (size_t)iter_best_idx * (size_t)K *
                             (size_t)route_stride;
        memcpy(h_global_routes, h_routes + best_offset,
               (size_t)K * (size_t)route_stride * sizeof(int));
        memcpy(h_global_route_lengths,
               h_route_lengths + (size_t)iter_best_idx * (size_t)K,
               (size_t)K * sizeof(int));
      }
      CHECK_CUDA(cudaMemcpy(d_global_routes, h_global_routes,
                            (size_t)K * (size_t)route_stride * sizeof(int),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_global_route_lengths, h_global_route_lengths,
                            (size_t)K * sizeof(int), cudaMemcpyHostToDevice));
      have_global_best = 1;
      no_improve_epochs = 0;
      stagnation_iters = 0;
      improved_global = 1;
      if (*best_cost > ACO_EPS) {
        params.tau_max = (float)(1.0 / ((1.0 - rho) * (*best_cost)));
        params.tau_min = params.tau_max * 0.05f;
      }
    } else {
      ++no_improve_epochs;
      ++stagnation_iters;
    }

    launch_evaporate_tau_v4(d_tau, n, (float)rho);
    CHECK_CUDA(cudaGetLastError());

    launch_deposit_solution_v4(
        d_tau,
        d_routes + ((size_t)iter_best_idx * (size_t)K * (size_t)route_stride),
        d_route_lengths + ((size_t)iter_best_idx * (size_t)K), K, route_stride,
        n, (float)((iter_deposit_weight * Q) / (double)iter_best_cost_f));
    CHECK_CUDA(cudaGetLastError());

    if (have_global_best && *best_cost < DBL_MAX) {
      launch_deposit_solution_v4(
          d_tau, d_global_routes, d_global_route_lengths, K, route_stride, n,
          (float)((global_deposit_weight * Q) / (*best_cost)));
      CHECK_CUDA(cudaGetLastError());
    }

    launch_clamp_tau_v4(d_tau, n, params.tau_min, params.tau_max);
    CHECK_CUDA(cudaGetLastError());

    if (stagnation_iters >= stagnation_trigger) {
      launch_recenter_tau_v4(d_tau, n, params.tau0);
      CHECK_CUDA(cudaGetLastError());
      stagnation_iters = 0;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    if (telemetry_csv) {
      telemetry_row.elapsed_s = wall_time_seconds() - start_wall;
      telemetry_row.iter_best_cost = iter_best_cost_f;
      telemetry_row.global_best_cost =
          (*best_cost < DBL_MAX / 2.0) ? *best_cost : -1.0;
      telemetry_row.improved_global = improved_global;
      CHECK_CUDA(cudaMemcpy(h_tau, d_tau, matrix_elems * sizeof(float),
                            cudaMemcpyDeviceToHost));
      compute_tau_stats(h_tau, n, &telemetry_row.tau_mean,
                        &telemetry_row.tau_std, &telemetry_row.tau_max_value,
                        &telemetry_row.tau_max_share);
      write_telemetry_row(telemetry_csv, iter, &telemetry_row);
    }

    if (max_runtime_sec > 0.0 &&
        (wall_time_seconds() - start_wall) >= max_runtime_sec) {
      break;
    }
    if (!improved_global && max_stagnation_epochs > 0 &&
        no_improve_epochs >= max_stagnation_epochs) {
      break;
    }
  }

  status = 0;

cleanup:
  if (iter_best_solution) {
    solution_free(iter_best_solution);
  }
  free(h_costs);
  free(h_ant_summary);
  free(h_routes);
  free(h_route_lengths);
  free(h_global_routes);
  free(h_global_route_lengths);
  free(h_tau);
  if (telemetry_csv) {
    fclose(telemetry_csv);
  }
  if (d_costs) {
    cudaFree(d_costs);
  }
  if (d_tau) {
    cudaFree(d_tau);
  }
  if (d_candidate_idx) {
    cudaFree(d_candidate_idx);
  }
  if (d_eta_beta) {
    cudaFree(d_eta_beta);
  }
  if (d_routes) {
    cudaFree(d_routes);
  }
  if (d_route_lengths) {
    cudaFree(d_route_lengths);
  }
  if (d_route_loads) {
    cudaFree(d_route_loads);
  }
  if (d_curr_nodes) {
    cudaFree(d_curr_nodes);
  }
  if (d_global_routes) {
    cudaFree(d_global_routes);
  }
  if (d_global_route_lengths) {
    cudaFree(d_global_route_lengths);
  }
  if (d_visited) {
    cudaFree(d_visited);
  }
  if (d_rng_states) {
    cudaFree(d_rng_states);
  }
  if (d_iter_stats) {
    cudaFree(d_iter_stats);
  }
  if (d_ant_summary) {
    cudaFree(d_ant_summary);
  }
  return status;
}
