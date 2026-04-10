extern "C" {
#include "aco.h"
#include "matrix.h"
#include "solution.h"
}

#include "aco_cuda_v1_kernels.h"

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

static int select_iter_best_host(const CudaV1AntSummary *summary, int m,
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

int aco_vrp_cuda_with_capacity(int n, int K, int vehicle_capacity_customers,
                               int m, double **c, double alpha, double beta,
                               double rho, double tau0, double Q,
                               unsigned int seed, Solution *best_solution,
                               double *best_cost) {
  CudaV1Params params;
  double max_runtime_sec = 0.0;
  int max_stagnation_epochs = 0;
  double min_rel_improvement = 1e-3;
  double start_wall;
  int no_improve_epochs = 0;
  int total_m;
  int cand_k;
  int cap;
  int visited_words;
  int route_stride;
  int side;
  size_t matrix_elems;
  float *h_costs = NULL;
  CudaV1AntSummary *h_ant_summary = NULL;
  int *h_routes = NULL;
  int *h_route_lengths = NULL;
  float *d_costs = NULL;
  float *d_tau = NULL;
  int *d_candidate_idx = NULL;
  float *d_eta_beta = NULL;
  int *d_routes = NULL;
  int *d_route_lengths = NULL;
  int *d_route_loads = NULL;
  int *d_curr_nodes = NULL;
  uint64_t *d_visited = NULL;
  unsigned int *d_rng_states = NULL;
  CudaV1AntSummary *d_ant_summary = NULL;
  Solution *iter_best_solution = NULL;
  int iter;
  int status = 1;

  load_timer_directives(&max_runtime_sec, &max_stagnation_epochs,
                        &min_rel_improvement);

  total_m = (m > 0) ? m : choose_auto_total_ants();
  cand_k = choose_candidate_count(n);
  cap = vehicle_capacity_customers;
  if (cap < 1) {
    cap = 1;
  }
  visited_words = (n / 64) + 1;
  route_stride = cap + 2;
  side = n + 1;
  matrix_elems = (size_t)side * (size_t)side;

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

  h_costs = flatten_costs_float(c, n);
  h_ant_summary =
      (CudaV1AntSummary *)malloc((size_t)total_m * sizeof(*h_ant_summary));
  h_routes = (int *)malloc((size_t)total_m * (size_t)K * (size_t)route_stride *
                           sizeof(int));
  h_route_lengths =
      (int *)malloc((size_t)total_m * (size_t)K * sizeof(int));
  iter_best_solution = solution_create(K, n);

  if (!h_costs || !h_ant_summary || !h_routes || !h_route_lengths ||
      !iter_best_solution) {
    fprintf(stderr, "cuda v1: host allocation failure\n");
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
      cudaMalloc((void **)&d_visited,
                 (size_t)total_m * (size_t)visited_words * sizeof(uint64_t)) !=
          cudaSuccess ||
      cudaMalloc((void **)&d_rng_states,
                 (size_t)total_m * sizeof(unsigned int)) != cudaSuccess ||
      cudaMalloc((void **)&d_ant_summary,
                 (size_t)total_m * sizeof(CudaV1AntSummary)) != cudaSuccess) {
    fprintf(stderr, "cuda v1: device allocation failure\n");
    goto cleanup;
  }

  CHECK_CUDA(cudaMemcpy(d_costs, h_costs, matrix_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  launch_build_candidate_lists_v1(d_costs, d_candidate_idx, d_eta_beta, n,
                                  cand_k, (float)beta);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  launch_init_tau_v1(d_tau, n, (float)tau0);
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

    if (max_runtime_sec > 0.0 &&
        (wall_time_seconds() - start_wall) >= max_runtime_sec) {
      break;
    }

    launch_reset_ant_state_v1(d_routes, d_route_lengths, d_route_loads,
                              d_curr_nodes, d_visited, d_rng_states,
                              d_ant_summary, params, seed + (unsigned int)iter);
    CHECK_CUDA(cudaGetLastError());

    launch_construct_solutions_v1(d_costs, d_tau, d_candidate_idx, d_eta_beta,
                                  d_routes, d_route_lengths, d_route_loads,
                                  d_curr_nodes, d_visited, d_rng_states,
                                  d_ant_summary, params);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ant_summary, d_ant_summary,
                          (size_t)total_m * sizeof(CudaV1AntSummary),
                          cudaMemcpyDeviceToHost));

    if (!select_iter_best_host(h_ant_summary, total_m, &iter_best_idx,
                               &iter_best_cost_f)) {
      ++no_improve_epochs;
      if (max_stagnation_epochs > 0 &&
          no_improve_epochs >= max_stagnation_epochs) {
        break;
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
      fprintf(stderr, "cuda v1: failed to copy iter best solution\n");
      goto cleanup;
    }

    if (!validate_cuda_solution(iter_best_solution, n, K, cap, errbuf,
                                sizeof(errbuf))) {
      fprintf(stderr, "cuda v1: invalid iter best solution: %s\n", errbuf);
      goto cleanup;
    }

    if (is_significant_improvement(*best_cost, (double)iter_best_cost_f,
                                   min_rel_improvement)) {
      *best_cost = (double)iter_best_cost_f;
      solution_copy(best_solution, iter_best_solution);
      no_improve_epochs = 0;
      improved_global = 1;
      if (*best_cost > ACO_EPS) {
        params.tau_max = (float)(1.0 / ((1.0 - rho) * (*best_cost)));
        params.tau_min = params.tau_max * 0.05f;
      }
    } else {
      ++no_improve_epochs;
    }

    launch_evaporate_tau_v1(d_tau, n, (float)rho);
    CHECK_CUDA(cudaGetLastError());

    launch_deposit_solution_v1(
        d_tau,
        d_routes + ((size_t)iter_best_idx * (size_t)K * (size_t)route_stride),
        d_route_lengths + ((size_t)iter_best_idx * (size_t)K), K, route_stride,
        n, (float)(Q / (double)iter_best_cost_f));
    CHECK_CUDA(cudaGetLastError());

    launch_clamp_tau_v1(d_tau, n, params.tau_min, params.tau_max);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

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
  if (d_visited) {
    cudaFree(d_visited);
  }
  if (d_rng_states) {
    cudaFree(d_rng_states);
  }
  if (d_ant_summary) {
    cudaFree(d_ant_summary);
  }
  return status;
}

int aco_vrp_cuda(int n, int K, int m, double **c, double alpha, double beta,
                 double rho, double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost) {
  int vehicle_capacity_customers = (K > 0) ? ((n + K - 1) / K) : n;
  return aco_vrp_cuda_with_capacity(
      n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q,
      seed, best_solution, best_cost);
}
