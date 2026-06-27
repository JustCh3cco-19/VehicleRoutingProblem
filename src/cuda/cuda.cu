extern "C" {
#include "aco.h"
#include "config.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"
}

#include "cuda/cuda_kernels.h"

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
      status = ACO_ERR_BACKEND;                                              \
      goto cleanup;                                                          \
    }                                                                        \
  } while (0)

#define CHECK_CUDA_KERNEL() CHECK_CUDA(cudaGetLastError())

static double wall_time_seconds(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

static int select_iter_best_host(const CudaAntSummary *summary, int m,
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
        (fabsf(summary[i].cost - cost) <= (float)CUDA_EPS && i < idx)) {
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
                                int K, int m, int ant_idx,
                                Solution *dst) {
  int v;
  solution_reset(dst);

  int global_step = 0;
  for (v = 0; v < K; ++v) {
    int len = route_lengths[ant_idx * K + v];
    Route *r = &dst->routes[v];

    if (len < 2 || len > r->cap) {
      return 0;
    }

    r->len = len;
    for (int t = 0; t < len; ++t) {
      r->nodes[t] = routes[(global_step + t) * m + ant_idx];
    }
    global_step += len - 1;
  }

  return 1;
}

/**
 * @brief Runs the CUDA ACO backend with explicit vehicle capacity.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param vehicle_capacity_customers Per-vehicle customer capacity.
 * @param m Number of ants (0 enables backend default tuning).
 * @param coords_x X coordinates.
 * @param coords_y Y coordinates.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_solution Output best solution.
 * @param best_cost Output best cost.
 * @return 0 on success, non-zero on failure.
 */
AcoStatus aco_vrp_cuda_with_capacity(int n, int K,
                                     int vehicle_capacity_customers, int m,
                                     float *coords_x, float *coords_y,
                                     double alpha, double beta, double rho,
                                     double tau0, double Q, unsigned int seed,
                                     Solution *best_solution,
                                     double *best_cost) {
  AcoStatus status = ACO_OK;
  float2 *h_coords = NULL;
  CudaAntSummary *h_ant_summary = NULL;
  int *h_routes = NULL;
  int *h_route_lengths = NULL;
  int *h_flat_routes = NULL;
  int *h_flat_lengths = NULL;
  float2 *d_coords = NULL;
  uint8_t *d_tau = NULL;
  int *d_candidate_idx = NULL;
  float *d_eta_beta = NULL;
  int *d_routes = NULL;
  int *d_route_lengths = NULL;
  uint64_t *d_visited_l1 = NULL;
  uint64_t *d_visited_l2 = NULL;
  unsigned int *d_rng_states = NULL;
  CudaAntSummary *d_ant_summary = NULL;
  CudaIterStats *d_iter_stats = NULL;
  int *d_flat_routes = NULL;
  int *d_flat_lengths = NULL;
  int max_steps = 0;
  double max_runtime_sec = 0.0;
  int max_stagnation_epochs = 0;
  double min_rel_improvement = 1e-3;
  double progress_interval_sec = 0.0;
  int iter = 0;
  CudaIterStats accumulated_stats = {0};
  int iter_since_best = 0;
  double start_time = 0.0;
  double next_progress_time = 0.0;
  double global_best = DBL_MAX;
  size_t total_elements = 0;
  size_t tau_alloc_elements = 0;

  if (n <= 0 || K <= 0 || !coords_x || !coords_y || !best_solution ||
      !best_cost) {
    return ACO_ERR_INVALID_INPUT;
  }

  AcoRuntimeConfig config;
  aco_runtime_config_load_env(&config);
  config.ants = m;
  config.seed = seed;

  if (m == 0) {
    m = (config.ants > 0) ? config.ants : 256;
  }

  int cand_k = (config.candidate_k > 0) ? config.candidate_k : 32;
  if (cand_k > n) {
    cand_k = n;
  }
  if (cand_k < 1) {
    cand_k = 1;
  }

  float log_tau_min = logf(0.0001f);
  float log_tau_max = logf(100.0f);
  uint8_t q_tau_min = 0;
  uint8_t q_tau_max = 255;
  float log_tau_step =
      (log_tau_max - log_tau_min) / (float)(q_tau_max - q_tau_min);
  float log_rho = logf(1.0f - (float)rho);
  uint8_t q_evap_delta = (uint8_t)fmaxf(1.0f, roundf(-log_rho / log_tau_step));

  CudaParams params = {0};
  params.n = n;
  params.K = K;
  params.m = m;
  params.cap = vehicle_capacity_customers;
  params.cand_k = cand_k;
  params.route_max_len = K * (n + 1);
  params.alpha = (float)alpha;
  params.beta = (float)beta;
  params.rho = (float)rho;
  params.tau0 = (float)tau0;
  params.Q = (float)Q * 100.0f;
  params.tau_min = 0.0001f;
  params.tau_max = 100.0f;
  params.log_tau_min = log_tau_min;
  params.log_tau_step = log_tau_step;
  params.q_tau_min = q_tau_min;
  params.q_tau_max = q_tau_max;
  params.q_evap_delta = q_evap_delta;

  params.visited_l1_words = (n + 64) / 64;
  params.visited_l2_words = (params.visited_l1_words + 63) / 64;
  params.visited_row_stride =
      ((params.visited_l1_words * 8 + 127) / 128) * 128;
  params.depot_close_weight = 2.0f;

  uint8_t q_tau0 =
      (uint8_t)fmaxf((float)q_tau_min,
                     fminf((float)q_tau_max,
                           roundf((logf(params.tau0) - log_tau_min) /
                                  log_tau_step)));

  h_coords = (float2 *)malloc((n + 1) * sizeof(float2));
  if (!h_coords) {
    status = ACO_ERR_ALLOCATION;
    goto cleanup;
  }
  for (int i = 0; i <= n; i++) {
    h_coords[i].x = coords_x[i];
    h_coords[i].y = coords_y[i];
  }

  total_elements = (size_t)(n + 1) * (size_t)(n + 1);
  tau_alloc_elements = (total_elements + 3) & ~3ull;

  CHECK_CUDA(cudaMalloc(&d_coords, (n + 1) * sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&d_tau, tau_alloc_elements * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&d_candidate_idx, (n + 1) * cand_k * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_eta_beta, (n + 1) * cand_k * sizeof(float)));


  max_steps = params.route_max_len + 1;
  CHECK_CUDA(cudaMalloc(&d_routes, max_steps * m * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_route_lengths, m * K * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&d_visited_l1, m * params.visited_row_stride));
  CHECK_CUDA(
      cudaMalloc(&d_visited_l2, m * params.visited_l2_words * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_rng_states, m * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&d_ant_summary, m * sizeof(CudaAntSummary)));
  CHECK_CUDA(cudaMalloc(&d_iter_stats, sizeof(CudaIterStats)));

  CHECK_CUDA(cudaMemcpy(d_coords, h_coords, (n + 1) * sizeof(float2),
                        cudaMemcpyHostToDevice));

  launch_init_tau(d_tau, n, q_tau0);
  CHECK_CUDA_KERNEL();
  CHECK_CUDA(cudaDeviceSynchronize());
  launch_build_candidate_lists(d_coords, d_candidate_idx, d_eta_beta, params);
  CHECK_CUDA_KERNEL();
  CHECK_CUDA(cudaDeviceSynchronize());

  h_ant_summary = (CudaAntSummary *)malloc(m * sizeof(CudaAntSummary));
  h_routes = (int *)malloc(max_steps * m * sizeof(int));
  h_route_lengths = (int *)malloc(m * K * sizeof(int));

  h_flat_routes = (int *)malloc(K * (n + 1) * sizeof(int));
  h_flat_lengths = (int *)malloc(K * sizeof(int));
  if (!h_ant_summary || !h_routes || !h_route_lengths || !h_flat_routes ||
      !h_flat_lengths) {
    status = ACO_ERR_ALLOCATION;
    goto cleanup;
  }
  CHECK_CUDA(cudaMalloc(&d_flat_routes, K * (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_flat_lengths, K * sizeof(int)));

  max_runtime_sec = config.timeout_seconds;
  max_stagnation_epochs = config.stagnation_epochs;
  min_rel_improvement = config.min_rel_improvement;
  progress_interval_sec = config.progress_interval_seconds;

  start_time = wall_time_seconds();
  next_progress_time =
      (progress_interval_sec > 0.0) ? start_time + progress_interval_sec : 0.0;

  if (config.log_level > ACO_LOG_SILENT) {
    fprintf(stderr,
            "CUDA Solver starting... (N=%d, K=%d, M=%d, candidate_k=%d, "
            "seed=%u)\n",
            n, K, m, cand_k, seed);
  }

  while (1) {
    double current_time = wall_time_seconds();
    if (max_runtime_sec > 0.0 && (current_time - start_time) > max_runtime_sec) {
      break;
    }
    if (max_stagnation_epochs > 0 && iter_since_best >= max_stagnation_epochs) {
      break;
    }

    launch_reset_ant_state(d_routes, d_route_lengths, d_visited_l1,
                           d_visited_l2, d_rng_states, d_ant_summary, params,
                           seed + iter);
    CHECK_CUDA_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemset(d_iter_stats, 0, sizeof(CudaIterStats)));

    launch_construct_solutions(d_coords, d_tau, d_candidate_idx, d_eta_beta,
                               d_routes, d_route_lengths, d_visited_l1,
                               d_visited_l2, d_rng_states, d_ant_summary,
                               d_iter_stats, params);
    CHECK_CUDA_KERNEL();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ant_summary, d_ant_summary,
                          m * sizeof(CudaAntSummary), cudaMemcpyDeviceToHost));

    CudaIterStats host_stats;
    if (cudaMemcpy(&host_stats, d_iter_stats, sizeof(CudaIterStats), cudaMemcpyDeviceToHost) == cudaSuccess) {
      accumulated_stats.candidate_moves += host_stats.candidate_moves;
      accumulated_stats.fallback_calls += host_stats.fallback_calls;
      accumulated_stats.fallback_moves += host_stats.fallback_moves;
      accumulated_stats.depot_offer_calls += host_stats.depot_offer_calls;
      accumulated_stats.depot_close_moves += host_stats.depot_close_moves;
      accumulated_stats.customer_moves += host_stats.customer_moves;
      accumulated_stats.nonempty_routes += host_stats.nonempty_routes;
      accumulated_stats.fallback_word_groups_scanned += host_stats.fallback_word_groups_scanned;
      accumulated_stats.fallback_nodes_scored += host_stats.fallback_nodes_scored;
    }

    int best_idx = -1;
    float best_iter_cost = FLT_MAX;
    int found = select_iter_best_host(h_ant_summary, m, &best_idx, &best_iter_cost);

    if (found) {
      double new_best = (double)best_iter_cost;
      if (new_best < global_best) {
        if (is_significant_improvement(global_best, new_best,
                                       min_rel_improvement)) {
          iter_since_best = 0;
        } else {
          iter_since_best++;
        }
        global_best = new_best;
        *best_cost = global_best;

        CHECK_CUDA(cudaMemcpy(h_routes, d_routes, max_steps * m * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_route_lengths, d_route_lengths,
                              m * K * sizeof(int), cudaMemcpyDeviceToHost));

        copy_ant_to_solution(h_routes, h_route_lengths, K, m, best_idx,
                             best_solution);

        for (int v = 0; v < K; v++) {
          h_flat_lengths[v] = best_solution->routes[v].len;
          for (int i = 0; i < best_solution->routes[v].len; i++) {
            h_flat_routes[v * (n + 1) + i] = best_solution->routes[v].nodes[i];
          }
        }
        CHECK_CUDA(cudaMemcpy(d_flat_routes, h_flat_routes,
                              K * (n + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_flat_lengths, h_flat_lengths, K * sizeof(int),
                              cudaMemcpyHostToDevice));
      } else {
        iter_since_best++;
      }

      launch_evaporate_tau(d_tau, n, params.q_evap_delta, params.q_tau_min);
      CHECK_CUDA_KERNEL();
      CHECK_CUDA(cudaDeviceSynchronize());

      float iter_deposit = (0.3f * params.Q) / best_iter_cost;
      launch_deposit_solution(d_tau, d_routes, d_route_lengths, iter_deposit,
                              best_idx, params);
      CHECK_CUDA_KERNEL();

      float global_deposit = (0.7f * params.Q) / (float)global_best;
      launch_deposit_flat_solution(d_tau, d_flat_routes, d_flat_lengths,
                                   global_deposit, params);
      CHECK_CUDA_KERNEL();

      CHECK_CUDA(cudaDeviceSynchronize());
    } else {
      iter_since_best++;
    }

    double progress_time = wall_time_seconds();
    if (config.log_level > ACO_LOG_SILENT && progress_interval_sec > 0.0 &&
        progress_time >= next_progress_time) {
      double elapsed = progress_time - start_time;
      if (max_runtime_sec > 0.0) {
        double remaining = max_runtime_sec - elapsed;
        if (remaining < 0.0) {
          remaining = 0.0;
        }
        fprintf(stderr,
                "[cuda] elapsed %.1fs, remaining %.1fs, iter %d, best %.3f\n",
                elapsed, remaining, iter + 1, global_best);
      } else {
        fprintf(stderr, "[cuda] elapsed %.1fs, iter %d, best %.3f\n", elapsed,
                iter + 1, global_best);
      }
      next_progress_time = progress_time + progress_interval_sec;
    }

    iter++;
  }

  if (config.log_level > ACO_LOG_SILENT) {
    fprintf(stderr, "\n[cuda] Iteration statistics summary (accumulated over %d iterations):\n", iter);
    fprintf(stderr, "[cuda]   Customer moves:      %llu\n", accumulated_stats.customer_moves);
    fprintf(stderr, "[cuda]   Candidate moves:     %llu\n", accumulated_stats.candidate_moves);
    fprintf(stderr, "[cuda]   Fallback calls:      %llu\n", accumulated_stats.fallback_calls);
    fprintf(stderr, "[cuda]   Fallback moves:      %llu\n", accumulated_stats.fallback_moves);
    fprintf(stderr, "[cuda]   Depot offer calls:   %llu\n", accumulated_stats.depot_offer_calls);
    fprintf(stderr, "[cuda]   Depot close moves:   %llu\n", accumulated_stats.depot_close_moves);
    fprintf(stderr, "[cuda]   Non-empty routes:    %llu\n", accumulated_stats.nonempty_routes);
    fprintf(stderr, "[cuda]   Fallback groups scn: %llu\n", accumulated_stats.fallback_word_groups_scanned);
    fprintf(stderr, "[cuda]   Fallback nodes scd:  %llu\n", accumulated_stats.fallback_nodes_scored);
  }

  status = (*best_cost < DBL_MAX) ? ACO_OK : ACO_ERR_NO_SOLUTION;

cleanup:
  cudaFree(d_coords);
  cudaFree(d_tau);
  cudaFree(d_candidate_idx);
  cudaFree(d_eta_beta);
  cudaFree(d_routes);
  cudaFree(d_route_lengths);
  cudaFree(d_visited_l1);
  cudaFree(d_visited_l2);
  cudaFree(d_rng_states);
  cudaFree(d_ant_summary);
  cudaFree(d_iter_stats);
  cudaFree(d_flat_routes);
  cudaFree(d_flat_lengths);

  free(h_coords);
  free(h_ant_summary);
  free(h_routes);
  free(h_route_lengths);
  free(h_flat_routes);
  free(h_flat_lengths);

  return status;
}
