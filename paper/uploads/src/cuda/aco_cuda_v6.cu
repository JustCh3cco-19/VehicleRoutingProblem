extern "C" {
#include "aco.h"
#include "instance_parser.h"
#include "matrix.h"
#include "solution.h"
}

#include "aco_cuda_v6_kernels.h"

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

static int select_iter_best_host(const CudaV6AntSummary *summary, int m,
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
        (fabsf(summary[i].cost - cost) <= (float)CUDA_V6_EPS && i < idx)) {
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

  // Note: in v6 routes are step-interleaved. routes[step * m + ant]
  // We need to reconstruct the vehicle routes from this interleaved format.
  // Actually, the kernel writes sequentially for each vehicle, but interleaves across ants.
  int global_step = 0;
  for (v = 0; v < K; ++v) {
    int len = route_lengths[ant_idx * K + v];
    Route *r = &dst->routes[v];

    if (len < 2 || len > r->cap) {
      return 0;
    }

    r->len = len;
    for (int t = 0; t < len; ++t) {
        // global_step increments for each step of each vehicle
        r->nodes[t] = routes[(global_step + t) * m + ant_idx];
    }
    global_step += len - 1; // lengths includes depot return, which is the start of the next if not the first.
                            // Actually, in the kernel, each vehicle starts at step 0 (depot) but we just keep incrementing global_step.
                            // Let's rely on the lengths array. 
                            // The kernel does: global_step++; routes[global_step*m+ant] = move;
  }

  return 1;
}

int aco_vrp_cuda_with_capacity_v6(int n, int K, int vehicle_capacity_customers,
                                  int m, float *coords_x, float *coords_y,
                                  double alpha, double beta, double rho,
                                  double tau0, double Q, unsigned int seed,
                                  Solution *best_solution, double *best_cost) {
  // Config
  int cand_k = 32;
  if (n <= 32) cand_k = n;
  
  if (m == 0) m = 256;
  
  // Pheromone quantization config
  float log_tau_min = logf(0.0001f); // Adjust as needed
  float log_tau_max = logf(100.0f);
  uint8_t q_tau_min = 0;
  uint8_t q_tau_max = 255;
  float log_tau_step = (log_tau_max - log_tau_min) / (float)(q_tau_max - q_tau_min);
  float log_rho = logf(1.0f - (float)rho);
  uint8_t q_evap_delta = (uint8_t)fmaxf(1.0f, roundf(-log_rho / log_tau_step));
  
  CudaV6Params params = {0};
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
  params.Q = (float)Q * 100.0f; // Multiplier to force quantization to trigger
  params.tau_min = 0.0001f;
  params.tau_max = 100.0f;
  params.log_tau_min = log_tau_min;
  params.log_tau_step = log_tau_step;
  params.q_tau_min = q_tau_min;
  params.q_tau_max = q_tau_max;
  params.q_evap_delta = q_evap_delta;

  params.visited_l1_words = (n + 64) / 64; // bits to words
  params.visited_l2_words = (params.visited_l1_words + 63) / 64;
  params.visited_row_stride = ((params.visited_l1_words * 8 + 127) / 128) * 128; // align to 128 bytes
  params.depot_close_weight = 2.0f; // Give depot a bit of bias

  uint8_t q_tau0 = (uint8_t)fmaxf((float)q_tau_min, fminf((float)q_tau_max, roundf((logf(params.tau0) - log_tau_min) / log_tau_step)));

  // Setup arrays
  float2 *h_coords = (float2 *)malloc((n + 1) * sizeof(float2));
  for (int i = 0; i <= n; i++) {
      h_coords[i].x = coords_x[i];
      h_coords[i].y = coords_y[i];
  }

  // Device allocations
  float2 *d_coords = NULL;
  uint8_t *d_tau = NULL;
  int *d_candidate_idx = NULL;
  float *d_eta_beta = NULL;
  int *d_routes = NULL;
  int *d_route_lengths = NULL;
  uint64_t *d_visited_l1 = NULL;
  uint64_t *d_visited_l2 = NULL;
  unsigned int *d_rng_states = NULL;
  CudaV6AntSummary *d_ant_summary = NULL;
  CudaV6IterStats *d_iter_stats = NULL;

  size_t total_elements = (size_t)(n + 1) * (size_t)(n + 1);
  size_t tau_alloc_elements = (total_elements + 3) & ~3ull;

  CHECK_CUDA(cudaMalloc(&d_coords, (n + 1) * sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&d_tau, tau_alloc_elements * sizeof(uint8_t)));
  CHECK_CUDA(cudaMalloc(&d_candidate_idx, (n + 1) * cand_k * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_eta_beta, (n + 1) * cand_k * sizeof(float)));
  
  // Routes: step interleaved. max_steps = K * (n+1) -> allocating max_steps * m
  int max_steps = params.route_max_len + 1;
  CHECK_CUDA(cudaMalloc(&d_routes, max_steps * m * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_route_lengths, m * K * sizeof(int)));
  
  CHECK_CUDA(cudaMalloc(&d_visited_l1, m * params.visited_row_stride));
  CHECK_CUDA(cudaMalloc(&d_visited_l2, m * params.visited_l2_words * sizeof(uint64_t)));
  CHECK_CUDA(cudaMalloc(&d_rng_states, m * sizeof(unsigned int)));
  CHECK_CUDA(cudaMalloc(&d_ant_summary, m * sizeof(CudaV6AntSummary)));
  CHECK_CUDA(cudaMalloc(&d_iter_stats, sizeof(CudaV6IterStats)));

  CHECK_CUDA(cudaMemcpy(d_coords, h_coords, (n + 1) * sizeof(float2), cudaMemcpyHostToDevice));

  // Initialize tau & candidates
  launch_init_tau_v6(d_tau, n, q_tau0);
  CHECK_CUDA(cudaDeviceSynchronize());
  launch_build_candidate_lists_v6(d_coords, d_candidate_idx, d_eta_beta, params);
  CHECK_CUDA(cudaDeviceSynchronize());

  // Host buffers for reading back
  CudaV6AntSummary *h_ant_summary = (CudaV6AntSummary *)malloc(m * sizeof(CudaV6AntSummary));
  int *h_routes = (int *)malloc(max_steps * m * sizeof(int));
  int *h_route_lengths = (int *)malloc(m * K * sizeof(int));

  // Global best flat storage for reinforcement
  int *h_flat_routes = (int *)malloc(K * (n + 1) * sizeof(int));
  int *h_flat_lengths = (int *)malloc(K * sizeof(int));
  int *d_flat_routes = NULL;
  int *d_flat_lengths = NULL;
  CHECK_CUDA(cudaMalloc(&d_flat_routes, K * (n + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_flat_lengths, K * sizeof(int)));

  double max_runtime_sec = 0.0;
  int max_stagnation_epochs = 0;
  double min_rel_improvement = 1e-3;
  load_timer_directives(&max_runtime_sec, &max_stagnation_epochs, &min_rel_improvement);

  int iter = 0;
  int iter_since_best = 0;
  double start_time = wall_time_seconds();
  double global_best = DBL_MAX;

  printf("CUDA v6 Solver starting... (N=%d, K=%d, M=%d)\n", n, K, m);

  while (1) {
    double current_time = wall_time_seconds();
    if (max_runtime_sec > 0.0 && (current_time - start_time) > max_runtime_sec) {
      break;
    }
    if (max_stagnation_epochs > 0 && iter_since_best >= max_stagnation_epochs) {
      break;
    }

    launch_reset_ant_state_v6(d_routes, d_route_lengths, d_visited_l1, d_visited_l2, d_rng_states, d_ant_summary, params, seed + iter);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemset(d_iter_stats, 0, sizeof(CudaV6IterStats)));

    launch_construct_solutions_v6(d_coords, d_tau, d_candidate_idx, d_eta_beta, d_routes, d_route_lengths, d_visited_l1, d_visited_l2, d_rng_states, d_ant_summary, d_iter_stats, params);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ant_summary, d_ant_summary, m * sizeof(CudaV6AntSummary), cudaMemcpyDeviceToHost));

    int best_idx = -1;
    float best_iter_cost = FLT_MAX;
    int found = select_iter_best_host(h_ant_summary, m, &best_idx, &best_iter_cost);

    if (found) {
      double new_best = (double)best_iter_cost;
      if (new_best < global_best) {
        if (is_significant_improvement(global_best, new_best, min_rel_improvement)) {
          iter_since_best = 0;
        } else {
          iter_since_best++;
        }
        global_best = new_best;
        *best_cost = global_best;

        CHECK_CUDA(cudaMemcpy(h_routes, d_routes, max_steps * m * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_route_lengths, d_route_lengths, m * K * sizeof(int), cudaMemcpyDeviceToHost));

        copy_ant_to_solution(h_routes, h_route_lengths, K, m, best_idx, best_solution);

        // Flatten global best for reinforcement
        for (int v = 0; v < K; v++) {
          h_flat_lengths[v] = best_solution->routes[v].len;
          for (int i = 0; i < best_solution->routes[v].len; i++) {
            h_flat_routes[v * (n + 1) + i] = best_solution->routes[v].nodes[i];
          }
        }
        CHECK_CUDA(cudaMemcpy(d_flat_routes, h_flat_routes, K * (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_flat_lengths, h_flat_lengths, K * sizeof(int), cudaMemcpyHostToDevice));
      } else {
        iter_since_best++;
      }

      launch_evaporate_tau_v6(d_tau, n, params.q_evap_delta, params.q_tau_min);
      CHECK_CUDA(cudaDeviceSynchronize());
      
      // Reinforce iteration best (30%)
      float iter_deposit = (0.3f * params.Q) / best_iter_cost;
      launch_deposit_solution_v6(d_tau, d_routes, d_route_lengths, iter_deposit, best_idx, params);
      
      // Reinforce global best (70%)
      float global_deposit = (0.7f * params.Q) / (float)global_best;
      launch_deposit_flat_solution_v6(d_tau, d_flat_routes, d_flat_lengths, global_deposit, params);
      
      CHECK_CUDA(cudaDeviceSynchronize());
    } else {
      iter_since_best++;
    }

    iter++;
  }

  // Cleanup
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

  return 0;
}
