#ifndef ACO_CUDA_KERNELS_H
#define ACO_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * @brief CUDA backend execution constants.
 */
#define CUDA_THREADS_PER_BLOCK 128
#define CUDA_WARP_SIZE 32
#define CUDA_CACHE_LINE_SIZE 128
#define CUDA_MAX_CANDIDATES 32
#define CUDA_EPS 1e-9f

/**
 * @brief Runtime parameters used by CUDA kernels.
 */
typedef struct {
  int n;
  int k;
  int m;
  int cap;
  int cand_k;
  int route_max_len;


  float alpha;
  float beta;
  float rho;
  float tau0;
  float q;
  float tau_min;
  float tau_max;


  float log_tau_min;
  float log_tau_step;
  uint8_t q_tau_min;
  uint8_t q_tau_max;
  uint8_t q_evap_delta;


  int visited_l1_words;
  int visited_l2_words;
  size_t visited_row_stride;

  float depot_close_weight;
} CudaParams;

/**
 * @brief Per-ant summary written by CUDA kernels.
 */
typedef struct {
  int feasible;
  int unvisited_count;
  float cost;
} CudaAntSummary;

/**
 * @brief Optional iteration-level counters for diagnostics/profiling.
 */
typedef struct {
  unsigned long long candidate_calls;
  unsigned long long candidate_moves;
  unsigned long long fallback_calls;
  unsigned long long fallback_moves;
  unsigned long long depot_offer_calls;
  unsigned long long depot_close_moves;
  unsigned long long customer_moves;
  unsigned long long nonempty_routes;
  unsigned long long fallback_word_groups_scanned;
  unsigned long long fallback_nodes_scored;
} CudaIterStats;

/**
 * @brief Initializes the log-quantized pheromone matrix.
 * @param d_tau Device pheromone matrix.
 * @param total_elements Number of matrix elements.
 * @param q_tau0 Initial quantized pheromone value.
 */
__global__ void kernel_init_tau(uint8_t *d_tau, uint64_t total_elements, uint8_t q_tau0);

/**
 * @brief Builds nearest-candidate lists and precomputed eta^beta values.
 * @param d_coords Device coordinates.
 * @param d_candidate_idx Device candidate index matrix.
 * @param d_eta_beta Device eta^beta matrix.
 * @param params CUDA runtime parameters.
 */
__global__ void kernel_build_candidate_lists(const float2 *d_coords,
                                                int *d_candidate_idx,
                                                float *d_eta_beta,
                                                CudaParams params);

/**
 * @brief Resets all per-ant state at iteration start.
 */
__global__ void kernel_reset_ant_state(int *d_routes,
                                          int *d_route_lengths,
                                          uint64_t *d_visited_l1,
                                          uint64_t *d_visited_l2,
                                          unsigned int *d_rng_states,
                                          CudaAntSummary *d_ant_summary,
                                          CudaParams params,
                                          unsigned int seed);

/**
 * @brief Main solution-construction kernel (warp-per-ant execution).
 */
__global__ void kernel_construct_solutions(const float2 *d_coords,
                                              const uint8_t *d_tau,
                                              const int *d_candidate_idx,
                                              const float *d_eta_beta,
                                              int *d_routes,
                                              int *d_route_lengths,
                                              uint64_t *d_visited_l1,
                                              uint64_t *d_visited_l2,
                                              unsigned int *d_rng_states,
                                              CudaAntSummary *d_ant_summary,
                                              CudaIterStats *d_iter_stats,
                                              CudaParams params);

/**
 * @brief Applies pheromone evaporation with saturated subtraction.
 */
__global__ void kernel_evaporate_tau(uint8_t *d_tau, uint64_t total_elements, uint8_t delta, uint8_t q_min);

/**
 * @brief Deposits pheromone using the best ant of the iteration.
 */
__global__ void kernel_deposit_solution(uint8_t *d_tau,
                                           const int *d_routes,
                                           const int *d_route_lengths,
                                           float deposit_amount,
                                           int best_ant,
                                           CudaParams params);

/**
 * @brief Deposits pheromone from a flattened global-best solution buffer.
 */
__global__ void kernel_deposit_flat_solution(uint8_t *d_tau,
                                                const int *d_flat_routes,
                                                const int *d_flat_lengths,
                                                float deposit_amount,
                                                CudaParams params);

/**
 * @brief Host launcher for kernel_init_tau.
 */
void launch_init_tau(uint8_t *d_tau, int n, uint8_t q_tau0);

/**
 * @brief Host launcher for kernel_build_candidate_lists.
 */
void launch_build_candidate_lists(const float2 *d_coords, int *d_candidate_idx, float *d_eta_beta, CudaParams params);

/**
 * @brief Host launcher for kernel_reset_ant_state.
 */
void launch_reset_ant_state(int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaAntSummary *d_ant_summary, CudaParams params, unsigned int seed);

/**
 * @brief Host launcher for kernel_construct_solutions.
 */
void launch_construct_solutions(const float2 *d_coords, const uint8_t *d_tau, const int *d_candidate_idx, const float *d_eta_beta, int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaAntSummary *d_ant_summary, CudaIterStats *d_iter_stats, CudaParams params);

/**
 * @brief Host launcher for kernel_evaporate_tau.
 */
void launch_evaporate_tau(uint8_t *d_tau, int n, uint8_t delta, uint8_t q_min);

/**
 * @brief Host launcher for kernel_deposit_solution.
 */
void launch_deposit_solution(uint8_t *d_tau, const int *d_routes, const int *d_route_lengths, float deposit_amount, int best_ant, CudaParams params);

/**
 * @brief Host launcher for kernel_deposit_flat_solution.
 */
void launch_deposit_flat_solution(uint8_t *d_tau, const int *d_flat_routes, const int *d_flat_lengths, float deposit_amount, CudaParams params);

#endif
