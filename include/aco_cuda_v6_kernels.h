#ifndef ACO_CUDA_V6_KERNELS_H
#define ACO_CUDA_V6_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

/* Configurazione Hardware-Specific */
#define CUDA_V6_THREADS_PER_BLOCK 128
#define CUDA_V6_WARP_SIZE 32
#define CUDA_V6_CACHE_LINE_SIZE 128
#define CUDA_V6_MAX_CANDIDATES 32
#define CUDA_V6_EPS 1e-9f

/* Strutture Dati Principali */

typedef struct {
  int n;                /* Numero di clienti */
  int K;                /* Numero di veicoli per formica */
  int m;                /* Numero totale di formiche */
  int cap;              /* Capacità veicoli */
  int cand_k;           /* Numero candidati (fissato a 32) */
  int route_max_len;    /* Lunghezza massima rotta (K * n_approx) */
  
  /* Parametri ACO */
  float alpha;
  float beta;
  float rho;
  float tau0;
  float Q;
  float tau_min;
  float tau_max;
  
  /* Parametri Quantizzazione Logaritmica */
  float log_tau_min;
  float log_tau_step;
  uint8_t q_tau_min;
  uint8_t q_tau_max;
  uint8_t q_evap_delta;

  /* Parametri Memoria e Allineamento */
  int visited_l1_words;     /* Parole 64-bit per bitset L1 */
  int visited_l2_words;     /* Parole 64-bit per bitset L2 (Summary) */
  size_t visited_row_stride; /* Stride allineato a 128 byte per bitset */
  
  float depot_close_weight; /* Peso euristico per tornare al deposito */
} CudaV6Params;

typedef struct {
  int feasible;
  int unvisited_count;
  float cost;
} CudaV6AntSummary;

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
} CudaV6IterStats;

/* Kernel Declarations */

/**
 * Inizializza la matrice dei feromoni log-quantizzata a 8-bit.
 * d_tau: Matrice N x N di uint8_t
 */
__global__ void kernel_init_tau_v6(uint8_t *d_tau, uint64_t total_elements, uint8_t q_tau0);

/**
 * Costruisce le liste dei 32 candidati più vicini partendo dalle coordinate.
 * Calcola anche eta_beta (1/d^beta) per uso immediato.
 */
__global__ void kernel_build_candidate_lists_v6(const float2 *d_coords,
                                                int *d_candidate_idx,
                                                float *d_eta_beta,
                                                CudaV6Params params);

/**
 * Reset dello stato delle formiche per una nuova iterazione.
 * Inizializza bitset L1 e L2 (Summary Mask).
 */
__global__ void kernel_reset_ant_state_v6(int *d_routes, 
                                          int *d_route_lengths,
                                          uint64_t *d_visited_l1,
                                          uint64_t *d_visited_l2,
                                          unsigned int *d_rng_states,
                                          CudaV6AntSummary *d_ant_summary,
                                          CudaV6Params params,
                                          unsigned int seed);

/**
 * Kernel Core: Costruzione delle soluzioni (Warp-per-Ant).
 * Utilizza Hierarchical Bitset Scanning e Calcolo Distanze On-the-fly.
 */
__global__ void kernel_construct_solutions_v6(const float2 *d_coords,
                                              const uint8_t *d_tau,
                                              const int *d_candidate_idx,
                                              const float *d_eta_beta,
                                              int *d_routes,
                                              int *d_route_lengths,
                                              uint64_t *d_visited_l1,
                                              uint64_t *d_visited_l2,
                                              unsigned int *d_rng_states,
                                              CudaV6AntSummary *d_ant_summary,
                                              CudaV6IterStats *d_iter_stats,
                                              CudaV6Params params);

/**
 * Evaporazione dei feromoni tramite sottrazione intera saturata.
 * d_tau: uint8_t array massivo (fino a 10GB).
 */
__global__ void kernel_evaporate_tau_v6(uint8_t *d_tau, uint64_t total_elements, uint8_t delta, uint8_t q_min);

/**
 * Deposito dei feromoni (Sparse Update).
 * Converte il deposito float in incrementi logaritmici uint8_t.
 */
__global__ void kernel_deposit_solution_v6(uint8_t *d_tau,
                                           const int *d_routes,
                                           const int *d_route_lengths,
                                           float deposit_amount,
                                           int best_ant,
                                           CudaV6Params params);

/* Host Launchers */

void launch_init_tau_v6(uint8_t *d_tau, int n, uint8_t q_tau0);
void launch_build_candidate_lists_v6(const float2 *d_coords, int *d_candidate_idx, float *d_eta_beta, CudaV6Params params);
void launch_reset_ant_state_v6(int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaV6AntSummary *d_ant_summary, CudaV6Params params, unsigned int seed);
void launch_construct_solutions_v6(const float2 *d_coords, const uint8_t *d_tau, const int *d_candidate_idx, const float *d_eta_beta, int *d_routes, int *d_route_lengths, uint64_t *d_visited_l1, uint64_t *d_visited_l2, unsigned int *d_rng_states, CudaV6AntSummary *d_ant_summary, CudaV6IterStats *d_iter_stats, CudaV6Params params);
void launch_evaporate_tau_v6(uint8_t *d_tau, int n, uint8_t delta, uint8_t q_min);
void launch_deposit_solution_v6(uint8_t *d_tau, const int *d_routes, const int *d_route_lengths, float deposit_amount, int best_ant, CudaV6Params params);

#endif
