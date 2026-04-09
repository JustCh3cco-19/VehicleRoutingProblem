#ifndef ACO_CUDA_KERNELS_H
#define ACO_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <curand_kernel.h>

// Costanti di design
#define MAX_CANDIDATES 32
#define WARP_SIZE 32

// Struttura per lo stato del veicolo in Shared Memory
struct VehicleState {
    int curr_node;
    float capacity;
    float cost;
};

// Firme dei Kernel (Hardware-Aware)

// K1: Setup statico delle liste candidati
__global__ void kernel_setup_aco(const float* d_costs, int* d_cand_list, int n);

// K2: Selezione dinamica dei candidati e precalcolo scores (Warp-Centric)
__global__ void kernel_precompute_candidate_scores(
    const float* d_tau, const float* d_costs, int* d_cand_list, 
    float* d_cand_scores, int n, float alpha, float beta);

// K4: Motore di costruzione soluzioni (Warp-Centric)
__global__ void kernel_construct_solutions(
    int m, int K, int n, float Q, unsigned int seed,
    const float* d_costs, const float* d_tau, float alpha, float beta,
    const int* d_cand_list, const float* d_cand_scores,
    float* d_ant_costs, int* d_ant_routes);

// K5: Evoluzione feromone
__global__ void kernel_evaporate_pheromones(float* d_tau, float rho, int size);
__global__ void kernel_deposit_pheromones(float* d_tau, const int* d_best_route, int n, int K, float deposit);

// Wrapper per il lancio dei kernel
void launch_setup_aco(const float* d_costs, int* d_cand_list, int n, int threads_per_block, size_t shared_mem_size);
void launch_precompute_candidate_scores(const float* d_tau, const float* d_costs, int* d_cand_list, float* d_cand_scores, int n, float alpha, float beta, int threads_per_block, size_t shared_mem_size);
void launch_construct_solutions(int m, int K, int n, float Q, unsigned int seed, const float* d_costs, const float* d_tau, float alpha, float beta, const int* d_cand_list, const float* d_cand_scores, float* d_ant_costs, int* d_ant_routes, int threads_per_block, size_t shared_mem_size);
void launch_evaporate_pheromones(float* d_tau, float rho, int size, int threads_per_block); 
void launch_deposit_pheromones(float* d_tau, const int* d_best_route, int n, int K, float deposit);

#endif // ACO_CUDA_KERNELS_H
