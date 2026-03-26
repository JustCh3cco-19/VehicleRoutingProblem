#include "aco.h"
#include "aco_cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>

#define THREADS_PER_BLOCK 128

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

float* flatten_costs_float(double **c, int n) {
    float *flat = (float*)malloc((n + 1) * (n + 1) * sizeof(float));
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            flat[i * (n + 1) + j] = (float)c[i][j];
        }
    }
    return flat;
}

int aco_vrp_cuda(int n, int K, double timeout_minutes, double improvement_threshold,
                 double **c, double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost) {

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    // Test con poche formiche per isolare il problema del timer
    int m = 64; 
    printf("[Init] GPU: %s, Ants: %d, Timeout: %.2f min\n", prop.name, m, timeout_minutes);

    int size_n = (n + 1) * (n + 1);
    float *h_c_flat = flatten_costs_float(c, n);
    
    float *d_costs, *d_tau, *d_cand_scores;
    int *d_cand_list, *d_ant_routes;
    float *d_ant_costs;

    CHECK_CUDA(cudaMalloc(&d_costs, size_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tau, size_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_list, (n + 1) * MAX_CANDIDATES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cand_scores, (n + 1) * MAX_CANDIDATES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ant_costs, m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ant_routes, m * (n + 2 * K) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_costs, h_c_flat, size_n * sizeof(float), cudaMemcpyHostToDevice));

    float *h_tau_init = (float*)malloc(size_n * sizeof(float));
    for (int i = 0; i < size_n; i++) h_tau_init[i] = (float)tau0;
    CHECK_CUDA(cudaMemcpy(d_tau, h_tau_init, size_n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_tau_init);

    launch_setup_aco(d_costs, d_cand_list, n, 32);
    CHECK_CUDA(cudaDeviceSynchronize());

    int words_per_mask = (n + 1 + 31) / 32;
    int warps_per_block = THREADS_PER_BLOCK / 32;
    size_t shared_mem_size_k4 = warps_per_block * (words_per_mask * sizeof(uint32_t) + K * sizeof(VehicleState));
    size_t shared_mem_size_k2 = words_per_mask * sizeof(uint32_t);

    *best_cost = DBL_MAX;
    int no_improvement_count = 0;
    float *h_ant_costs = (float*)malloc(m * sizeof(float));
    
    auto start_time = std::chrono::steady_clock::now();
    double timeout_seconds = timeout_minutes * 60.0;
    int iter = 0;

    printf("[Main] Loop start...\n");

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        
        if (elapsed >= timeout_seconds) {
            printf("\n[Main] TIMER TIMEOUT: %.2fs >= %.2fs. Exiting loop.\n", elapsed, timeout_seconds);
            break;
        }

        CHECK_CUDA(cudaMemset(d_ant_routes, 0, m * (n + 2 * K) * sizeof(int)));

        // K2
        launch_precompute_candidate_scores(d_tau, d_costs, d_cand_list, d_cand_scores, n, (float)alpha, (float)beta, 32, shared_mem_size_k2);
        
        // K4
        launch_construct_solutions(m, K, n, (float)Q, seed + iter, d_costs, d_tau, (float)alpha, (float)beta, d_cand_list, d_cand_scores, d_ant_costs, d_ant_routes, THREADS_PER_BLOCK, shared_mem_size_k4);

        // Sincronizzazione critica per il timer
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "CUDA Sync Error: %s\n", cudaGetErrorString(sync_err));
            break;
        }

        CHECK_CUDA(cudaMemcpy(h_ant_costs, d_ant_costs, m * sizeof(float), cudaMemcpyDeviceToHost));
        
        float iter_best = FLT_MAX;
        int iter_best_idx = -1;
        for (int i = 0; i < m; i++) {
            if (h_ant_costs[i] < iter_best) {
                iter_best = h_ant_costs[i];
                iter_best_idx = i;
            }
        }

        if (iter_best_idx != -1) {
            float deposit = (float)Q / iter_best;
            launch_deposit_pheromones(d_tau, d_ant_routes + iter_best_idx * (n + 2 * K), n, K, deposit);
            
            if (iter_best < *best_cost) {
                *best_cost = iter_best;
                no_improvement_count = 0;
                
                // Aggiornamento soluzione host (solo se migliorata)
                int total_steps = n + 2 * K;
                int *h_best_ant_route = (int*)malloc(total_steps * sizeof(int));
                CHECK_CUDA(cudaMemcpy(h_best_ant_route, d_ant_routes + iter_best_idx * total_steps, total_steps * sizeof(int), cudaMemcpyDeviceToHost));
                
                solution_reset(best_solution);
                for (int v = 0; v < K; v++) {
                    Route *r = &best_solution->routes[v];
                    r->len = 0; r->nodes[r->len++] = 0;
                    for (int s = v; s < total_steps; s += K) {
                        int node = h_best_ant_route[s];
                        if (node > 0 && node <= n) r->nodes[r->len++] = node;
                    }
                    r->nodes[r->len++] = 0;
                }
                free(h_best_ant_route);
            } else {
                no_improvement_count++;
            }
        }

        // K5
        int evaporate_threads = 256;
        int evaporate_blocks = (size_n + evaporate_threads - 1) / evaporate_threads;
        kernel_evaporate_pheromones<<<evaporate_blocks, evaporate_threads>>>(d_tau, (float)rho, size_n);

        // Print SEMPRE
        printf("Iter %d: Best = %.3f, Elapsed: %.2fs\n", iter, *best_cost, elapsed);
        fflush(stdout);
        
        iter++;
        if (no_improvement_count >= 100) {
            printf("[Main] Early stoppage at iter %d\n", iter);
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_costs)); CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUDA(cudaFree(d_cand_list)); CHECK_CUDA(cudaFree(d_cand_scores));
    CHECK_CUDA(cudaFree(d_ant_costs)); CHECK_CUDA(cudaFree(d_ant_routes));
    free(h_c_flat); free(h_ant_costs);

    return 0;
}
