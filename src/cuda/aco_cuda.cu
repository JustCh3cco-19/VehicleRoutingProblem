extern "C" {
#include "aco.h"
#include "matrix.h"
#include "solution.h"
}
#include "aco_cuda_kernels.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

#define THREADS_PER_BLOCK 128

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

static double get_wall_time() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
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

int aco_vrp_cuda(int n, int K, int m, double timeout_minutes, double improvement_threshold,
                 double **c, double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost) {

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    
    // Se m=0, usiamo il compromesso ideale (64 formiche) per bilanciare occupancy e speed
    int total_m = (m <= 0) ? 64 : m; 
    printf("[Init] GPU: %s, Ants: %d, Timeout: %.2f min\n", prop.name, total_m, timeout_minutes);

    int size_n = (n + 1) * (n + 1);
    float *h_c_flat = flatten_costs_float(c, n);
    
    float *d_costs, *d_tau, *d_cand_scores;
    int *d_cand_list, *d_ant_routes;
    float *d_ant_costs;

    int route_steps = n + 2;
    int total_ant_steps = K * route_steps;

    CHECK_CUDA(cudaMalloc(&d_costs, size_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_tau, size_n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_cand_list, (n + 1) * MAX_CANDIDATES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_cand_scores, (n + 1) * MAX_CANDIDATES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ant_costs, total_m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ant_routes, total_m * total_ant_steps * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_costs, h_c_flat, size_n * sizeof(float), cudaMemcpyHostToDevice));

    float *h_tau_init = (float*)malloc(size_n * sizeof(float));
    for (int i = 0; i < size_n; i++) h_tau_init[i] = (float)tau0;
    CHECK_CUDA(cudaMemcpy(d_tau, h_tau_init, size_n * sizeof(float), cudaMemcpyHostToDevice));
    free(h_tau_init);

    int words_per_mask = (n + 1 + 31) / 32;
    size_t shared_mem_size_setup = words_per_mask * sizeof(uint32_t);
    launch_setup_aco(d_costs, d_cand_list, n, 32, shared_mem_size_setup);
    CHECK_CUDA(cudaDeviceSynchronize());

    int warps_per_block = THREADS_PER_BLOCK / 32;
    int words_per_vehicle = sizeof(VehicleState) / sizeof(uint32_t);
    size_t shared_mem_size_k4 = warps_per_block * (words_per_mask + K * words_per_vehicle) * sizeof(uint32_t);
    size_t shared_mem_size_k2 = words_per_mask * sizeof(uint32_t);

    *best_cost = DBL_MAX;
    int no_improvement_count = 0;
    float *h_ant_costs = (float*)malloc(total_m * sizeof(float));
    
    double start_time = get_wall_time();
    double timeout_seconds = timeout_minutes * 60.0;
    int iter = 0;

    printf("[Main] Loop start...\n");

    while (true) {
        double now = get_wall_time();
        double elapsed = now - start_time;
        
        if (elapsed >= timeout_seconds) {
            printf("\n[Main] TIMER TIMEOUT: %.2fs >= %.2fs. Exiting loop.\n", elapsed, timeout_seconds);
            break;
        }

        CHECK_CUDA(cudaMemset(d_ant_routes, 0, total_m * total_ant_steps * sizeof(int)));

        // K2
        launch_precompute_candidate_scores(d_tau, d_costs, d_cand_list, d_cand_scores, n, (float)alpha, (float)beta, 32, shared_mem_size_k2);

        // K4
        launch_construct_solutions(total_m, K, n, (float)Q, seed + iter, d_costs, d_tau, (float)alpha, (float)beta, d_cand_list, d_cand_scores, d_ant_costs, d_ant_routes, THREADS_PER_BLOCK, shared_mem_size_k4);

        // Sincronizzazione critica per il timer
        cudaError_t sync_err = cudaDeviceSynchronize();
        if (sync_err != cudaSuccess) {
            fprintf(stderr, "CUDA Sync Error: %s\n", cudaGetErrorString(sync_err));
            break;
        }

        CHECK_CUDA(cudaMemcpy(h_ant_costs, d_ant_costs, total_m * sizeof(float), cudaMemcpyDeviceToHost));
        
        float iter_best = FLT_MAX;
        int iter_best_idx = -1;
        for (int i = 0; i < total_m; i++) {
            if (h_ant_costs[i] < iter_best) {
                iter_best = h_ant_costs[i];
                iter_best_idx = i;
            }
        }

        if (iter_best_idx != -1) {
            float deposit = (float)Q / iter_best;
            launch_deposit_pheromones(d_tau, d_ant_routes + iter_best_idx * total_ant_steps, n, K, deposit);
            
            if (iter_best < *best_cost) {
                *best_cost = iter_best;
                no_improvement_count = 0;
                
                // Aggiornamento soluzione host (sequenziale)
                int *h_best_ant_routes = (int*)malloc(total_ant_steps * sizeof(int));
                CHECK_CUDA(cudaMemcpy(h_best_ant_routes, d_ant_routes + iter_best_idx * total_ant_steps, total_ant_steps * sizeof(int), cudaMemcpyDeviceToHost));
                
                solution_reset(best_solution);
                for (int v = 0; v < K; v++) {
                    Route *r = &best_solution->routes[v];
                    r->len = 0; 
                    r->nodes[r->len++] = 0;
                    
                    int route_offset = v * route_steps;
                    for (int s = 0; s < route_steps; s++) {
                        int node = h_best_ant_routes[route_offset + s];
                        if (node > 0 && node <= n) {
                            r->nodes[r->len++] = node;
                        } else if (s > 0 && node == 0) {
                            break;
                        }
                    }
                    r->nodes[r->len++] = 0;
                }
                free(h_best_ant_routes);
            } else {
                no_improvement_count++;
            }
        }

        // K5
        launch_evaporate_pheromones(d_tau, (float)rho, size_n, 256);

        // Print SEMPRE
        printf("Iter %d: Best = %.3f, Elapsed: %.2fs\n", iter, *best_cost, elapsed);
        fflush(stdout);
        
        iter++;
        // Early stoppage scatta solo se improvement_threshold > 0
        if (improvement_threshold > 0.0 && no_improvement_count >= 100) {
            printf("[Main] Early stoppage at iter %d (No improvement for 100 iterations)\n", iter);
            break;
        }
    }

    CHECK_CUDA(cudaFree(d_costs)); CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUDA(cudaFree(d_cand_list)); CHECK_CUDA(cudaFree(d_cand_scores));
    CHECK_CUDA(cudaFree(d_ant_costs)); CHECK_CUDA(cudaFree(d_ant_routes));
    free(h_c_flat); free(h_ant_costs);

    return 0;
}
