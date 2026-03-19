#include "aco.h"
#include "aco_cuda_kernels.h"
#include "aco_cuda_host_utils.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define THREADS_PER_BLOCK 256

int aco_vrp_cuda(int n, int K, int m, int T, double **c,
                 double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost) {
    
    int size = (n + 1) * (n + 1);
    double *h_c_flat = flatten_matrix(c, n);
    if (!h_c_flat) return 1;

    double *d_c, *d_eta, *d_tau, *d_scores;
    cudaMalloc(&d_c, size * sizeof(double));
    cudaMalloc(&d_eta, size * sizeof(double));
    cudaMalloc(&d_tau, size * sizeof(double));
    cudaMalloc(&d_scores, size * sizeof(double));

    cudaMemcpy(d_c, h_c_flat, size * sizeof(double), cudaMemcpyHostToDevice);

    launch_init_matrices(d_eta, d_tau, d_c, n, tau0, THREADS_PER_BLOCK);

    curandState *d_curand_states;
    cudaMalloc(&d_curand_states, m * sizeof(curandState));
    init_curand_states(d_curand_states, m, seed, THREADS_PER_BLOCK);

    int *d_routes;
    int *d_route_lens;
    double *d_costs;
    bool *d_visited;
    cudaMalloc(&d_routes, m * K * (n + 2) * sizeof(int));
    cudaMalloc(&d_route_lens, m * K * sizeof(int));
    cudaMalloc(&d_costs, m * sizeof(double));
    cudaMalloc(&d_visited, m * (n + 1) * sizeof(bool));

    double *h_costs = (double *)malloc(m * sizeof(double));
    int *h_routes = (int *)malloc(m * K * (n + 2) * sizeof(int));
    int *h_route_lens = (int *)malloc(m * K * sizeof(int));
    double *h_tau = (double *)malloc(size * sizeof(double));

    Solution *iter_best = solution_create(K, n);
    *best_cost = DBL_MAX;
    solution_reset(best_solution);

    for (int iter = 0; iter < T; ++iter) {
        launch_precompute_scores(d_scores, d_eta, d_tau, n, alpha, beta, THREADS_PER_BLOCK);
        
        launch_construct_solutions(m, K, n, d_scores, d_c, d_costs, d_routes, d_route_lens,
                                   d_curand_states, d_visited, THREADS_PER_BLOCK);
        
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

        cudaMemcpy(h_costs, d_costs, m * sizeof(double), cudaMemcpyDeviceToHost);
        
        double iter_best_cost = DBL_MAX;
        int best_ant = -1;

        for (int ant = 0; ant < m; ++ant) {
            if (h_costs[ant] < iter_best_cost) {
                iter_best_cost = h_costs[ant];
                best_ant = ant;
            }
        }

        if (best_ant != -1) {
            cudaMemcpy(h_routes + best_ant * K * (n + 2), 
                       d_routes + best_ant * K * (n + 2), 
                       K * (n + 2) * sizeof(int), 
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_route_lens + best_ant * K, 
                       d_route_lens + best_ant * K, 
                       K * sizeof(int), 
                       cudaMemcpyDeviceToHost);

            solution_reset(iter_best);
            int *best_ant_routes = h_routes + best_ant * K * (n + 2);
            int *best_ant_lens = h_route_lens + best_ant * K;

            for (int vehicle = 0; vehicle < K; ++vehicle) {
                Route *r = &iter_best->routes[vehicle];
                r->len = best_ant_lens[vehicle];
                for (int t = 0; t < r->len; ++t) {
                    r->nodes[t] = best_ant_routes[vehicle * (n + 2) + t];
                }
            }

            if (iter_best_cost < *best_cost) {
                *best_cost = iter_best_cost;
                solution_copy(best_solution, iter_best);
            }

            launch_evaporate_pheromones(d_tau, n, rho, THREADS_PER_BLOCK);
            cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));

            cudaMemcpy(h_tau, d_tau, size * sizeof(double), cudaMemcpyDeviceToHost);
            deposit_pheromones_host(h_tau, iter_best, Q / iter_best_cost, n);
            cudaMemcpy(d_tau, h_tau, size * sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    cudaFree(d_c);
    cudaFree(d_eta);
    cudaFree(d_tau);
    cudaFree(d_scores);
    cudaFree(d_curand_states);
    cudaFree(d_routes);
    cudaFree(d_route_lens);
    cudaFree(d_costs);
    cudaFree(d_visited);

    free(h_c_flat);
    free(h_costs);
    free(h_routes);
    free(h_route_lens);
    free(h_tau);
    solution_free(iter_best);

    return 0;
}
