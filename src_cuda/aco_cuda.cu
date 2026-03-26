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
#define MAX_CANDIDATES 32

int aco_vrp_cuda(int n, int K, int m, int T, double **c,
                 double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 bool use_uniform_pheromone, int convergence_iter,
                 double epsilon,
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

    // Precompute candidate lists
    int *d_candidates;
    cudaMalloc(&d_candidates, (n + 1) * MAX_CANDIDATES * sizeof(int));
    launch_precompute_candidate_lists(d_candidates, d_c, n, THREADS_PER_BLOCK);

    curandState *d_curand_states;
    cudaMalloc(&d_curand_states, m * sizeof(curandState));
    init_curand_states(d_curand_states, m, seed, THREADS_PER_BLOCK);

    int *d_routes;
    int *d_route_lens;
    double *d_costs;
    cudaMalloc(&d_routes, m * K * (n + 2) * sizeof(int));
    cudaMalloc(&d_route_lens, m * K * sizeof(int));
    cudaMalloc(&d_costs, m * sizeof(double));

    double *h_costs = (double *)malloc(m * sizeof(double));
    int *h_routes = (int *)malloc(m * K * (n + 2) * sizeof(int));
    int *h_route_lens = (int *)malloc(m * K * sizeof(int));

    Solution *iter_best = solution_create(K, n);
    *best_cost = DBL_MAX;
    solution_reset(best_solution);

    int no_improvement_count = 0;

    for (int iter = 0; iter < T; ++iter) {
        launch_precompute_scores(d_scores, d_eta, d_tau, n, alpha, beta, THREADS_PER_BLOCK);
        
        launch_construct_solutions(m, K, n, d_scores, d_c, d_costs, d_routes, d_route_lens,
                                   d_curand_states, d_candidates, THREADS_PER_BLOCK);
        
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
            bool improved_significantly = false;
            
            if (iter_best_cost < *best_cost) {
                double improvement_rel = (*best_cost == DBL_MAX) ? 1.0 : ((*best_cost - iter_best_cost) / *best_cost);
                
                if (improvement_rel > epsilon) {
                    improved_significantly = true;
                }

                *best_cost = iter_best_cost;
                
                cudaMemcpy(h_routes + best_ant * K * (n + 2), 
                           d_routes + best_ant * K * (n + 2), 
                           K * (n + 2) * sizeof(int), 
                           cudaMemcpyDeviceToHost);
                cudaMemcpy(h_route_lens + best_ant * K, 
                           d_route_lens + best_ant * K, 
                           K * sizeof(int), 
                           cudaMemcpyDeviceToHost);

                solution_reset(best_solution);
                int *best_ant_routes = h_routes + best_ant * K * (n + 2);
                int *best_ant_lens = h_route_lens + best_ant * K;

                for (int vehicle = 0; vehicle < K; ++vehicle) {
                    Route *r = &best_solution->routes[vehicle];
                    r->len = best_ant_lens[vehicle];
                    for (int t = 0; t < r->len; ++t) {
                        r->nodes[t] = best_ant_routes[vehicle * (n + 2) + t];
                    }
                }
            }

            if (improved_significantly) {
                no_improvement_count = 0;
            } else {
                no_improvement_count++;
            }

            launch_evaporate_pheromones(d_tau, n, rho, THREADS_PER_BLOCK);
            
            double deposit = use_uniform_pheromone ? Q : (Q / iter_best_cost);
            // Deposit from the current iteration's best ant to guide the search
            launch_deposit_pheromones(d_tau, n, K, 
                                      d_routes + best_ant * K * (n + 2), 
                                      d_route_lens + best_ant * K, 
                                      deposit, 
                                      THREADS_PER_BLOCK);

            if (convergence_iter > 0 && no_improvement_count >= convergence_iter) {
                printf("Converged at iteration %d (relative improvement < %e for %d iterations)\n", iter, epsilon, convergence_iter);
                break;
            }
            if (iter == T - 1) {
                printf("Reached maximum iterations %d\n", T);
            }
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
    cudaFree(d_candidates);

    free(h_c_flat);
    free(h_costs);
    free(h_routes);
    free(h_route_lens);
    solution_free(iter_best);

    return 0;
}
