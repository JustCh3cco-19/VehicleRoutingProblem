#ifndef ACO_CUDA_KERNELS_H
#define ACO_CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

void launch_init_matrices(double *d_eta, double *d_tau, const double *d_c, int n, double tau0, int threads_per_block);

void launch_precompute_scores(double *d_scores, const double *d_eta, const double *d_tau,
                              int n, double alpha, double beta, int threads_per_block);

void launch_construct_solutions(int m, int K, int n, const double *d_scores,
                                const double *d_c, double *d_costs, int *d_routes,
                                int *d_route_lens, void *d_curand_states, bool *d_visited, int threads_per_block);

void launch_evaporate_pheromones(double *d_tau, int n, double rho, int threads_per_block);

void launch_deposit_pheromones(double *d_tau, int n, int K, const int *d_routes, const int *d_route_lens, double deposit, int threads_per_block);

void init_curand_states(void *d_curand_states, int m, unsigned int seed, int threads_per_block);

#ifdef __cplusplus
}
#endif

#endif
