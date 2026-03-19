#include "aco_cuda_kernels.h"
#include <curand_kernel.h>
#include <math.h>

#define EPS 1e-9

__global__ void init_curand_states_kernel(curandState *states, int m, unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

void init_curand_states(void *d_curand_states, int m, unsigned int seed, int threads_per_block) {
    int blocks = (m + threads_per_block - 1) / threads_per_block;
    init_curand_states_kernel<<<blocks, threads_per_block>>>((curandState *)d_curand_states, m, seed);
}

__global__ void init_matrices_kernel(double *eta, double *tau, const double *c, int n, double tau0) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i == j) {
            eta[idx] = 0.0;
            tau[idx] = 0.0;
        } else {
            eta[idx] = 1.0 / (c[idx] + EPS);
            tau[idx] = tau0;
        }
    }
}

void launch_init_matrices(double *d_eta, double *d_tau, const double *d_c, int n, double tau0, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    init_matrices_kernel<<<blocks, threads_per_block>>>(d_eta, d_tau, d_c, n, tau0);
}

__global__ void precompute_scores_kernel(double *scores, const double *eta, const double *tau, int n, double alpha, double beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i != j) {
            scores[idx] = pow(tau[idx], alpha) * pow(eta[idx], beta);
        } else {
            scores[idx] = 0.0;
        }
    }
}

void launch_precompute_scores(double *d_scores, const double *d_eta, const double *d_tau,
                              int n, double alpha, double beta, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    precompute_scores_kernel<<<blocks, threads_per_block>>>(d_scores, d_eta, d_tau, n, alpha, beta);
}

__global__ void construct_solutions_kernel(int m, int K, int n, const double *scores,
                                           const double *c, double *costs, int *routes,
                                           int *route_lens, curandState *states, bool *visited) {
    int ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= m) return;

    curandState local_state = states[ant];
    bool *my_visited = &visited[ant * (n + 1)];
    for (int i = 0; i <= n; ++i) {
        my_visited[i] = false;
    }

    int *my_routes = &routes[ant * K * (n + 2)];
    int *my_route_lens = &route_lens[ant * K];
    double total_cost = 0.0;
    int unvisited_count = n;

    for (int vehicle = 0; vehicle < K; ++vehicle) {
        int *r = &my_routes[vehicle * (n + 2)];
        int len = 0;
        r[len++] = 0;
        int current = 0;

        while (unvisited_count > 0 && unvisited_count > (K - 1 - vehicle)) {
            double denom = 0.0;
            for (int j = 1; j <= n; ++j) {
                if (!my_visited[j]) {
                    denom += scores[current * (n + 1) + j];
                }
            }

            int next_node = -1;
            if (denom <= 0.0) {
                for (int j = 1; j <= n; ++j) {
                    if (!my_visited[j]) {
                        next_node = j;
                        break;
                    }
                }
                if (next_node == -1) next_node = 0;
            } else {
                double r_val = curand_uniform_double(&local_state);
                double threshold = r_val * denom;
                double cumulative = 0.0;
                int last = 1;
                for (int j = 1; j <= n; ++j) {
                    if (!my_visited[j]) {
                        cumulative += scores[current * (n + 1) + j];
                        last = j;
                        if (cumulative >= threshold) {
                            next_node = j;
                            break;
                        }
                    }
                }
                if (next_node == -1) next_node = last;
            }

            r[len++] = next_node;
            my_visited[next_node] = true;
            --unvisited_count;
            total_cost += c[current * (n + 1) + next_node];
            current = next_node;
        }

        r[len++] = 0;
        total_cost += c[current * (n + 1) + 0];
        my_route_lens[vehicle] = len;
    }

    if (unvisited_count > 0) {
        int vehicle = K - 1;
        int *last_r = &my_routes[vehicle * (n + 2)];
        int len = my_route_lens[vehicle];
        int current = last_r[len - 2];
        total_cost -= c[current * (n + 1) + 0];
        len--;

        for (int j = 1; j <= n; ++j) {
            if (!my_visited[j]) {
                last_r[len++] = j;
                my_visited[j] = true;
                total_cost += c[current * (n + 1) + j];
                current = j;
            }
        }
        last_r[len++] = 0;
        total_cost += c[current * (n + 1) + 0];
        my_route_lens[vehicle] = len;
    }

    costs[ant] = total_cost;
    states[ant] = local_state;
}

void launch_construct_solutions(int m, int K, int n, const double *d_scores,
                                const double *d_c, double *d_costs, int *d_routes,
                                int *d_route_lens, void *d_curand_states, bool *d_visited, int threads_per_block) {
    int blocks = (m + threads_per_block - 1) / threads_per_block;
    construct_solutions_kernel<<<blocks, threads_per_block>>>(m, K, n, d_scores, d_c, d_costs,
                                                              d_routes, d_route_lens,
                                                              (curandState *)d_curand_states, d_visited);
}

__global__ void evaporate_pheromones_kernel(double *tau, int n, double rho) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (n + 1) * (n + 1);
    if (idx < size) {
        int i = idx / (n + 1);
        int j = idx % (n + 1);
        if (i != j) {
            tau[idx] *= (1.0 - rho);
        }
    }
}

void launch_evaporate_pheromones(double *d_tau, int n, double rho, int threads_per_block) {
    int size = (n + 1) * (n + 1);
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    evaporate_pheromones_kernel<<<blocks, threads_per_block>>>(d_tau, n, rho);
}
