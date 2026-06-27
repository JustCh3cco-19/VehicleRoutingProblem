#ifndef ACO_CUDA_KERNELS_H
# define ACO_CUDA_KERNELS_H

# include <cuda_runtime.h>
# include <stdint.h>

# define CUDA_THREADS_PER_BLOCK 128
# define CUDA_WARP_SIZE 32
# define CUDA_CACHE_LINE_SIZE 128
# define CUDA_MAX_CANDIDATES 32
# define CUDA_EPS 1e-9f

struct s_cuda_params
{
	int					n;
	int					k;
	int					m;
	int					cap;
	int					cand_k;
	int					route_max_len;
	float				alpha;
	float				beta;
	float				rho;
	float				tau0;
	float				q;
	float				tau_min;
	float				tau_max;
	float				log_tau_min;
	float				log_tau_step;
	uint8_t				q_tau_min;
	uint8_t				q_tau_max;
	uint8_t				q_evap_delta;
	int					visited_l1_words;
	int					visited_l2_words;
	size_t				visited_row_stride;
	float				depot_close_weight;
};
typedef struct s_cuda_params	t_cuda_params;

struct s_cuda_ant_summary
{
	int					feasible;
	int					unvisited_count;
	float				cost;
};
typedef struct s_cuda_ant_summary	t_cuda_ant_summary;

struct s_cuda_iter_stats
{
	unsigned long long	candidate_calls;
	unsigned long long	candidate_moves;
	unsigned long long	fallback_calls;
	unsigned long long	fallback_moves;
	unsigned long long	depot_offer_calls;
	unsigned long long	depot_close_moves;
	unsigned long long	customer_moves;
	unsigned long long	nonempty_routes;
	unsigned long long	fallback_word_groups_scanned;
	unsigned long long	fallback_nodes_scored;
};
typedef struct s_cuda_iter_stats	t_cuda_iter_stats;

__global__ void	kernel_init_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t q_tau0);
__global__ void	kernel_build_candidate_lists(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
__global__ void	kernel_reset_ant_state(int *d_routes, int *d_route_lengths,
					uint64_t *d_visited_l1, uint64_t *d_visited_l2,
					unsigned int *d_rng_states,
					t_cuda_ant_summary *d_ant_summary,
					t_cuda_params params, unsigned int seed);
__global__ void	kernel_construct_solutions(const float2 *d_coords,
					const uint8_t *d_tau, const int *d_candidate_idx,
					const float *d_eta_beta, int *d_routes,
					int *d_route_lengths, uint64_t *d_visited_l1,
					uint64_t *d_visited_l2, unsigned int *d_rng_states,
					t_cuda_ant_summary *d_ant_summary,
					t_cuda_iter_stats *d_iter_stats, t_cuda_params params);
__global__ void	kernel_evaporate_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t delta, uint8_t q_min);
__global__ void	kernel_deposit_solution(uint8_t *d_tau,
					const int *d_routes, const int *d_route_lengths,
					float deposit_amount, int best_ant,
					t_cuda_params params);
__global__ void	kernel_deposit_flat_solution(uint8_t *d_tau,
					const int *d_flat_routes, const int *d_flat_lengths,
					float deposit_amount, t_cuda_params params);

void			launch_init_tau(uint8_t *d_tau, int n, uint8_t q_tau0);
void			launch_build_candidate_lists(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
void			launch_reset_ant_state(int *d_routes, int *d_route_lengths,
					uint64_t *d_visited_l1, uint64_t *d_visited_l2,
					unsigned int *d_rng_states,
					t_cuda_ant_summary *d_ant_summary,
					t_cuda_params params, unsigned int seed);
void			launch_construct_solutions(const float2 *d_coords,
					const uint8_t *d_tau, const int *d_candidate_idx,
					const float *d_eta_beta, int *d_routes,
					int *d_route_lengths, uint64_t *d_visited_l1,
					uint64_t *d_visited_l2, unsigned int *d_rng_states,
					t_cuda_ant_summary *d_ant_summary,
					t_cuda_iter_stats *d_iter_stats, t_cuda_params params);
void			launch_evaporate_tau(uint8_t *d_tau, int n, uint8_t delta,
					uint8_t q_min);
void			launch_deposit_solution(uint8_t *d_tau, const int *d_routes,
					const int *d_route_lengths, float deposit_amount,
					int best_ant, t_cuda_params params);
void			launch_deposit_flat_solution(uint8_t *d_tau,
					const int *d_flat_routes, const int *d_flat_lengths,
					float deposit_amount, t_cuda_params params);

#endif
