#ifndef CUDA_KERNELS_H
# define CUDA_KERNELS_H

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

struct s_cuda_coords
{
	float				*x;
	float				*y;
};
typedef struct s_cuda_coords	t_cuda_coords;


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

struct s_cuda_ant_state
{
	int					*routes;
	int					*route_lengths;
	uint64_t			*visited_l1;
	uint64_t			*visited_l2;
	unsigned int		*rng_states;
	t_cuda_ant_summary	*ant_summary;
};
typedef struct s_cuda_ant_state	t_cuda_ant_state;

struct s_cuda_problem_data
{
	const float2		*coords;
	const uint8_t		*tau;
	const int			*candidate_idx;
	const float			*eta_beta;
};
typedef struct s_cuda_problem_data	t_cuda_problem_data;

struct s_cuda_deposit_params
{
	float				amount;
	int					best_ant;
};
typedef struct s_cuda_deposit_params	t_cuda_deposit_params;

struct s_cuda_flat_routes
{
	int					*routes;
	int					*lengths;
};
typedef struct s_cuda_flat_routes	t_cuda_flat_routes;

struct s_cuda_select_ctx
{
	int					current;
	uint64_t			*visited_l1;
	uint64_t			*visited_l2;
	unsigned int		*rng_state;
	float				*total_score;
};
typedef struct s_cuda_select_ctx	t_cuda_select_ctx;

struct s_cuda_fallback_state
{
	float2				p_curr;
	float				local_sum;
	int					local_selected;
	int					groups_scanned;
	int					nodes_scored;
	t_cuda_params		params;
};
typedef struct s_cuda_fallback_state	t_cuda_fallback_state;

struct s_cuda_construct_ctx
{
	int					ant;
	int					remaining;
	float				total_cost;
	int					global_step;
	unsigned int		rng_state;
	int					vehicle;
	t_cuda_iter_stats	*iter_stats;
	t_cuda_params		params;
	int					current;
	int					route_load;
	int					route_len;
	float				score_cand;
	int					chosen_fallback;
};
typedef struct s_cuda_construct_ctx	t_cuda_construct_ctx;


__global__ void	kern_init_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t q_tau0);
__global__ void	kern_build_candidates(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
__global__ void	kern_reset_ants(t_cuda_ant_state ants,
					t_cuda_params params, unsigned int seed);
__global__ void	kern_construct_solutions(t_cuda_problem_data prob,
					t_cuda_ant_state ants, t_cuda_iter_stats *d_iter_stats,
					t_cuda_params params);
__global__ void	kern_evaporate_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t delta, uint8_t q_min);
__global__ void	kern_deposit_best(uint8_t *d_tau, t_cuda_ant_state ants,
					t_cuda_deposit_params dep, t_cuda_params params);
__global__ void	kern_deposit_flat(uint8_t *d_tau, t_cuda_flat_routes flat,
					float deposit_amount, t_cuda_params params);

void			launch_init_tau(uint8_t *d_tau, int n, uint8_t q_tau0);
void			launch_build_candidates(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params);
void			launch_reset_ants(t_cuda_ant_state ants,
					t_cuda_params params, unsigned int seed);
void			launch_construct_solutions(t_cuda_problem_data prob,
					t_cuda_ant_state ants, t_cuda_iter_stats *d_iter_stats,
					t_cuda_params params);
void			launch_evaporate_tau(uint8_t *d_tau, int n, uint8_t delta,
					uint8_t q_min);
void			launch_deposit_best(uint8_t *d_tau, t_cuda_ant_state ants,
					t_cuda_deposit_params dep, t_cuda_params params);
void			launch_deposit_flat(uint8_t *d_tau, t_cuda_flat_routes flat,
					float deposit_amount, t_cuda_params params);

#endif
