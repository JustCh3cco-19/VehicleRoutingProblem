#ifndef MPI_INTERNAL_H
# define MPI_INTERNAL_H

#include "solver.h"
#include "config.h"
#include <stddef.h>
#include <stdint.h>


# ifdef USE_MPI
#include <mpi.h>
# endif

enum e_par_constants
{
	par_alignment = 64,
	par_max_candidates = 512
};

struct s_par_sparse_delta
{
	uint32_t				edge_idx;
	float					increment;
};
typedef struct s_par_sparse_delta	t_par_sparse_delta;

struct s_par_matrix
{
	int						n;
	int						stride;
	float					*data;
	float					**rows;
};
typedef struct s_par_matrix	t_par_matrix;

struct s_par_shared
{
	int						n;
	int						cand_k;
	int						stride;
	int						visited_words;
	int						meta_words;
	int						*cand_idx;
	float					*eta_beta;
};
typedef struct s_par_shared	t_par_shared;

struct s_par_workspace
{
	t_solution				*sol;
	uint64_t				*visited;
	uint64_t				*meta_active;
	int						*route_loads;
	unsigned int			rng_state;
	t_solution				*thread_best;
};
typedef struct s_par_workspace	t_par_workspace;

struct s_par_tour_ctx
{
	t_par_workspace				*ws;
	const t_par_shared			*s;
	int							k;
	int							cap;
	const t_par_matrix			*c;
	const float					*scores;
	int							remaining;
};
typedef struct s_par_tour_ctx	t_par_tour_ctx;

struct s_par_select_ctx
{
	const int					*cands;
	const float					*sc;
	int							t;
	int							node;
	float						w;
	float						denom;
	float						thres;
	float						cum;
	int							nodes[1024];
	float						scores[1024];
	int						count;
};
typedef struct s_par_select_ctx	t_par_select_ctx;

struct s_nearest_state
{
	int			best;
	float		best_d;
	const float	*row;
};
typedef struct s_nearest_state	t_nearest_state;

struct s_cand_list
{
	int		*nodes;
	float	*dists;
	int		k;
};
typedef struct s_cand_list	t_cand_list;

struct s_cand_init_params
{
	int						n;
	int						cand_k;
	const t_par_matrix		*c_mat;
	double					beta;
};
typedef struct s_cand_init_params	t_cand_init_params;

# ifdef USE_MPI
struct s_par_async_context
{
	MPI_Request				req;
	t_par_sparse_delta		*recv_buf;
	int						*counts;
	int						*displs;
	MPI_Datatype			type;
	int						active;
};
typedef struct s_par_async_context	t_par_async_context;
# endif

size_t				par_align_up(size_t value);
void				*par_aligned_calloc(size_t bytes);
double				par_wall_time(void);
float				par_fast_powf(float base, float exp);
double				par_rand01(unsigned int *state);
bool				par_route_append(t_route *r, int node);
int					par_is_improvement(double prev_best, double new_best,
						double min_rel_improvement);
t_par_matrix		*par_matrix_create(int n);
void				par_matrix_free(t_par_matrix *m);
int					par_choose_candidate_count(int n,
						int requested_candidate_k);
int					par_shared_init(t_par_shared *s,
						const t_cand_init_params *params);
void				par_shared_free(t_par_shared *s);

int					par_ws_init(t_par_workspace *ws, int k,
						const t_par_shared *s);
void				par_ws_free(t_par_workspace *ws);
int					par_nearest_unvisited(const t_par_shared *s, int curr,
						const t_par_workspace *ws,
						const t_par_matrix *c);
bool				par_build_ant(t_par_tour_ctx *ctx);
double				par_solution_cost(const t_solution *s, float **c);

# ifdef USE_MPI
void				par_async_init(t_par_async_context *ctx, int mpi_size);
void				par_async_cleanup(t_par_async_context *ctx);
void				par_async_wait_and_apply(t_par_async_context *ctx,
						t_par_matrix *tau, int mpi_rank, int mpi_size);
void				par_async_start(t_par_async_context *ctx,
						t_par_sparse_delta *local_deltas,
						int local_count, int mpi_size);
# endif

struct s_par_solver_ctx
{
	int						n;
	int						k;
	int						cap;
	int						m;
	double					**c;
	double					alpha;
	double					beta;
	double					rho;
	double					tau0;
	double					q;
	unsigned int			seed;
	t_solution				*best_sol;
	double					*best_cost;
	const t_config		*config;
	int						mpi_rank;
	int						mpi_size;
	int						cand_k;
	int						total_m;
	int						ant_off;
	int						local_m;
	double					max_runtime_sec;
	int						max_stagnation_epochs;
	double					min_rel_improvement;
	double					progress_interval_sec;
	int						log_level;
	int						fixed_epochs;
	t_par_matrix			*tau_mat;
	t_par_matrix			*c_mat;
	t_par_shared			shared;
	float					*score_mat;
	t_solution				*iter_best_sol_rank;
	double					iter_best_cost_g;
	double					start_time;
	double					next_progress_time;
	int						iter_since_best;
	t_par_sparse_delta		*rank_deltas;
	int						rank_delta_count;
	int						workspace_failed;
# ifdef USE_MPI
	t_par_async_context		async_ctx;
# endif
};
typedef struct s_par_solver_ctx	t_par_solver_ctx;

int					par_solver_alloc(t_par_solver_ctx *ctx);
int					par_solver_init(t_par_solver_ctx *ctx);
void				par_solver_free(t_par_solver_ctx *ctx);
void				par_solver_scores(t_par_solver_ctx *ctx);
void				par_solver_ants(t_par_solver_ctx *ctx, t_par_workspace *ws,
						int iter);
void				par_solver_reduce_best(t_par_solver_ctx *ctx, int iter);
void				par_solver_evaporate(t_par_solver_ctx *ctx);
void				par_solver_deposit(t_par_solver_ctx *ctx);
void				par_solver_log_start(t_par_solver_ctx *ctx);
void				par_solver_thread_run(t_par_solver_ctx *ctx);

/* Compatibility Typedefs for Non-Refactored code */
typedef t_par_sparse_delta		aco_mpi_sparse_delta_t;
typedef t_par_matrix			aco_mpi_matrix_float_t;
typedef t_par_shared			aco_mpi_rank_shared_t;
typedef t_par_workspace			aco_mpi_workspace_t;
# ifdef USE_MPI
typedef t_par_async_context		aco_mpi_async_sparse_context_t;
# endif


#endif
