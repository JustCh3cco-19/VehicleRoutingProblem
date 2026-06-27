#ifndef SEQ_INTERNAL_H
# define SEQ_INTERNAL_H

# include "../internal.h"
# include <stdbool.h>

struct s_seq_shared
{
	int						n;
	int						candidate_k;
	int						stride;
	int						visited_words;
	int						*candidate_idx;
	float					*eta_beta;
	float					*score;
};
typedef struct s_seq_shared	t_seq_shared;

struct s_seq_workspace
{
	t_solution				*sol;
	uint64_t				*visited;
	int						*route_loads;
	unsigned int			rng_state;
};
typedef struct s_seq_workspace	t_seq_workspace;
typedef t_seq_shared		SeqShared;
typedef t_seq_workspace		SeqWorkspace;

struct s_select_ctx
{
	const int				*cand_row;
	const float				*score_row;
	int						k;
	int						t;
	int						node;
	double					w;
	double					denom;
	double					threshold;
	double					cumulative;
	int						last_valid;
	int						nodes[1024];
	double					scores[1024];
	int						count;
};

struct s_aco_params
{
	int						n;
	int						k;
	int						m;
	int						vehicle_capacity_customers;
	double					alpha;
	double					beta;
	double					rho;
	double					tau0;
	double					q;
	unsigned int			seed;
};
typedef struct s_aco_params	t_aco_params;

struct s_seq_ctx
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
	t_aco_config			params;
	double					**tau;
	t_solution				*iter_best;
	t_seq_shared			shared;
	t_seq_workspace			ws;
	int						stagnation_iters;
	int						stagnation_trigger;
	double			        tau_max;
	double			        tau_min;
	double					start_wall;
	double					next_progress_wall;
	int						no_improve_epochs;
	int						iter;
	double					iter_best_cost;
	int						iter_best_ant;
	int						improved_global;
};
typedef struct s_seq_ctx	t_seq_ctx;

static inline int	visited_is_set(const uint64_t *visited, int node)
{
	return ((int)((visited[(unsigned int)node >> 6] >>
				((unsigned int)node & 63u)) &
				1u));
}

static inline void	visited_set(uint64_t *visited, int node)
{
	visited[(unsigned int)node >> 6] |= (uint64_t)1u
		<< ((unsigned int)node & 63u);
}

/* Sequential Utilities */
enum e_seq_constants
{
	kSeqAcoAlignment = 64,
	kSeqDefaultSparseCandidateCount = 64,
	kSeqDenseCandidateLimit = 512
};

size_t				seq_align_up(size_t value, size_t alignment);
void				*seq_aligned_calloc(size_t bytes);
int					seq_clamp(int x, int lo, int hi);
double				seq_fast_pow(double base, double exponent);
int					seq_stride(int cols, size_t elem_size);
int					seq_choose_candidate_count(int n, int requested_candidate_k);
double				seq_wall_time(void);
int					seq_is_improvement(double prev_best,
						double new_best, double min_rel_improvement);
int					seq_choose_auto_ants(int n);

/* Sequential Candidate & Scoring Management (candidates.c) */
int					seq_shared_init(t_seq_shared *shared, int n, int candidate_k);
void				seq_shared_free(t_seq_shared *shared);
void				seq_shared_build_candidates(t_seq_shared *shared, double **c,
						double beta);
void				seq_shared_update_scores(t_seq_shared *shared,
						double **restrict tau, double alpha);

/* Sequential Tour & Workspace Management (tour.c, tour_select.c, tour_select_large.c, workspace.c) */
int					seq_workspace_init(t_seq_workspace *ws, int k, int n,
						int visited_words);
void				seq_workspace_free(t_seq_workspace *ws);
bool				build_ant_solution(t_seq_workspace *ws,
						const t_seq_shared *shared, int k,
						int vehicle_capacity_customers, double **restrict c);
double				seq_rand01(unsigned int *state);
int					find_nearest_unvisited(const t_seq_shared *shared, int current,
						const uint64_t *restrict visited, double **restrict c);
int					select_small(const t_seq_shared *shared, int current,
						const uint64_t *visited, unsigned int *rng_state);
int					select_large(const t_seq_shared *shared, int current,
						const uint64_t *visited, unsigned int *rng_state);
int					select_next_customer(const t_seq_shared *shared, int current,
						const uint64_t *visited, double **c, unsigned int *rng_state);

/* Sequential Context and Lifecycle Helpers (seq_ctx.c, seq_epoch.c, seq_pheromone.c) */
int					allocate_ctx(t_seq_ctx *ctx);
void				init_ctx(t_seq_ctx *ctx);
void				cleanup_ctx(t_seq_ctx *ctx);
void				run_ant(t_seq_ctx *ctx, int ant);
void				run_epoch(t_seq_ctx *ctx);
void				evaporate_tau(t_seq_ctx *ctx);
void				deposit_iter_best(t_seq_ctx *ctx);
void				deposit_global_best(t_seq_ctx *ctx);
void				clamp_pheromones(t_seq_ctx *ctx);
void				reset_pheromones(t_seq_ctx *ctx);

/* Candidates AVX2 Helpers (candidates_avx.c) */
void				update_score_row_alpha1(const int *restrict cand_row,
						const float *restrict eta_row, float *restrict score_row,
						const double *restrict tau_row, int k);
void				update_score_row_alpha2(const int *restrict cand_row,
						const float *restrict eta_row, float *restrict score_row,
						const double *restrict tau_row, int k);

#endif
