#ifndef INTERNAL_H
# define INTERNAL_H

# include "solver.h"
# include "config.h"
# include <stddef.h>
# include <stdint.h>

struct s_aco_score_cache
{
	int						n;
	int						line_len;
	int						l1_lines;
	int						l2_lines;
	int						*l1_keys;
	int						*l2_keys;
	double					*l1_rows;
	double					*l2_rows;
	size_t					l1_hits;
	size_t					l2_hits;
	size_t					l3_misses;
};
typedef struct s_aco_score_cache	t_score_cache;

struct s_aco_cache_stats
{
	size_t					l1_hits;
	size_t					l2_hits;
	size_t					l3_misses;
};
typedef struct s_aco_cache_stats	t_cache_stats;

struct s_aco_rank_shared
{
	int						n;
	int						candidate_k;
	int						stride;
	int						visited_words;
	int						*candidate_idx;
	float					*eta_beta;
};
typedef struct s_aco_rank_shared	t_rank_shared;

struct s_aco_thread_workspace
{
	t_solution				*sol;
	t_solution				*thread_best;
	uint64_t				*visited;
	int						*route_loads;
	unsigned int			rng_state;
};
typedef struct s_aco_thread_workspace	t_thread_workspace;

struct s_score_params
{
	double					**tau;
	double					**eta;
	double					alpha;
	double					beta;
};
typedef struct s_score_params	t_score_params;

struct s_shared_select_ctx
{
	int						current;
	const int				*unvisited_nodes;
	int						unvisited_count;
	const t_score_params	*score_params;
	double					roulette_r;
	double					*candidate_scores;
	int						*selected_index;
	t_score_cache			*score_cache;
};
typedef struct s_shared_select_ctx	t_shared_select_ctx;

struct s_ant_build_ctx
{
	t_solution				*sol;
	int						n;
	int						k;
	const t_score_params	*score_params;
	int						vehicle_capacity_customers;
	t_score_cache			*score_cache;
	unsigned int			*rng_state;
	int						*unvisited_nodes;
	double					*candidate_scores;
	double					*random_draws;
};
typedef struct s_ant_build_ctx	t_ant_build_ctx;

/* Shared helpers */
double				fast_pow_nonneg(double base, double exponent);
const double		*score_cache_get_row(t_score_cache *cache, int current,
						const t_score_params *params);
int					shared_select_next(t_shared_select_ctx *ctx);
bool				shared_build_ant_solution(t_ant_build_ctx *ctx);

t_score_cache	*score_cache_create(int n, int l1_lines, int l2_lines);
void				score_cache_invalidate(t_score_cache *cache);
void				score_cache_reset_stats(t_score_cache *cache);
void				score_cache_get_stats(const t_score_cache *cache,
						t_cache_stats *out);
void				score_cache_free(t_score_cache *cache);

#endif
