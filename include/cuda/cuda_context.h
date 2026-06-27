#ifndef CUDA_CONTEXT_H
# define CUDA_CONTEXT_H

#include "cuda_types.h"

extern "C" {
#include "cli_common.h"
#include "config.h"
#include "instance_parser.h"
#include "solution.h"
#include "solver.h"
}

struct s_cuda_solver_ctx
{
	t_solver_params		*params;
	t_cuda_coords		coords;
	t_solution			*best_sol;
	double				*best_cost;
	t_config			config;
	t_cuda_params		cuda_params;
	int					m;
	int					cand_k;
	uint8_t				q_tau0;
	float2				*h_coords;
	t_cuda_ant_summary	*h_ant_summary;
	int					*h_routes;
	int					*h_route_lengths;
	int					*h_flat_routes;
	int					*h_flat_lengths;
	float2				*d_coords;
	uint8_t				*d_tau;
	int					*d_candidate_idx;
	float				*d_eta_beta;
	t_cuda_ant_state	ants;
	t_cuda_iter_stats	*d_iter_stats;
	t_cuda_flat_routes	flat;
	int					max_steps;
	double				max_runtime;
	int					max_stagnation;
	double				min_rel_imp;
	double				progress_int;
	int					iter;
	t_cuda_iter_stats	accum_stats;
	int					iter_since_best;
	double				start_time;
	double				next_progress;
	double				global_best;
	size_t				total_elements;
	size_t				tau_elements;
};
typedef struct s_cuda_solver_ctx	t_cuda_solver_ctx;

struct s_cuda_main_ctx
{
	t_cli_options		options;
	t_vrp_instance		instance;
	float				*coords_x;
	float				*coords_y;
	t_solution			*best;
	double				best_cost;
};
typedef struct s_cuda_main_ctx	t_cuda_main_ctx;

#endif
