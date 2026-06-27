extern "C" {
# include "solver.h"
# include "config.h"
# include "instance_parser.h"
# include "matrix.h"
# include "solution.h"
}

# include "cuda/cuda_kernels.h"

# include <cuda_runtime.h>
# include <float.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>

# define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err__));                                    \
      status = SOLVER_ERR_BACKEND;                                           \
      goto cleanup;                                                          \
    }                                                                        \
  } while (0)

# define CHECK_CUDA_KERNEL() CHECK_CUDA(cudaGetLastError())

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

static double	wall_time_seconds(void)
{
	struct timespec	ts;

	timespec_get(&ts, TIME_UTC);
	return ((double)ts.tv_sec + (double)ts.tv_nsec * 1e-9);
}

static int	is_significant_improvement(double prev_best, double new_best,
				double min_rel_improvement)
{
	double	abs_gain;
	double	rel_gain;

	if (prev_best >= DBL_MAX || new_best >= DBL_MAX)
		return (new_best < prev_best);
	if (new_best >= prev_best - SOLVER_EPS)
		return (0);
	abs_gain = prev_best - new_best;
	rel_gain = abs_gain / fmax(prev_best, SOLVER_EPS);
	return (rel_gain + SOLVER_EPS >= min_rel_improvement);
}

static int	select_iter_best_host(const t_cuda_ant_summary *summary, int m,
				int *best_idx, float *best_cost)
{
	int		found;
	int		idx;
	float	cost;
	int		i;

	found = 0;
	idx = -1;
	cost = FLT_MAX;
	i = 0;
	for (; i < m; ++i)
	{
		if (!summary[i].feasible)
			continue ;
		if (!found || summary[i].cost < cost
			|| (fabsf(summary[i].cost - cost) <= (float)CUDA_EPS && i < idx))
		{
			found = 1;
			idx = i;
			cost = summary[i].cost;
		}
	}
	*best_idx = idx;
	*best_cost = cost;
	return (found);
}

static int	copy_ant_to_solution(const int *routes, const int *route_lengths,
				int k, int m, int ant_idx,
				t_solution *dst)
{
	int		v;
	int		global_step;
	int		len;
	t_route	*r;
	int		t;

	solution_reset(dst);
	global_step = 0;
	v = 0;
	for (; v < k; ++v)
	{
		len = route_lengths[ant_idx * k + v];
		r = &dst->routes[v];
		if (len < 2 || len > r->cap)
			return (0);
		r->len = len;
		t = 0;
		for (; t < len; ++t)
			r->nodes[t] = routes[(global_step + t) * m + ant_idx];
		global_step += len - 1;
	}
	return (1);
}

static void	init_tau_constants(t_cuda_solver_ctx *ctx)
{
	float	log_tau_min;
	float	log_tau_max;
	float	log_rho;

	log_tau_min = logf(0.0001f);
	log_tau_max = logf(100.0f);
	ctx->cuda_params.log_tau_min = log_tau_min;
	ctx->cuda_params.log_tau_step = (log_tau_max - log_tau_min) / 255.0f;
	log_rho = logf(1.0f - (float)ctx->params->rho);
	ctx->cuda_params.q_evap_delta = (uint8_t)fmaxf(1.0f,
			roundf(-log_rho / ctx->cuda_params.log_tau_step));
	ctx->q_tau0 = (uint8_t)fmaxf(0.0f, fminf(255.0f,
				roundf((logf(ctx->cuda_params.tau0) - log_tau_min)
					/ ctx->cuda_params.log_tau_step)));
}

static void	init_cuda_params_basic(t_cuda_solver_ctx *ctx)
{
	ctx->cuda_params.n = ctx->params->n;
	ctx->cuda_params.k = ctx->params->k;
	ctx->cuda_params.m = ctx->m;
	ctx->cuda_params.cap = ctx->params->vehicle_capacity_customers;
	ctx->cuda_params.cand_k = ctx->cand_k;
	ctx->cuda_params.route_max_len = ctx->params->k * (ctx->params->n + 1);
	ctx->cuda_params.alpha = (float)ctx->params->alpha;
	ctx->cuda_params.beta = (float)ctx->params->beta;
	ctx->cuda_params.rho = (float)ctx->params->rho;
	ctx->cuda_params.tau0 = (float)ctx->params->tau0;
	ctx->cuda_params.q = (float)ctx->params->q * 100.0f;
	ctx->cuda_params.tau_min = 0.0001f;
	ctx->cuda_params.tau_max = 100.0f;
	ctx->cuda_params.q_tau_min = 0;
	ctx->cuda_params.q_tau_max = 255;
}

static t_status	cuda_init_params(t_cuda_solver_ctx *ctx)
{
	runtime_config_load_env(&ctx->config);
	ctx->config.ants = ctx->params->m;
	ctx->config.seed = ctx->params->seed;
	ctx->m = (ctx->params->m > 0) ? ctx->params->m : 256;
	ctx->cand_k = (ctx->config.candidate_k > 0) ? ctx->config.candidate_k : 32;
	if (ctx->cand_k > ctx->params->n)
		ctx->cand_k = ctx->params->n;
	if (ctx->cand_k < 1)
		ctx->cand_k = 1;
	init_cuda_params_basic(ctx);
	init_tau_constants(ctx);
	ctx->cuda_params.visited_l1_words = (ctx->params->n + 64) / 64;
	ctx->cuda_params.visited_l2_words = (ctx->cuda_params.visited_l1_words + 63) / 64;
	ctx->cuda_params.visited_row_stride = ((ctx->cuda_params.visited_l1_words * 8 + 127) / 128) * 128;
	ctx->cuda_params.depot_close_weight = 2.0f;
	return (SOLVER_OK);
}

static t_status	cuda_alloc_host(t_cuda_solver_ctx *ctx)
{
	int	i;

	ctx->h_coords = (float2 *)malloc((ctx->params->n + 1) * sizeof(float2));
	if (!ctx->h_coords)
		return (SOLVER_ERR_ALLOCATION);
	i = 0;
	for (; i <= ctx->params->n; i++)
	{
		ctx->h_coords[i].x = ctx->coords.x[i];
		ctx->h_coords[i].y = ctx->coords.y[i];
	}
	ctx->total_elements = (size_t)(ctx->params->n + 1) * (size_t)(ctx->params->n + 1);
	ctx->tau_elements = (ctx->total_elements + 3) & ~3ull;
	ctx->h_ant_summary = (t_cuda_ant_summary *)malloc(ctx->m * sizeof(t_cuda_ant_summary));
	ctx->max_steps = ctx->cuda_params.route_max_len + 1;
	ctx->h_routes = (int *)malloc(ctx->max_steps * ctx->m * sizeof(int));
	ctx->h_route_lengths = (int *)malloc(ctx->m * ctx->params->k * sizeof(int));
	ctx->h_flat_routes = (int *)malloc(ctx->params->k * (ctx->params->n + 1) * sizeof(int));
	ctx->h_flat_lengths = (int *)malloc(ctx->params->k * sizeof(int));
	if (!ctx->h_ant_summary || !ctx->h_routes || !ctx->h_route_lengths
		|| !ctx->h_flat_routes || !ctx->h_flat_lengths)
		return (SOLVER_ERR_ALLOCATION);
	return (SOLVER_OK);
}

static t_status	cuda_alloc_device(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMalloc(&ctx->d_coords, (ctx->params->n + 1) * sizeof(float2)));
	CHECK_CUDA(cudaMalloc(&ctx->d_tau, ctx->tau_elements * sizeof(uint8_t)));
	CHECK_CUDA(cudaMalloc(&ctx->d_candidate_idx, (ctx->params->n + 1) * ctx->cand_k * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->d_eta_beta, (ctx->params->n + 1) * ctx->cand_k * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.routes, ctx->max_steps * ctx->m * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.route_lengths, ctx->m * ctx->params->k * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.visited_l1, ctx->m * ctx->cuda_params.visited_row_stride));
	CHECK_CUDA(cudaMalloc(&ctx->ants.visited_l2, ctx->m * ctx->cuda_params.visited_l2_words * sizeof(uint64_t)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.rng_states, ctx->m * sizeof(unsigned int)));
	CHECK_CUDA(cudaMalloc(&ctx->ants.ant_summary, ctx->m * sizeof(t_cuda_ant_summary)));
	CHECK_CUDA(cudaMalloc(&ctx->d_iter_stats, sizeof(t_cuda_iter_stats)));
	CHECK_CUDA(cudaMalloc(&ctx->flat.routes, ctx->params->k * (ctx->params->n + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&ctx->flat.lengths, ctx->params->k * sizeof(int)));
	return (SOLVER_OK);
cleanup:
	return (status);
}

static t_status	cuda_init_device(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMemcpy(ctx->d_coords, ctx->h_coords,
			(ctx->params->n + 1) * sizeof(float2), cudaMemcpyHostToDevice));
	launch_init_tau(ctx->d_tau, ctx->params->n, ctx->q_tau0);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	launch_build_candidates(ctx->d_coords, ctx->d_candidate_idx,
		ctx->d_eta_beta, ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	return (SOLVER_OK);
cleanup:
	return (status);
}

static t_status	cuda_iter_step(t_cuda_solver_ctx *ctx, t_cuda_problem_data prob)
{
	t_status	status;

	status = SOLVER_OK;
	launch_reset_ants(ctx->ants, ctx->cuda_params, ctx->params->seed + ctx->iter);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemset(ctx->d_iter_stats, 0, sizeof(t_cuda_iter_stats)));
	launch_construct_solutions(prob, ctx->ants, ctx->d_iter_stats,
		ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_CUDA(cudaMemcpy(ctx->h_ant_summary, ctx->ants.ant_summary,
			ctx->m * sizeof(t_cuda_ant_summary), cudaMemcpyDeviceToHost));
	return (SOLVER_OK);
cleanup:
	return (status);
}

static void	update_accum_stats(t_cuda_solver_ctx *ctx)
{
	t_cuda_iter_stats	host_stats;

	if (cudaMemcpy(&host_stats, ctx->d_iter_stats, sizeof(t_cuda_iter_stats),
			cudaMemcpyDeviceToHost) == cudaSuccess)
	{
		ctx->accum_stats.candidate_moves += host_stats.candidate_moves;
		ctx->accum_stats.fallback_calls += host_stats.fallback_calls;
		ctx->accum_stats.fallback_moves += host_stats.fallback_moves;
		ctx->accum_stats.depot_offer_calls += host_stats.depot_offer_calls;
		ctx->accum_stats.depot_close_moves += host_stats.depot_close_moves;
		ctx->accum_stats.customer_moves += host_stats.customer_moves;
		ctx->accum_stats.nonempty_routes += host_stats.nonempty_routes;
		ctx->accum_stats.fallback_word_groups_scanned
			+= host_stats.fallback_word_groups_scanned;
		ctx->accum_stats.fallback_nodes_scored
			+= host_stats.fallback_nodes_scored;
	}
}

static void	update_flat_solution(t_cuda_solver_ctx *ctx, int best_idx)
{
	int	v;
	int	i;

	v = 0;
	for (; v < ctx->params->k; v++)
	{
		ctx->h_flat_lengths[v] = ctx->best_sol->routes[v].len;
		i = 0;
		for (; i < ctx->best_sol->routes[v].len; i++)
		{
			ctx->h_flat_routes[v * (ctx->params->n + 1) + i]
				= ctx->best_sol->routes[v].nodes[i];
		}
	}
	cudaMemcpy(ctx->flat.routes, ctx->h_flat_routes,
		ctx->params->k * (ctx->params->n + 1) * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(ctx->flat.lengths, ctx->h_flat_lengths,
		ctx->params->k * sizeof(int), cudaMemcpyHostToDevice);
}

static t_status	copy_best_data(t_cuda_solver_ctx *ctx, int best_idx)
{
	t_status	status;

	status = SOLVER_OK;
	CHECK_CUDA(cudaMemcpy(ctx->h_routes, ctx->ants.routes,
			ctx->max_steps * ctx->m * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(ctx->h_route_lengths, ctx->ants.route_lengths,
			ctx->m * ctx->params->k * sizeof(int), cudaMemcpyDeviceToHost));
	copy_ant_to_solution(ctx->h_routes, ctx->h_route_lengths,
		ctx->params->k, ctx->m, best_idx, ctx->best_sol);
	update_flat_solution(ctx, best_idx);
	return (SOLVER_OK);
cleanup:
	return (status);
}

static t_status	update_global_best(t_cuda_solver_ctx *ctx, int best_idx,
					float best_iter_cost)
{
	double		new_best;

	new_best = (double)best_iter_cost;
	if (new_best < ctx->global_best)
	{
		if (is_significant_improvement(ctx->global_best, new_best,
				ctx->min_rel_imp))
			ctx->iter_since_best = 0;
		else
			ctx->iter_since_best++;
		ctx->global_best = new_best;
		*(ctx->best_cost) = ctx->global_best;
		return (copy_best_data(ctx, best_idx));
	}
	ctx->iter_since_best++;
	return (SOLVER_OK);
}

static t_status	update_pheromones(t_cuda_solver_ctx *ctx, int best_idx,
					float best_iter_cost)
{
	t_status				status;
	t_cuda_deposit_params	dep;

	status = SOLVER_OK;
	launch_evaporate_tau(ctx->d_tau, ctx->params->n,
		ctx->cuda_params.q_evap_delta, ctx->cuda_params.q_tau_min);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	dep.amount = (0.3f * ctx->cuda_params.q) / best_iter_cost;
	dep.best_ant = best_idx;
	launch_deposit_best(ctx->d_tau, ctx->ants, dep, ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	launch_deposit_flat(ctx->d_tau, ctx->flat,
		(0.7f * ctx->cuda_params.q) / (float)ctx->global_best,
		ctx->cuda_params);
	CHECK_CUDA_KERNEL();
	CHECK_CUDA(cudaDeviceSynchronize());
	return (SOLVER_OK);
cleanup:
	return (status);
}

static void	log_progress(t_cuda_solver_ctx *ctx, double progress_time)
{
	double	elapsed;

	elapsed = progress_time - ctx->start_time;
	if (ctx->max_runtime > 0.0)
	{
		double remaining = ctx->max_runtime - elapsed;
		if (remaining < 0.0)
			remaining = 0.0;
		fprintf(stderr,
			"[cuda] elapsed %.1fs, remaining %.1fs, iter %d, best %.3f\n",
			elapsed, remaining, ctx->iter + 1, ctx->global_best);
	}
	else
	{
		fprintf(stderr, "[cuda] elapsed %.1fs, iter %d, best %.3f\n",
			elapsed, ctx->iter + 1, ctx->global_best);
	}
	ctx->next_progress = progress_time + ctx->progress_int;
}

static int	should_stop(t_cuda_solver_ctx *ctx)
{
	double	current_time;

	current_time = wall_time_seconds();
	if (ctx->max_runtime > 0.0
		&& (current_time - ctx->start_time) > ctx->max_runtime)
		return (1);
	if (ctx->max_stagnation > 0
		&& ctx->iter_since_best >= ctx->max_stagnation)
		return (1);
	if (ctx->config.log_level > LOG_SILENT && ctx->progress_int > 0.0
		&& current_time >= ctx->next_progress)
		log_progress(ctx, current_time);
	return (0);
}

static void	cuda_cleanup(t_cuda_solver_ctx *ctx)
{
	cudaFree(ctx->d_coords);
	cudaFree(ctx->d_tau);
	cudaFree(ctx->d_candidate_idx);
	cudaFree(ctx->d_eta_beta);
	cudaFree(ctx->ants.routes);
	cudaFree(ctx->ants.route_lengths);
	cudaFree(ctx->ants.visited_l1);
	cudaFree(ctx->ants.visited_l2);
	cudaFree(ctx->ants.rng_states);
	cudaFree(ctx->ants.ant_summary);
	cudaFree(ctx->d_iter_stats);
	cudaFree(ctx->flat.routes);
	cudaFree(ctx->flat.lengths);
	free(ctx->h_coords);
	free(ctx->h_ant_summary);
	free(ctx->h_routes);
	free(ctx->h_route_lengths);
	free(ctx->h_flat_routes);
	free(ctx->h_flat_lengths);
}

static void	print_iteration_stats(t_cuda_solver_ctx *ctx)
{
	fprintf(stderr, "\n[cuda] Iteration statistics summary (%d iterations):\n", ctx->iter);
	fprintf(stderr, "[cuda]   Customer moves:      %llu\n", ctx->accum_stats.customer_moves);
	fprintf(stderr, "[cuda]   Candidate moves:     %llu\n", ctx->accum_stats.candidate_moves);
	fprintf(stderr, "[cuda]   Fallback calls:      %llu\n", ctx->accum_stats.fallback_calls);
	fprintf(stderr, "[cuda]   Fallback moves:      %llu\n", ctx->accum_stats.fallback_moves);
	fprintf(stderr, "[cuda]   Depot offer calls:   %llu\n", ctx->accum_stats.depot_offer_calls);
	fprintf(stderr, "[cuda]   Depot close moves:   %llu\n", ctx->accum_stats.depot_close_moves);
	fprintf(stderr, "[cuda]   Non-empty routes:    %llu\n", ctx->accum_stats.nonempty_routes);
	fprintf(stderr, "[cuda]   Fallback groups scn: %llu\n", ctx->accum_stats.fallback_word_groups_scanned);
	fprintf(stderr, "[cuda]   Fallback nodes scd:  %llu\n", ctx->accum_stats.fallback_nodes_scored);
}

static t_status	run_solver_loop(t_cuda_solver_ctx *ctx)
{
	t_status			status;
	t_cuda_problem_data	prob;
	int					best_idx;
	float				best_iter_cost;

	prob.coords = ctx->d_coords;
	prob.tau = ctx->d_tau;
	prob.candidate_idx = ctx->d_candidate_idx;
	prob.eta_beta = ctx->d_eta_beta;
	while (!should_stop(ctx))
	{
		status = cuda_iter_step(ctx, prob);
		if (status != SOLVER_OK)
			return (status);
		update_accum_stats(ctx);
		if (select_iter_best_host(ctx->h_ant_summary, ctx->m, &best_idx, &best_iter_cost))
		{
			status = update_global_best(ctx, best_idx, best_iter_cost);
			if (status != SOLVER_OK)
				return (status);
			status = update_pheromones(ctx, best_idx, best_iter_cost);
			if (status != SOLVER_OK)
				return (status);
		}
		else
			ctx->iter_since_best++;
		ctx->iter++;
	}
	return (SOLVER_OK);
}

static void	init_loop_timers(t_cuda_solver_ctx *ctx)
{
	ctx->max_runtime = ctx->config.timeout_seconds;
	if (ctx->max_runtime <= 0.0)
		ctx->max_runtime = 300.0;
	ctx->max_stagnation = ctx->config.stagnation_epochs;
	ctx->min_rel_imp = ctx->config.min_rel_improvement;
	ctx->progress_int = ctx->config.progress_interval_seconds;
	ctx->start_time = wall_time_seconds();
	if (ctx->progress_int > 0.0)
		ctx->next_progress = ctx->start_time + ctx->progress_int;
	else
		ctx->next_progress = 0.0;
}

static t_status	run_solver_orchestrator(t_cuda_solver_ctx *ctx)
{
	t_status	status;

	init_loop_timers(ctx);
	if (ctx->config.log_level > LOG_SILENT)
		fprintf(stderr,
			"CUDA Solver starting... (N=%d, k=%d, M=%d, candidate_k=%d, seed=%u)\n",
			ctx->params->n, ctx->params->k, ctx->m, ctx->cand_k,
			ctx->params->seed);
	status = run_solver_loop(ctx);
	if (ctx->config.log_level > LOG_SILENT)
		print_iteration_stats(ctx);
	return (status);
}

t_status	vrp_solve_cuda(t_solver_params *params,
				t_cuda_coords coords,
				t_solution *best_solution,
				double *best_cost)
{
	t_cuda_solver_ctx	ctx;
	t_status			status;

	if (params->n <= 0 || params->k <= 0 || !coords.x || !coords.y
		|| !best_solution || !best_cost)
		return (SOLVER_ERR_INVALID_INPUT);
	ctx.params = params;
	ctx.coords = coords;
	ctx.best_sol = best_solution;
	ctx.best_cost = best_cost;
	ctx.iter = 0;
	ctx.iter_since_best = 0;
	ctx.global_best = DBL_MAX;
	memset(&ctx.accum_stats, 0, sizeof(t_cuda_iter_stats));
	status = cuda_init_params(&ctx);
	if (status == SOLVER_OK)
		status = cuda_alloc_host(&ctx);
	if (status == SOLVER_OK)
		status = cuda_alloc_device(&ctx);
	if (status == SOLVER_OK)
		status = cuda_init_device(&ctx);
	if (status == SOLVER_OK)
		status = run_solver_orchestrator(&ctx);
	cuda_cleanup(&ctx);
	if (status == SOLVER_OK)
		status = (*best_cost < DBL_MAX) ? SOLVER_OK : SOLVER_ERR_NO_SOLUTION;
	return (status);
}
