extern "C" {
#include "config.h"
#include "solver.h"
}

#include "cuda/cuda_internal.h"
#include <math.h>
#include <stdlib.h>

static void	cuda_init_tau_constants(t_cuda_solver_ctx *ctx)
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

static void	cuda_init_params_basic(t_cuda_solver_ctx *ctx)
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

t_status	cuda_init_params(t_cuda_solver_ctx *ctx)
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
	cuda_init_params_basic(ctx);
	cuda_init_tau_constants(ctx);
	ctx->cuda_params.visited_l1_words = (ctx->params->n + 64) / 64;
	ctx->cuda_params.visited_l2_words =
		(ctx->cuda_params.visited_l1_words + 63) / 64;
	ctx->cuda_params.visited_row_stride =
		((ctx->cuda_params.visited_l1_words * 8 + 127) / 128) * 128;
	ctx->cuda_params.depot_close_weight = 2.0f;
	return (SOLVER_OK);
}

t_status	cuda_alloc_host(t_cuda_solver_ctx *ctx)
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
	ctx->total_elements = (size_t)(ctx->params->n + 1)
		* (size_t)(ctx->params->n + 1);
	ctx->tau_elements = (ctx->total_elements + 3) & ~3ull;
	ctx->h_ant_summary = (t_cuda_ant_summary *)malloc(ctx->m
			* sizeof(t_cuda_ant_summary));
	ctx->max_steps = ctx->cuda_params.route_max_len + 1;
	ctx->h_routes = (int *)malloc(ctx->max_steps * ctx->m * sizeof(int));
	ctx->h_route_lengths = (int *)malloc(ctx->m * ctx->params->k * sizeof(int));
	ctx->h_flat_routes = (int *)malloc(ctx->params->k * (ctx->params->n + 1)
			* sizeof(int));
	ctx->h_flat_lengths = (int *)malloc(ctx->params->k * sizeof(int));
	if (!ctx->h_ant_summary || !ctx->h_routes || !ctx->h_route_lengths
		|| !ctx->h_flat_routes || !ctx->h_flat_lengths)
		return (SOLVER_ERR_ALLOCATION);
	return (SOLVER_OK);
}
