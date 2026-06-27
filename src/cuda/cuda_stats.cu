#include "cuda/cuda_internal.h"
#include <stdio.h>

void	cuda_update_accum_stats(t_cuda_solver_ctx *ctx)
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

static void	cuda_log_progress(t_cuda_solver_ctx *ctx, double progress_time)
{
	double	elapsed;
	double	remaining;

	elapsed = progress_time - ctx->start_time;
	if (ctx->max_runtime > 0.0)
	{
		remaining = ctx->max_runtime - elapsed;
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

int	cuda_should_stop(t_cuda_solver_ctx *ctx)
{
	double	current_time;

	current_time = cuda_wall_time_seconds();
	if (ctx->max_runtime > 0.0
		&& (current_time - ctx->start_time) > ctx->max_runtime)
		return (1);
	if (ctx->max_stagnation > 0
		&& ctx->iter_since_best >= ctx->max_stagnation)
		return (1);
	if (ctx->config.log_level > LOG_SILENT && ctx->progress_int > 0.0
		&& current_time >= ctx->next_progress)
		cuda_log_progress(ctx, current_time);
	return (0);
}

void	cuda_print_iteration_stats(t_cuda_solver_ctx *ctx)
{
	fprintf(stderr,
		"\n[cuda] Iteration statistics summary (%d iterations):\n",
		ctx->iter);
	fprintf(stderr, "[cuda]   Customer moves:      %llu\n",
		ctx->accum_stats.customer_moves);
	fprintf(stderr, "[cuda]   Candidate moves:     %llu\n",
		ctx->accum_stats.candidate_moves);
	fprintf(stderr, "[cuda]   Fallback calls:      %llu\n",
		ctx->accum_stats.fallback_calls);
	fprintf(stderr, "[cuda]   Fallback moves:      %llu\n",
		ctx->accum_stats.fallback_moves);
	fprintf(stderr, "[cuda]   Depot offer calls:   %llu\n",
		ctx->accum_stats.depot_offer_calls);
	fprintf(stderr, "[cuda]   Depot close moves:   %llu\n",
		ctx->accum_stats.depot_close_moves);
	fprintf(stderr, "[cuda]   Non-empty routes:    %llu\n",
		ctx->accum_stats.nonempty_routes);
	fprintf(stderr, "[cuda]   Fallback groups scn: %llu\n",
		ctx->accum_stats.fallback_word_groups_scanned);
	fprintf(stderr, "[cuda]   Fallback nodes scd:  %llu\n",
		ctx->accum_stats.fallback_nodes_scored);
}
