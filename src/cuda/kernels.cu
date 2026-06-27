#include "cuda/cuda_kernels.h"
#include <math.h>
#include <stdio.h>

__device__ static inline float	cuda_rand01(unsigned int *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return ((float)(*state) / 4294967295.0f);
}

__device__ static inline float	dequantize_tau(uint8_t q,
									t_cuda_params params)
{
	return (expf((float)q * params.log_tau_step + params.log_tau_min));
}

__device__ static inline uint8_t	quantize_tau(float tau,
										t_cuda_params params)
{
	float	log_t;
	float	q_f;

	if (tau <= 0.0f)
		return (params.q_tau_min);
	log_t = logf(tau);
	q_f = (log_t - params.log_tau_min) / params.log_tau_step;
	if (q_f <= (float)params.q_tau_min)
		return (params.q_tau_min);
	if (q_f >= (float)params.q_tau_max)
		return (params.q_tau_max);
	return ((uint8_t)roundf(q_f));
}

__device__ static inline float	calc_dist(float2 a, float2 b)
{
	float	dx;
	float	dy;

	dx = a.x - b.x;
	dy = a.y - b.y;
	return (sqrtf(dx * dx + dy * dy));
}

__device__ static inline float	cuda_warp_sum(float value)
{
	int	offset;

	offset = CUDA_WARP_SIZE / 2;
	for (; offset > 0; offset >>= 1)
		value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
	return (value);
}

__device__ static inline float	cuda_warp_prefix_sum(float value, int lane)
{
	int		offset;
	float	other;

	offset = 1;
	for (; offset < CUDA_WARP_SIZE; offset <<= 1)
	{
		other = __shfl_up_sync(0xFFFFFFFFu, value, offset);
		if (lane >= offset)
			value += other;
	}
	return (value);
}

__device__ static inline uint64_t	cuda_relevant_mask_for_word(int word_idx,
										int n)
{
	int			start_node;
	int			end_node;
	uint64_t	mask;
	int			valid_bits;

	start_node = word_idx << 6;
	end_node = start_node + 63;
	mask = 0xFFFFFFFFFFFFFFFFull;
	if (start_node == 0)
		mask &= ~1ull;
	if (end_node > n)
	{
		valid_bits = n - start_node + 1;
		if (valid_bits <= 0)
			return (0ull);
		if (valid_bits < 64)
			mask &= ((1ull << valid_bits) - 1ull);
	}
	return (mask);
}

__global__ void	kern_init_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t q_tau0)
{
	uint64_t	idx;

	idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
		d_tau[idx] = q_tau0;
}

__global__ void	kern_evaporate_tau(uint8_t *d_tau, uint64_t total_elements,
					uint8_t delta, uint8_t q_min)
{
	uint64_t	idx;
	uint8_t		q;

	idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total_elements)
	{
		q = d_tau[idx];
		if (q > q_min + delta)
			d_tau[idx] = q - delta;
		else
			d_tau[idx] = q_min;
	}
}

__device__ static void	build_candidates_for_node(int i,
							const float2 *d_coords,
							int *d_candidate_idx,
							float *d_eta_beta,
							t_cuda_params params)
{
	int		best_nodes[CUDA_MAX_CANDIDATES];
	float	best_costs[CUDA_MAX_CANDIDATES];
	float2	p_i;
	int		j;
	int		pos;
	int		t;
	float	d;

	t = 0;
	for (; t < params.cand_k; ++t)
	{
		best_nodes[t] = 0;
		best_costs[t] = INFINITY;
	}
	p_i = d_coords[i];
	j = 1;
	for (; j <= params.n; ++j)
	{
		if (i == j)
			continue ;
		d = calc_dist(p_i, d_coords[j]);
		pos = -1;
		t = 0;
		for (; t < params.cand_k; ++t)
		{
			if (d < best_costs[t])
			{
				pos = t;
				break ;
			}
		}
		if (pos >= 0)
		{
			t = params.cand_k - 1;
			for (; t > pos; --t)
			{
				best_costs[t] = best_costs[t - 1];
				best_nodes[t] = best_nodes[t - 1];
			}
			best_costs[pos] = d;
			best_nodes[pos] = j;
		}
	}
	t = 0;
	for (; t < params.cand_k; ++t)
	{
		j = i * params.cand_k + t;
		d_candidate_idx[j] = best_nodes[t];
		if (best_nodes[t] > 0)
			d_eta_beta[j] = powf(1.0f / (best_costs[t] + CUDA_EPS),
					params.beta);
		else
			d_eta_beta[j] = 0.0f;
	}
}

__global__ void	kern_build_candidates(const float2 *d_coords,
					int *d_candidate_idx, float *d_eta_beta,
					t_cuda_params params)
{
	int	i;

	i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i <= params.n)
		build_candidates_for_node(i, d_coords, d_candidate_idx,
			d_eta_beta, params);
}

__device__ static void	reset_ant_visited(t_cuda_ant_state ants,
							t_cuda_params params, int ant)
{
	uint64_t	*l1_row;
	uint64_t	*l2_row;
	int			w;

	l1_row = ants.visited_l1 + ant * (params.visited_row_stride / 8);
	l2_row = ants.visited_l2 + ant * params.visited_l2_words;
	w = 0;
	for (; w < params.visited_l1_words; ++w)
		l1_row[w] = ~cuda_relevant_mask_for_word(w, params.n);
	w = 0;
	for (; w < params.visited_l2_words; ++w)
		l2_row[w] = 0ull;
}

__global__ void	kern_reset_ants(t_cuda_ant_state ants,
					t_cuda_params params, unsigned int seed)
{
	int	ant;
	int	v;

	ant = blockIdx.x * blockDim.x + threadIdx.x;
	if (ant >= params.m)
		return ;
	reset_ant_visited(ants, params, ant);
	v = 0;
	for (; v < params.k; ++v)
	{
		ants.routes[0 * params.m + ant] = 0;
		ants.route_lengths[ant * params.k + v] = 1;
	}
	ants.rng_states[ant] = seed ^ (1664525u * (unsigned int)(ant + 1));
	ants.ant_summary[ant].feasible = 0;
	ants.ant_summary[ant].unvisited_count = params.n;
	ants.ant_summary[ant].cost = 0.0f;
}

__device__ static float	get_candidate_score(int lane,
							t_cuda_select_ctx *ctx,
							t_cuda_problem_data prob,
							t_cuda_params params)
{
	int			base;
	int			candidate;
	int			word_idx;
	uint64_t	tau_idx;
	float		tau;

	base = ctx->current * params.cand_k;
	candidate = prob.candidate_idx[base + lane];
	if (candidate <= 0)
		return (0.0f);
	word_idx = candidate >> 6;
	if ((ctx->visited_l1[word_idx] & (1ull << (candidate & 63))) != 0)
		return (0.0f);
	tau_idx = (uint64_t)ctx->current * (params.n + 1) + candidate;
	tau = dequantize_tau(prob.tau[tau_idx], params);
	return (powf(tau, params.alpha) * prob.eta_beta[base + lane]);
}

__device__ static int	warp_select_by_threshold(float score,
							float total, int candidate,
							unsigned int *rng_state)
{
	int				lane;
	float			threshold;
	float			prefix;
	unsigned int	mask_sel;

	lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
	threshold = 0.0f;
	if (lane == 0)
		threshold = cuda_rand01(rng_state) * total;
	threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);
	prefix = cuda_warp_prefix_sum(score, lane);
	mask_sel = __ballot_sync(0xFFFFFFFFu, score > 0.0f
			&& prefix >= threshold && (prefix - score) < threshold);
	if (mask_sel != 0u)
		return (__shfl_sync(0xFFFFFFFFu, candidate,
				__ffs((int)mask_sel) - 1));
	return (-1);
}

__device__ static int	select_candidate(t_cuda_select_ctx *ctx,
							t_cuda_problem_data prob,
							t_cuda_params params)
{
	int		lane;
	int		candidate;
	float	score;
	float	total;

	lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
	candidate = 0;
	score = 0.0f;
	if (lane < params.cand_k)
	{
		candidate = prob.candidate_idx[ctx->current * params.cand_k + lane];
		score = get_candidate_score(lane, ctx, prob, params);
	}
	total = cuda_warp_sum(score);
	total = __shfl_sync(0xFFFFFFFFu, total, 0);
	if (ctx->total_score)
		*(ctx->total_score) = total;
	if (total <= 0.0f)
		return (-1);
	return (warp_select_by_threshold(score, total, candidate, ctx->rng_state));
}

__device__ static uint64_t	get_chunk_mask(int word_idx,
								t_cuda_select_ctx *ctx,
								t_cuda_fallback_state *state)
{
	if (word_idx < state->params.visited_l1_words
		&& (ctx->visited_l2[word_idx >> 6]
			& (1ull << (word_idx & 63))) == 0)
	{
		state->groups_scanned++;
		return ((~ctx->visited_l1[word_idx])
			& cuda_relevant_mask_for_word(word_idx, state->params.n));
	}
	return (0ull);
}

__device__ static void	scan_word_chunk(int word_idx,
							t_cuda_select_ctx *ctx,
							t_cuda_problem_data prob,
							t_cuda_fallback_state *state)
{
	uint64_t	free_mask;
	int			node;
	float		dx;
	float		dy;

	free_mask = get_chunk_mask(word_idx, ctx, state);
	if (__ballot_sync(0xFFFFFFFFu, free_mask != 0ull) == 0u
		|| free_mask == 0ull)
		return ;
	while (free_mask != 0ull)
	{
		node = (word_idx << 6) + (__ffsll((long long)free_mask) - 1);
		state->nodes_scored++;
		dx = state->p_curr.x - prob.coords[node].x;
		dy = state->p_curr.y - prob.coords[node].y;
		dx = powf(dequantize_tau(prob.tau[(uint64_t)ctx->current
					* (state->params.n + 1) + node], state->params),
				state->params.alpha) * powf(rsqrtf(dx * dx + dy * dy
					+ CUDA_EPS), state->params.beta);
		state->local_sum += dx;
		if (state->local_sum > 0.0f && cuda_rand01(ctx->rng_state)
			<= (dx / state->local_sum))
			state->local_selected = node;
		free_mask &= (free_mask - 1ull);
	}
}

__device__ static int	update_fallback_stats(int lane,
							t_cuda_fallback_state *state,
							t_cuda_iter_stats *d_iter_stats)
{
	int	offset;
	int	tot_g;
	int	tot_n;

	tot_g = state->groups_scanned;
	tot_n = state->nodes_scored;
	offset = 16;
	for (; offset > 0; offset /= 2)
	{
		tot_g += __shfl_down_sync(0xFFFFFFFFu, tot_g, offset);
		tot_n += __shfl_down_sync(0xFFFFFFFFu, tot_n, offset);
	}
	if (lane == 0 && d_iter_stats)
	{
		atomicAdd(&(d_iter_stats->fallback_word_groups_scanned),
			(unsigned long long)tot_g);
		atomicAdd(&(d_iter_stats->fallback_nodes_scored),
			(unsigned long long)tot_n);
	}
	return (__shfl_sync(0xFFFFFFFFu, state->local_selected, 0));
}

__device__ static int	reduce_fallback_result(int lane,
							t_cuda_fallback_state *state,
							t_cuda_iter_stats *d_iter_stats,
							t_cuda_select_ctx *ctx)
{
	int		offset;
	float	other_sum;
	int		other_node;
	float	combined_sum;

	offset = 1;
	for (; offset < CUDA_WARP_SIZE; offset *= 2)
	{
		other_sum = __shfl_down_sync(0xFFFFFFFFu, state->local_sum, offset);
		other_node = __shfl_down_sync(0xFFFFFFFFu, state->local_selected,
				offset);
		combined_sum = state->local_sum + other_sum;
		if (combined_sum > 0.0f && cuda_rand01(ctx->rng_state)
			<= (other_sum / combined_sum))
			state->local_selected = other_node;
		state->local_sum = combined_sum;
	}
	return (update_fallback_stats(lane, state, d_iter_stats));
}

__device__ static int	select_fallback(t_cuda_select_ctx *ctx,
							t_cuda_problem_data prob,
							t_cuda_iter_stats *d_iter_stats,
							t_cuda_params params)
{
	int						lane;
	int						word_chunks;
	int						chunk;
	t_cuda_fallback_state	state;

	lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
	if (lane == 0 && d_iter_stats)
		atomicAdd(&(d_iter_stats->fallback_calls), 1ULL);
	state.p_curr = prob.coords[ctx->current];
	state.local_sum = 0.0f;
	state.local_selected = -1;
	state.groups_scanned = 0;
	state.nodes_scored = 0;
	state.params = params;
	word_chunks = (params.visited_l1_words + CUDA_WARP_SIZE - 1)
		/ CUDA_WARP_SIZE;
	chunk = 0;
	for (; chunk < word_chunks; ++chunk)
		scan_word_chunk(chunk * CUDA_WARP_SIZE + lane, ctx, prob, &state);
	if (ctx->total_score)
		*(ctx->total_score) = __shfl_sync(0xFFFFFFFFu, state.local_sum, 0);
	return (reduce_fallback_result(lane, &state, d_iter_stats, ctx));
}

__device__ static int	make_move_decision(int move,
							float depot_score,
							t_cuda_construct_ctx *ctx)
{
	float	threshold;

	threshold = 0.0f;
	if (move > 0 && depot_score > 0.0f)
	{
		if ((threadIdx.x & (CUDA_WARP_SIZE - 1)) == 0)
			threshold = cuda_rand01(&ctx->rng_state)
				* (ctx->score_cand + depot_score);
		threshold = __shfl_sync(0xFFFFFFFFu, threshold, 0);
		if (threshold >= ctx->score_cand)
			return (0);
	}
	else if (move <= 0 && depot_score <= 0.0f)
		return (0);
	return (1);
}

__device__ static int	evaluate_move(int move,
							t_cuda_problem_data prob,
							t_cuda_construct_ctx *ctx)
{
	int		close_allowed;
	float	d;
	float	eta;
	float	depot_score;

	close_allowed = (ctx->remaining <= (ctx->params.k - ctx->vehicle - 1)
			* ctx->params.cap);
	if (close_allowed && (threadIdx.x & (CUDA_WARP_SIZE - 1)) == 0
		&& ctx->iter_stats)
		atomicAdd(&(ctx->iter_stats->depot_offer_calls), 1ULL);
	depot_score = 0.0f;
	if (close_allowed)
	{
		d = calc_dist(prob.coords[__shfl_sync(0xFFFFFFFFu, ctx->current, 0)],
				prob.coords[0]);
		eta = 1.0f / (d + CUDA_EPS);
		depot_score = ctx->params.depot_close_weight
			* powf(eta, ctx->params.beta);
	}
	return (make_move_decision(move, depot_score, ctx));
}

__device__ static void	update_visited_and_stats(int move,
							t_cuda_ant_state ants,
							t_cuda_construct_ctx *ctx)
{
	int			word_idx;
	uint64_t	*l1_row;

	if (ctx->iter_stats)
	{
		atomicAdd(&(ctx->iter_stats->customer_moves), 1ULL);
		if (ctx->chosen_fallback)
			atomicAdd(&(ctx->iter_stats->fallback_moves), 1ULL);
		else
			atomicAdd(&(ctx->iter_stats->candidate_moves), 1ULL);
	}
	l1_row = ants.visited_l1 + ctx->ant * (ctx->params.visited_row_stride / 8);
	word_idx = move >> 6;
	l1_row[word_idx] |= (1ull << (move & 63));
	if (l1_row[word_idx] ==
		~cuda_relevant_mask_for_word(word_idx, ctx->params.n))
		ants.visited_l2[ctx->ant * ctx->params.visited_l2_words
			+ (word_idx >> 6)] |= (1ull << (word_idx & 63));
}

__device__ static void	execute_move(int move,
							t_cuda_problem_data prob,
							t_cuda_ant_state ants,
							t_cuda_construct_ctx *ctx)
{
	if ((threadIdx.x & (CUDA_WARP_SIZE - 1)) == 0)
	{
		ctx->global_step++;
		ants.routes[ctx->global_step * ctx->params.m + ctx->ant] = move;
		ctx->route_len++;
		ctx->route_load++;
		update_visited_and_stats(move, ants, ctx);
		ctx->total_cost += calc_dist(prob.coords[ctx->current],
				prob.coords[move]);
		ctx->current = move;
		ctx->remaining--;
	}
	__syncwarp();
	ctx->global_step = __shfl_sync(0xFFFFFFFFu, ctx->global_step, 0);
	ctx->route_len = __shfl_sync(0xFFFFFFFFu, ctx->route_len, 0);
	ctx->route_load = __shfl_sync(0xFFFFFFFFu, ctx->route_load, 0);
	ctx->current = __shfl_sync(0xFFFFFFFFu, ctx->current, 0);
	ctx->remaining = __shfl_sync(0xFFFFFFFFu, ctx->remaining, 0);
}

__device__ static void	close_route(t_cuda_problem_data prob,
							t_cuda_ant_state ants,
							t_cuda_construct_ctx *ctx)
{
	if ((threadIdx.x & (CUDA_WARP_SIZE - 1)) == 0)
	{
		ctx->global_step++;
		ants.routes[ctx->global_step * ctx->params.m + ctx->ant] = 0;
		ctx->total_cost += calc_dist(prob.coords[ctx->current],
				prob.coords[0]);
		ants.route_lengths[ctx->ant * ctx->params.k + ctx->vehicle]
			= ctx->route_len + 1;
		if (ctx->iter_stats)
		{
			atomicAdd(&(ctx->iter_stats->depot_close_moves), 1ULL);
			if (ctx->route_len > 1)
				atomicAdd(&(ctx->iter_stats->nonempty_routes), 1ULL);
		}
	}
	__syncwarp();
	ctx->global_step = __shfl_sync(0xFFFFFFFFu, ctx->global_step, 0);
}

__device__ static int	try_route_step(t_cuda_problem_data prob,
							t_cuda_ant_state ants,
							t_cuda_construct_ctx *ctx)
{
	int					move;
	t_cuda_select_ctx	sel;

	sel.current = __shfl_sync(0xFFFFFFFFu, ctx->current, 0);
	sel.visited_l1 = ants.visited_l1
		+ ctx->ant * (ctx->params.visited_row_stride / 8);
	sel.visited_l2 = ants.visited_l2 + ctx->ant * ctx->params.visited_l2_words;
	sel.rng_state = &ctx->rng_state;
	sel.total_score = &ctx->score_cand;
	move = select_candidate(&sel, prob, ctx->params);
	ctx->chosen_fallback = 0;
	if (move <= 0)
	{
		move = select_fallback(&sel, prob, ctx->iter_stats, ctx->params);
		ctx->chosen_fallback = 1;
	}
	if (!evaluate_move(move, prob, ctx))
		return (0);
	execute_move(move, prob, ants, ctx);
	return (1);
}


__device__ static void	build_vehicle_route(t_cuda_problem_data prob,
							t_cuda_ant_state ants,
							t_cuda_construct_ctx *ctx)
{
	ctx->current = 0;
	ctx->route_load = 0;
	ctx->route_len = 1;
	if ((threadIdx.x & (CUDA_WARP_SIZE - 1)) != 0)
	{
		ctx->current = 0;
		ctx->route_load = 0;
	}
	while (1)
	{
		if (__shfl_sync(0xFFFFFFFFu, ctx->remaining, 0) <= 0
			|| __shfl_sync(0xFFFFFFFFu, ctx->route_load, 0) >= ctx->params.cap)
			break ;
		if (!try_route_step(prob, ants, ctx))
			break ;
	}
	close_route(prob, ants, ctx);
}

__global__ void	kern_construct_solutions(t_cuda_problem_data prob,
					t_cuda_ant_state ants,
					t_cuda_iter_stats *d_iter_stats,
					t_cuda_params params)
{
	int						lane;
	int						ant;
	t_cuda_construct_ctx	ctx;

	lane = threadIdx.x & (CUDA_WARP_SIZE - 1);
	ant = (blockIdx.x * blockDim.x + threadIdx.x) / CUDA_WARP_SIZE;
	if (ant >= params.m)
		return ;
	ctx.ant = ant;
	ctx.iter_stats = d_iter_stats;
	ctx.params = params;
	if (lane == 0)
	{
		ctx.rng_state = ants.rng_states[ant];
		ctx.remaining = params.n;
		ctx.total_cost = 0.0f;
		ctx.global_step = 0;
	}
	ctx.vehicle = 0;
	for (; ctx.vehicle < params.k; ++ctx.vehicle)
		build_vehicle_route(prob, ants, &ctx);
	if (lane == 0)
	{
		ants.rng_states[ant] = ctx.rng_state;
		ants.ant_summary[ant].feasible = (ctx.remaining == 0) ? 1 : 0;
		ants.ant_summary[ant].unvisited_count = ctx.remaining;
		ants.ant_summary[ant].cost = ctx.total_cost;
	}
}

__device__ void	atomic_max_u8(uint8_t *address, uint8_t val)
{
	unsigned int	*base_address;
	unsigned int	shift;
	unsigned int	mask;
	unsigned int	assumed;
	unsigned int	old;

	base_address = (unsigned int *)((size_t)address & ~3);
	shift = ((size_t)address & 3) * 8;
	mask = 0xFF << shift;
	old = *base_address;
	do {
		assumed = old;
		if (((assumed >> shift) & 0xFF) >= val)
			break ;
		old = atomicCAS(base_address, assumed,
				(assumed & ~mask) | (val << shift));
	} while (assumed != old);
}

__device__ static void	deposit_step(uint8_t *d_tau,
							uint64_t tau_idx,
							float amount,
							t_cuda_params params)
{
	float	old_tau;
	uint8_t	new_q;

	old_tau = dequantize_tau(d_tau[tau_idx], params);
	new_q = quantize_tau(old_tau + amount, params);
	atomic_max_u8(&d_tau[tau_idx], new_q);
}

__global__ void	kern_deposit_best(uint8_t *d_tau,
					t_cuda_ant_state ants,
					t_cuda_deposit_params dep,
					t_cuda_params params)
{
	int	veh_idx;
	int	route_len;
	int	prev;
	int	step;
	int	pos;

	veh_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (veh_idx >= params.k)
		return ;
	route_len = ants.route_lengths[dep.best_ant * params.k + veh_idx];
	prev = 0;
	step = 0;
	pos = 0;
	for (; pos < veh_idx; ++pos)
		step += ants.route_lengths[dep.best_ant * params.k + pos] - 1;
	pos = 1;
	for (; pos < route_len; ++pos)
	{
		deposit_step(d_tau, (uint64_t)prev * (params.n + 1)
			+ ants.routes[(step + pos) * params.m + dep.best_ant],
			dep.amount, params);
		deposit_step(d_tau, (uint64_t)ants.routes[(step + pos) * params.m
				+ dep.best_ant] * (params.n + 1) + prev,
			dep.amount, params);
		prev = ants.routes[(step + pos) * params.m + dep.best_ant];
	}
}

__global__ void	kern_deposit_flat(uint8_t *d_tau,
					t_cuda_flat_routes flat,
					float deposit_amount,
					t_cuda_params params)
{
	int	veh_idx;
	int	route_len;
	int	prev;
	int	pos;
	int	offset;

	veh_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (veh_idx >= params.k)
		return ;
	route_len = flat.lengths[veh_idx];
	prev = 0;
	offset = veh_idx * (params.n + 1);
	pos = 1;
	for (; pos < route_len; ++pos)
	{
		deposit_step(d_tau, (uint64_t)prev * (params.n + 1)
			+ flat.routes[offset + pos], deposit_amount, params);
		deposit_step(d_tau, (uint64_t)flat.routes[offset + pos]
			* (params.n + 1) + prev, deposit_amount, params);
		prev = flat.routes[offset + pos];
	}
}

void	launch_init_tau(uint8_t *d_tau, int n, uint8_t q_tau0)
{
	uint64_t	total;
	int			blocks;

	total = (uint64_t)(n + 1) * (n + 1);
	blocks = (total + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;
	kern_init_tau<<<blocks, CUDA_THREADS_PER_BLOCK>>>(d_tau, total, q_tau0);
}
