#ifndef CUDA_INTERNAL_H
# define CUDA_INTERNAL_H

#include "cuda_context.h"
#include "cuda_kernels.h"

# define CHECK_CUDA(call) if (!cuda_check((call), &status)) goto cleanup
# define CHECK_CUDA_KERNEL() CHECK_CUDA(cudaGetLastError())

int			cuda_check(cudaError_t err, t_status *status);
double		cuda_wall_time_seconds(void);
int			cuda_is_significant_improvement(double prev_best,
				double new_best, double min_rel_improvement);
int			cuda_select_iter_best_host(const t_cuda_ant_summary *summary,
				int m, int *best_idx, float *best_cost);
t_status	cuda_init_params(t_cuda_solver_ctx *ctx);
t_status	cuda_alloc_host(t_cuda_solver_ctx *ctx);
t_status	cuda_alloc_device(t_cuda_solver_ctx *ctx);
t_status	cuda_init_device(t_cuda_solver_ctx *ctx);
t_status	cuda_iter_step(t_cuda_solver_ctx *ctx,
				t_cuda_problem_data prob);
void		cuda_update_accum_stats(t_cuda_solver_ctx *ctx);
t_status	cuda_copy_best_data(t_cuda_solver_ctx *ctx, int best_idx);
t_status	cuda_update_global_best(t_cuda_solver_ctx *ctx, int best_idx,
				float best_iter_cost);
t_status	cuda_update_pheromones(t_cuda_solver_ctx *ctx, int best_idx,
				float best_iter_cost);
int			cuda_should_stop(t_cuda_solver_ctx *ctx);
void		cuda_cleanup(t_cuda_solver_ctx *ctx);
void		cuda_print_iteration_stats(t_cuda_solver_ctx *ctx);

#endif
