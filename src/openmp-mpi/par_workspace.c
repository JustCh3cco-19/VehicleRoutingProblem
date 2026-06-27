#include "aco.h"
#include "openmp-mpi/mpi_internal.h"
#include "matrix.h"
#include "solution.h"
#include <stdlib.h>

void	par_ws_free(aco_mpi_workspace_t *ws)
{
	if (!ws)
		return ;
	free(ws->route_loads);
	free(ws->visited);
	free(ws->meta_active);
	solution_free(ws->thread_best);
	solution_free(ws->sol);
}

int	par_ws_init(aco_mpi_workspace_t *ws, int k, int n, int words,
		int meta_words)
{
	size_t	visited_bytes;
	size_t	meta_bytes;
	size_t	route_load_bytes;

	visited_bytes = 0;
	meta_bytes = 0;
	route_load_bytes = 0;
	if (!matrix_mul_size((size_t)words, sizeof(uint64_t), &visited_bytes) ||
		!matrix_mul_size((size_t)meta_words, sizeof(uint64_t), &meta_bytes) ||
		!matrix_mul_size((size_t)k, sizeof(int), &route_load_bytes))
		return (0);
	ws->sol = solution_create(k, n);
	ws->thread_best = solution_create(k, n);
	ws->visited = par_aligned_calloc(visited_bytes);
	ws->meta_active = par_aligned_calloc(meta_bytes);
	ws->route_loads = calloc(1u, route_load_bytes);
	if (!ws->sol || !ws->thread_best || !ws->visited || !ws->meta_active ||
		!ws->route_loads)
	{
		par_ws_free(ws);
		return (0);
	}
	return (1);
}
