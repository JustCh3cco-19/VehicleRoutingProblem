# include "solver.h"
#include "seq/internal.h"
#include "solution.h"
#include <stdlib.h>
#include <string.h>

int	seq_workspace_init(t_seq_workspace *ws, int k, int n,
		int visited_words)
{
	memset(ws, 0, sizeof(*ws));
	ws->sol = solution_create(k, n);
	ws->visited = seq_aligned_calloc((size_t)visited_words
			* sizeof(uint64_t));
	ws->route_loads = calloc((size_t)k, sizeof(int));
	ws->rng_state = 1u;
	if (!ws->sol || !ws->visited || !ws->route_loads)
	{
		free(ws->route_loads);
		free(ws->visited);
		solution_free(ws->sol);
		memset(ws, 0, sizeof(*ws));
		return (0);
	}
	return (1);
}

void	seq_workspace_free(t_seq_workspace *ws)
{
	if (!ws)
		return ;
	free(ws->route_loads);
	free(ws->visited);
	solution_free(ws->sol);
	memset(ws, 0, sizeof(*ws));
}
