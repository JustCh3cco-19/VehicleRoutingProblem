#include "openmp-mpi/mpi_internal.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef USE_MPI

void	par_async_init(t_par_async_context *ctx, int mpi_size)
{
	int	blocklengths[2];
	MPI_Aint	offsets[2];
	MPI_Datatype	types[2];

	ctx->req = MPI_REQUEST_NULL;
	ctx->recv_buf = NULL;
	ctx->counts = calloc((size_t)mpi_size, sizeof(int));
	ctx->displs = calloc((size_t)mpi_size, sizeof(int));
	blocklengths[0] = 1;
	blocklengths[1] = 1;
	offsets[0] = offsetof(t_par_sparse_delta, edge_idx);
	offsets[1] = offsetof(t_par_sparse_delta, increment);
	types[0] = MPI_UINT32_T;
	types[1] = MPI_FLOAT;
	MPI_Type_create_struct(2, blocklengths, offsets, types, &ctx->type);
	MPI_Type_commit(&ctx->type);
	ctx->active = 0;
}

void	par_async_cleanup(t_par_async_context *ctx)
{
	if (!ctx)
		return ;
	if (ctx->active)
		MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
	free(ctx->counts);
	free(ctx->displs);
	free(ctx->recv_buf);
	MPI_Type_free(&ctx->type);
	ctx->active = 0;
}

void	par_async_wait_and_apply(t_par_async_context *ctx,
		t_par_matrix *tau, int mpi_rank, int mpi_size)
{
	float	inv;
	int		r;
	int		i;

	if (!ctx->active)
		return ;
	MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
	inv = 1.0f / (float)mpi_size;
	r = 0;
	while (r < mpi_size)
	{
		if (r != mpi_rank)
		{
			i = ctx->displs[r];
			while (i < ctx->displs[r] + ctx->counts[r])
			{
				tau->data[ctx->recv_buf[i].edge_idx] +=
					ctx->recv_buf[i].increment * inv;
				i++;
			}
		}
		r++;
	}
	free(ctx->recv_buf);
	ctx->recv_buf = NULL;
	ctx->active = 0;
}

void	par_async_start(t_par_async_context *ctx,
		t_par_sparse_delta *local_deltas, int local_count, int mpi_size)
{
	int	my_c;
	int	total_recv;
	int	i;

	my_c = local_count;
	MPI_Allgather(&my_c, 1, MPI_INT, ctx->counts, 1, MPI_INT, MPI_COMM_WORLD);
	total_recv = 0;
	i = 0;
	while (i < mpi_size)
	{
		ctx->displs[i] = total_recv;
		total_recv += ctx->counts[i];
		i++;
	}
	if (total_recv > 0)
	{
		ctx->recv_buf = malloc((size_t)total_recv * sizeof(t_par_sparse_delta));
		MPI_Iallgatherv(local_deltas, local_count, ctx->type, ctx->recv_buf,
			ctx->counts, ctx->displs, ctx->type, MPI_COMM_WORLD,
			&ctx->req);
		ctx->active = 1;
	}
	else
	{
		ctx->active = 0;
		ctx->req = MPI_REQUEST_NULL;
	}
}

#endif
