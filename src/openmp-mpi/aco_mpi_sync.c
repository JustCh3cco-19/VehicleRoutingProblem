#include "aco_mpi_internal.h"

#ifdef USE_MPI
#include <stddef.h>
#include <stdlib.h>

void aco_mpi_async_sparse_init(aco_mpi_async_sparse_context_t *ctx,
                               int mpi_size) {
  ctx->req = MPI_REQUEST_NULL;
  ctx->recv_buf = NULL;
  ctx->counts = calloc((size_t)mpi_size, sizeof(int));
  ctx->displs = calloc((size_t)mpi_size, sizeof(int));
  int blocklengths[2] = {
      [0] = 1,
      [1] = 1,
  };
  MPI_Aint offsets[2];
  offsets[0] = offsetof(aco_mpi_sparse_delta_t, edge_idx);
  offsets[1] = offsetof(aco_mpi_sparse_delta_t, increment);
  MPI_Datatype types[2] = {
      [0] = MPI_UINT32_T,
      [1] = MPI_FLOAT,
  };
  MPI_Type_create_struct(2, blocklengths, offsets, types, &ctx->type);
  MPI_Type_commit(&ctx->type);
  ctx->active = 0;
}

void aco_mpi_async_sparse_cleanup(aco_mpi_async_sparse_context_t *ctx) {
  if (!ctx) {
    return;
  }
  if (ctx->active) {
    MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
  }
  free(ctx->counts);
  free(ctx->displs);
  free(ctx->recv_buf);
  MPI_Type_free(&ctx->type);
  ctx->active = 0;
}

void aco_mpi_async_sparse_wait_and_apply(aco_mpi_async_sparse_context_t *ctx,
                                         aco_mpi_matrix_float_t *tau,
                                         int mpi_rank, int mpi_size) {
  if (!ctx->active) {
    return;
  }
  MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
  float inv = 1.0f / (float)mpi_size;
  for (int r = 0; r < mpi_size; r++) {
    if (r == mpi_rank) {
      continue;
    }
    for (int i = ctx->displs[r]; i < ctx->displs[r] + ctx->counts[r]; i++) {
      tau->data[ctx->recv_buf[i].edge_idx] += ctx->recv_buf[i].increment * inv;
    }
  }
  free(ctx->recv_buf);
  ctx->recv_buf = NULL;
  ctx->active = 0;
}

void aco_mpi_async_sparse_start(aco_mpi_async_sparse_context_t *ctx,
                                aco_mpi_sparse_delta_t *local_deltas,
                                int local_count, int mpi_size) {
  int my_c = local_count;
  MPI_Allgather(&my_c, 1, MPI_INT, ctx->counts, 1, MPI_INT, MPI_COMM_WORLD);
  int total_recv = 0;
  for (int i = 0; i < mpi_size; i++) {
    ctx->displs[i] = total_recv;
    total_recv += ctx->counts[i];
  }
  if (total_recv > 0) {
    ctx->recv_buf = malloc((size_t)total_recv * sizeof(aco_mpi_sparse_delta_t));
    MPI_Iallgatherv(local_deltas, local_count, ctx->type, ctx->recv_buf,
                    ctx->counts, ctx->displs, ctx->type, MPI_COMM_WORLD,
                    &ctx->req);
    ctx->active = 1;
  } else {
    ctx->active = 0;
    ctx->req = MPI_REQUEST_NULL;
  }
}
#endif
