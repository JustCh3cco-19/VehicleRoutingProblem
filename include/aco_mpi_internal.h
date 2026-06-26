#ifndef ACO_MPI_INTERNAL_H
#define ACO_MPI_INTERNAL_H

#include "aco.h"

#include <stddef.h>
#include <stdint.h>

#ifdef USE_MPI
#include <mpi.h>
#endif

enum {
  kAcoMpiAlignment = 64,
  kAcoMpiMaxCandidates = 512,
};

typedef struct {
  uint32_t edge_idx;
  float increment;
} aco_mpi_sparse_delta_t;

typedef struct {
  int n;
  int stride;
  float *data;
  float **rows;
} aco_mpi_matrix_float_t;

typedef struct {
  int n;
  int cand_k;
  int stride;
  int visited_words;
  int meta_words;
  int *cand_idx;
  float *eta_beta;
} aco_mpi_rank_shared_t;

#ifdef USE_MPI
typedef struct {
  MPI_Request req;
  aco_mpi_sparse_delta_t *recv_buf;
  int *counts;
  int *displs;
  MPI_Datatype type;
  int active;
} aco_mpi_async_sparse_context_t;
#endif

size_t aco_mpi_align_up_64(size_t value);
void *aco_mpi_aligned_calloc_64(size_t bytes);
aco_mpi_matrix_float_t *aco_mpi_matrix_create_float(int n);
void aco_mpi_matrix_free_float(aco_mpi_matrix_float_t *m);
int aco_mpi_choose_candidate_count(int n, int requested_candidate_k);
int aco_mpi_shared_init(aco_mpi_rank_shared_t *s, int n, int cand_k,
                        const aco_mpi_matrix_float_t *c_mat, double beta);
void aco_mpi_shared_free(aco_mpi_rank_shared_t *s);

#ifdef USE_MPI
void aco_mpi_async_sparse_init(aco_mpi_async_sparse_context_t *ctx,
                               int mpi_size);
void aco_mpi_async_sparse_cleanup(aco_mpi_async_sparse_context_t *ctx);
void aco_mpi_async_sparse_wait_and_apply(aco_mpi_async_sparse_context_t *ctx,
                                         aco_mpi_matrix_float_t *tau,
                                         int mpi_rank, int mpi_size);
void aco_mpi_async_sparse_start(aco_mpi_async_sparse_context_t *ctx,
                                aco_mpi_sparse_delta_t *local_deltas,
                                int local_count, int mpi_size);
#endif

#endif
