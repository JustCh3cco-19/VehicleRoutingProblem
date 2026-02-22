#include "aco.h"

#include "aco_cuda_host_utils.h"
#include "aco_cuda_kernels.h"

#include <float.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK_GOTO(call, label)                                            \
  do {                                                                           \
    cudaError_t err__ = (call);                                                  \
    if (err__ != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s:%d (%s): %s\\n", __FILE__, __LINE__,      \
              #call, cudaGetErrorString(err__));                                 \
      rc = -1;                                                                   \
      goto label;                                                                \
    }                                                                            \
  } while (0)

int aco_vrp_cuda(int n, int K, int m, int T, double **c,
                 double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 Solution *best_solution, double *best_cost) {
  if (n > ACO_CUDA_MAX_N) {
    fprintf(stderr, "n=%d exceeds CUDA limit ACO_CUDA_MAX_N=%d\n",
            n, ACO_CUDA_MAX_N);
    return -1;
  }

  int rc = 0;
  int block = ACO_CUDA_BLOCK_SIZE;
  int grid_matrix = 0;
  int grid_ants = 0;

  float alpha_f = (float)alpha;
  float beta_f = (float)beta;
  float rho_f = (float)rho;
  float tau0_f = (float)tau0;
  float Q_f = (float)Q;

  AcoCudaBuffers buf;
  aco_cuda_buffers_init(&buf);

  Solution *iter_best = solution_create(K, n);
  if (!iter_best) {
    fprintf(stderr, "allocation failure in aco_vrp_cuda\n");
    return -1;
  }

  CUDA_CHECK_GOTO(aco_cuda_buffers_alloc(&buf, n, K, m, ACO_CUDA_MAX_CANDIDATES),
                  cleanup);

  aco_cuda_copy_cost_to_flat(c, n, buf.h_c);
  aco_cuda_build_candidates(c, n, buf.candidate_count, buf.h_candidates);

  CUDA_CHECK_GOTO(cudaMemcpyAsync(buf.d_c, buf.h_c, buf.matrix_bytes,
                                  cudaMemcpyHostToDevice, buf.stream_compute),
                  cleanup);
  CUDA_CHECK_GOTO(cudaMemcpyAsync(buf.d_candidates, buf.h_candidates,
                                  buf.candidates_bytes,
                                  cudaMemcpyHostToDevice, buf.stream_compute),
                  cleanup);

  grid_matrix = (buf.matrix_elems + block - 1) / block;
  grid_ants = (m + block - 1) / block;

  aco_cuda_init_eta_tau_kernel<<<grid_matrix, block, 0, buf.stream_compute>>>(
      buf.d_eta, buf.d_tau, buf.d_c, n, tau0_f);
  CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
  CUDA_CHECK_GOTO(cudaStreamSynchronize(buf.stream_compute), cleanup);

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  for (int iter = 0; iter < T; ++iter) {
    aco_cuda_construct_ants_kernel<<<grid_ants, block, 0, buf.stream_compute>>>(
        n, K, m, iter, buf.d_c, buf.d_tau, buf.d_eta,
        buf.d_candidates, buf.candidate_count,
        alpha_f, beta_f, seed,
        buf.d_routes, buf.d_lens, buf.d_costs);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    int reduce_in = grid_ants;
    float *reduce_cost_in = buf.d_reduce_costs_a;
    int *reduce_id_in = buf.d_reduce_ids_a;
    float *reduce_cost_out = buf.d_reduce_costs_b;
    int *reduce_id_out = buf.d_reduce_ids_b;

    aco_cuda_reduce_costs_stage_kernel<<<reduce_in, block, 0, buf.stream_compute>>>(
        buf.d_costs, m, reduce_cost_in, reduce_id_in);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    while (reduce_in > 1) {
      int reduce_out = (reduce_in + block - 1) / block;
      aco_cuda_reduce_pairs_stage_kernel<<<reduce_out, block, 0, buf.stream_compute>>>(
          reduce_cost_in, reduce_id_in, reduce_in, reduce_cost_out, reduce_id_out);
      CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

      float *tmp_cost = reduce_cost_in;
      int *tmp_id = reduce_id_in;
      reduce_cost_in = reduce_cost_out;
      reduce_id_in = reduce_id_out;
      reduce_cost_out = tmp_cost;
      reduce_id_out = tmp_id;
      reduce_in = reduce_out;
    }

    float iter_best_cost_f = FLT_MAX;
    int iter_best_ant = m;

    CUDA_CHECK_GOTO(cudaStreamSynchronize(buf.stream_compute), cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(&iter_best_cost_f, reduce_cost_in,
                               sizeof(float), cudaMemcpyDeviceToHost),
                    cleanup);
    CUDA_CHECK_GOTO(cudaMemcpy(&iter_best_ant, reduce_id_in,
                               sizeof(int), cudaMemcpyDeviceToHost),
                    cleanup);

    aco_cuda_evaporate_tau_kernel<<<grid_matrix, block, 0, buf.stream_compute>>>(
        buf.d_tau, n, rho_f);
    CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);

    if (iter_best_ant < m && iter_best_cost_f < FLT_MAX) {
      float deposit = Q_f / iter_best_cost_f;
      aco_cuda_deposit_best_tau_kernel<<<1, 1, 0, buf.stream_compute>>>(
          buf.d_tau, n, K, buf.max_route_len,
          buf.d_routes, buf.d_lens,
          iter_best_ant, deposit);
      CUDA_CHECK_GOTO(cudaGetLastError(), cleanup);
    }

    if (iter_best_ant < m && iter_best_cost_f < FLT_MAX) {
      double iter_cost = (double)iter_best_cost_f;
      if (aco_cuda_better_candidate(iter_best_cost_f, iter_best_ant,
                                    (float)(*best_cost), m + 1)) {
        CUDA_CHECK_GOTO(aco_cuda_copy_ant_from_device(&buf, iter_best_ant,
                                                      iter_best),
                        cleanup);
        *best_cost = iter_cost;
        solution_copy(best_solution, iter_best);
      }
    }
  }

cleanup:
  aco_cuda_buffers_free(&buf);
  solution_free(iter_best);
  return rc;
}
