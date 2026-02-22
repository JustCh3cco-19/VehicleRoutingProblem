#include "aco_mpi_utils.h"

void aco_mpi_ant_range_for_rank(int m, int rank, int world_size,
                                int *offset, int *count) {
  /* Balanced block distribution: first "rem" ranks get one extra ant. */
  int base = m / world_size;
  int rem = m % world_size;
  if (rank < rem) {
    *count = base + 1;
    *offset = rank * (*count);
  } else {
    *count = base;
    *offset = rem * (base + 1) + (rank - rem) * base;
  }
}

void aco_mpi_solution_to_flat(const Solution *s, int K, int n,
                              int *lens, int *flat_nodes) {
  /* Flatten variable-size routes into fixed-size slots for MPI_Bcast. */
  int max_route_len = n + 2;
  for (int i = 0; i < K; ++i) {
    const Route *r = &s->routes[i];
    int base = i * max_route_len;
    lens[i] = r->len;
    for (int t = 0; t < r->len; ++t) {
      flat_nodes[base + t] = r->nodes[t];
    }
    for (int t = r->len; t < max_route_len; ++t) {
      flat_nodes[base + t] = 0;
    }
  }
}

void aco_mpi_flat_to_solution(Solution *dst, int K, int n,
                              const int *lens, const int *flat_nodes) {
  int max_route_len = n + 2;
  solution_reset(dst);
  for (int i = 0; i < K; ++i) {
    int base = i * max_route_len;
    Route *r = &dst->routes[i];
    for (int t = 0; t < lens[i]; ++t) {
      route_append(r, flat_nodes[base + t]);
    }
  }
}
