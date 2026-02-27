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

int aco_mpi_rank_for_ant(int m, int world_size, int ant_id) {
  int base = m / world_size;
  int rem = m % world_size;
  int first_block = rem * (base + 1);

  if (ant_id < 0 || ant_id >= m) {
    return world_size;
  }
  if (ant_id < first_block) {
    return ant_id / (base + 1);
  }
  if (base == 0) {
    return world_size;
  }
  return rem + (ant_id - first_block) / base;
}

int aco_mpi_solution_to_flat_packed(const Solution *s, int K,
                                    int *lens, int *flat_nodes) {
  int pos = 0;
  for (int i = 0; i < K; ++i) {
    const Route *r = &s->routes[i];
    lens[i] = r->len;
    for (int t = 0; t < r->len; ++t) {
      flat_nodes[pos++] = r->nodes[t];
    }
  }
  return pos;
}

void aco_mpi_flat_packed_to_solution(Solution *dst, int K,
                                     const int *lens, int total_nodes,
                                     const int *flat_nodes) {
  int pos = 0;
  solution_reset(dst);
  for (int i = 0; i < K; ++i) {
    Route *r = &dst->routes[i];
    int len = lens[i];
    for (int t = 0; t < len && pos < total_nodes; ++t) {
      route_append(r, flat_nodes[pos++]);
    }
  }
}
