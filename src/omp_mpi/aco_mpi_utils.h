#ifndef ACO_MPI_UTILS_H
#define ACO_MPI_UTILS_H

#include "solution.h"

void aco_mpi_ant_range_for_rank(int m, int rank, int world_size,
                                int *offset, int *count);
int aco_mpi_rank_for_ant(int m, int world_size, int ant_id);

int aco_mpi_solution_to_flat_packed(const Solution *s, int K,
                                    int *lens, int *flat_nodes);

void aco_mpi_flat_packed_to_solution(Solution *dst, int K,
                                     const int *lens, int total_nodes,
                                     const int *flat_nodes);

#endif
