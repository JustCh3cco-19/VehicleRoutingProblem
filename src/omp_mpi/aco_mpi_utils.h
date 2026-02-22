#ifndef ACO_MPI_UTILS_H
#define ACO_MPI_UTILS_H

#include "solution.h"

void aco_mpi_ant_range_for_rank(int m, int rank, int world_size,
                                int *offset, int *count);

void aco_mpi_solution_to_flat(const Solution *s, int K, int n,
                              int *lens, int *flat_nodes);

void aco_mpi_flat_to_solution(Solution *dst, int K, int n,
                              const int *lens, const int *flat_nodes);

#endif
