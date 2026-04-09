#ifndef INSTANCE_PARSER_H
#define INSTANCE_PARSER_H

/* Minimal TSPLIB/VRP parser (EUC_2D) for the project formulation:
 * - depot demand must be 0
 * - every customer demand must be 1
 * - vehicle capacity must be n - K + 3, with K read from VEHICLES
 * Coordinates are converted to a full symmetric distance matrix.
 */
int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out);

#endif
