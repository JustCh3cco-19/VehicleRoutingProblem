#ifndef INSTANCE_PARSER_H
#define INSTANCE_PARSER_H

/* Minimal TSPLIB/VRP parser (EUC_2D):
 * - DIMENSION
 * - NODE_COORD_SECTION
 * Coordinates are converted to a full symmetric distance matrix.
 */
int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out);

#endif
