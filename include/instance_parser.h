#ifndef INSTANCE_PARSER_H
#define INSTANCE_PARSER_H

/* Minimal TSPLIB/VRP parser (EUC_2D) for the project's unit-demand CVRP
 * instances. Coordinates are converted to a full symmetric distance matrix.
 */
typedef struct {
  int vehicles;
  int capacity;
} VrpInstanceMeta;

int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out);
int vrp_load_tsplib_euc2d_matrix_ex(const char *path, int *n_out,
                                    double ***c_out,
                                    VrpInstanceMeta *meta_out);

#endif
