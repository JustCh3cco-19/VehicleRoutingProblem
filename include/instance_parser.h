#ifndef INSTANCE_PARSER_H
#define INSTANCE_PARSER_H

/**
 * @brief Metadata extracted from a TSPLIB CVRP instance.
 */
typedef struct {
  int vehicles;
  int capacity;
} VrpInstanceMeta;

/**
 * @brief Loads an EUC_2D TSPLIB instance into a full symmetric distance matrix.
 * @param path Input file path.
 * @param n_out Output number of customers.
 * @param c_out Output distance matrix.
 * @return 0 on success, non-zero on error.
 */
int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out);

/**
 * @brief Loads an EUC_2D TSPLIB instance with additional metadata.
 * @param path Input file path.
 * @param n_out Output number of customers.
 * @param c_out Output distance matrix.
 * @param meta_out Output metadata (vehicles/capacity).
 * @return 0 on success, non-zero on error.
 */
int vrp_load_tsplib_euc2d_matrix_ex(const char *path, int *n_out,
                                    double ***c_out,
                                    VrpInstanceMeta *meta_out);

/**
 * @brief Loads an EUC_2D TSPLIB instance as coordinate arrays.
 * @param path Input file path.
 * @param n_out Output number of customers.
 * @param coords_x_out Output x coordinates.
 * @param coords_y_out Output y coordinates.
 * @param meta_out Output metadata (vehicles/capacity).
 * @return 0 on success, non-zero on error.
 */
int vrp_load_tsplib_euc2d_coords(const char *path, int *n_out,
                                 float **coords_x_out, float **coords_y_out,
                                 VrpInstanceMeta *meta_out);

#endif
