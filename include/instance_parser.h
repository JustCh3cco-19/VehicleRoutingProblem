#ifndef INSTANCE_PARSER_H
#define INSTANCE_PARSER_H

typedef struct {
  int n;
  int vehicles;
  int capacity;
  double *coords_x;
  double *coords_y;
  int *demands;
} VrpInstance;

void vrp_instance_init(VrpInstance *instance);
void vrp_instance_free(VrpInstance *instance);
int vrp_load_tsplib_instance(const char *path, VrpInstance *instance);
int vrp_instance_create_euc2d_matrix(const VrpInstance *instance,
                                     double ***c_out);
int vrp_instance_create_float_coords(const VrpInstance *instance,
                                     float **coords_x_out,
                                     float **coords_y_out);

#endif
