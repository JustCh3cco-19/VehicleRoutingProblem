#include "instance_parser.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief Executes `euc_2d`.
 * @param x1 Function parameter.
 * @param y1 Function parameter.
 * @param x2 Function parameter.
 * @param y2 Function parameter.
 * @return Function result.
 */
static double euc_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

void vrp_instance_init(VrpInstance *instance) {
    if (!instance) return;
    memset(instance, 0, sizeof(*instance));
}

void vrp_instance_free(VrpInstance *instance) {
    if (!instance) return;
    free(instance->coords_x);
    free(instance->coords_y);
    free(instance->demands);
    vrp_instance_init(instance);
}

/**
 * @brief Executes `vrp_load_tsplib_instance`.
 * @param path Input instance path.
 * @param instance Output instance.
 * @return Function result.
 */
int vrp_load_tsplib_instance(const char *path, VrpInstance *instance) {
    if (!path || !instance) {
        return 1;
    }
    vrp_instance_init(instance);

    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int n = 0;
    int K = 0;
    int capacity = 0;
    char line[256];
    int have_coords = 0;
    int have_demands = 0;
    int have_capacity = 0;
    int have_vehicles = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "DIMENSION", 9) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                n = atoi(colon + 1) - 1;
            } else {
                sscanf(line, "DIMENSION: %d", &n);
                if (n == 0) sscanf(line, "DIMENSION %d", &n);
                n -= 1;
            }
        } else if (strncmp(line, "VEHICLES", 8) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                K = atoi(colon + 1);
            } else {
                sscanf(line, "VEHICLES: %d", &K);
                if (K == 0) sscanf(line, "VEHICLES %d", &K);
            }
            have_vehicles = (K > 0);
        } else if (strncmp(line, "CAPACITY", 8) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                capacity = atoi(colon + 1);
            } else {
                sscanf(line, "CAPACITY: %d", &capacity);
                if (capacity == 0) sscanf(line, "CAPACITY %d", &capacity);
            }
            have_capacity = (capacity > 0);
        } else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0) {
            if (n <= 0) {
                fprintf(stderr, "DIMENSION must be defined before NODE_COORD_SECTION\n");
                fclose(f);
                return 1;
            }
            instance->coords_x = malloc((size_t)(n + 1) * sizeof(double));
            instance->coords_y = malloc((size_t)(n + 1) * sizeof(double));
            if (!instance->coords_x || !instance->coords_y) {
                vrp_instance_free(instance);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                if (fscanf(f, "%d %lf %lf", &id, &instance->coords_x[i], &instance->coords_y[i]) != 3) {
                    fprintf(stderr, "Failed to read coordinates at index %d\n", i);
                    vrp_instance_free(instance);
                    fclose(f);
                    return 1;
                }
            }
            have_coords = 1;
        } else if (strncmp(line, "DEMAND_SECTION", 14) == 0) {
            if (n <= 0) {
                fprintf(stderr, "DIMENSION must be defined before DEMAND_SECTION\n");
                fclose(f);
                return 1;
            }
            instance->demands = malloc((size_t)(n + 1) * sizeof(int));
            if (!instance->demands) {
                vrp_instance_free(instance);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                if (fscanf(f, "%d %d", &id, &instance->demands[i]) != 2) {
                    fprintf(stderr, "Failed to read demands at index %d\n", i);
                    vrp_instance_free(instance);
                    fclose(f);
                    return 1;
                }
            }
            have_demands = 1;
        }
    }

    if (n <= 0 || !have_coords || !instance->coords_x || !instance->coords_y) {
        fprintf(stderr, "Invalid or missing DIMENSION/NODE_COORD_SECTION\n");
        vrp_instance_free(instance);
        fclose(f);
        return 1;
    }

    if (!have_vehicles || !have_capacity || !have_demands) {
        fprintf(stderr, "Missing VEHICLES/CAPACITY/DEMAND_SECTION in %s\n", path);
        vrp_instance_free(instance);
        fclose(f);
        return 1;
    }

    if (instance->demands[0] != 0) {
        fprintf(stderr, "Depot demand must be 0 in %s\n", path);
        vrp_instance_free(instance);
        fclose(f);
        return 1;
    }
    for (int i = 1; i <= n; ++i) {
        if (instance->demands[i] != 1) {
            fprintf(stderr, "Customer demand must be 1 for all customers in %s\n", path);
            vrp_instance_free(instance);
            fclose(f);
            return 1;
        }
    }

    if (capacity <= 0) {
        fprintf(stderr, "CAPACITY must be positive in %s\n", path);
        vrp_instance_free(instance);
        fclose(f);
        return 1;
    }

    fclose(f);
    instance->n = n;
    instance->vehicles = K;
    instance->capacity = capacity;
    return 0;
}

int vrp_instance_create_euc2d_matrix(const VrpInstance *instance,
                                     double ***c_out) {
    if (!instance || !c_out || instance->n <= 0 || !instance->coords_x ||
        !instance->coords_y) {
        return 1;
    }

    int n = instance->n;
    double **c = matrix_alloc(n);
    if (!c) {
        return 1;
    }

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            c[i][j] = euc_2d(instance->coords_x[i], instance->coords_y[i],
                             instance->coords_x[j], instance->coords_y[j]);
        }
    }

    *c_out = c;
    return 0;
}

int vrp_instance_create_float_coords(const VrpInstance *instance,
                                     float **coords_x_out,
                                     float **coords_y_out) {
    if (!instance || !coords_x_out || !coords_y_out || instance->n <= 0 ||
        !instance->coords_x || !instance->coords_y) {
        return 1;
    }

    float *coords_x = malloc((size_t)(instance->n + 1) * sizeof(float));
    float *coords_y = malloc((size_t)(instance->n + 1) * sizeof(float));
    if (!coords_x || !coords_y) {
        free(coords_x);
        free(coords_y);
        return 1;
    }

    for (int i = 0; i <= instance->n; ++i) {
        coords_x[i] = (float)instance->coords_x[i];
        coords_y[i] = (float)instance->coords_y[i];
    }

    *coords_x_out = coords_x;
    *coords_y_out = coords_y;
    return 0;
}
