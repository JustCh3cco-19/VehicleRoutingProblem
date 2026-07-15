#include "instance_parser.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double euc_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int vrp_load_tsplib_euc2d_matrix_ex(const char *path, int *n_out,
                                    double ***c_out,
                                    VrpInstanceMeta *meta_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int n = 0;
    int K = 0;
    int capacity = 0;
    char line[256];
    double *coords_x = NULL;
    double *coords_y = NULL;
    int *demands = NULL;
    int have_coords = 0;
    int have_demands = 0;
    int have_capacity = 0;
    int have_vehicles = 0;

    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "DIMENSION", 9) == 0) {
            char *colon = strchr(line, ':');
            if (colon) {
                n = atoi(colon + 1) - 1; // n is number of customers, DIMENSION is n+1
            } else {
                sscanf(line, "DIMENSION: %d", &n); // try without space
                if (n == 0) sscanf(line, "DIMENSION %d", &n); // try without colon
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
            coords_x = malloc((size_t)(n + 1) * sizeof(double));
            coords_y = malloc((size_t)(n + 1) * sizeof(double));
            if (!coords_x || !coords_y) {
                free(coords_x);
                free(coords_y);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                if (fscanf(f, "%d %lf %lf", &id, &coords_x[i], &coords_y[i]) != 3) {
                    fprintf(stderr, "Failed to read coordinates at index %d\n", i);
                    free(coords_x);
                    free(coords_y);
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
            demands = malloc((size_t)(n + 1) * sizeof(int));
            if (!demands) {
                free(coords_x);
                free(coords_y);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                if (fscanf(f, "%d %d", &id, &demands[i]) != 2) {
                    fprintf(stderr, "Failed to read demands at index %d\n", i);
                    free(coords_x);
                    free(coords_y);
                    free(demands);
                    fclose(f);
                    return 1;
                }
            }
            have_demands = 1;
        }
    }

    if (n <= 0 || !have_coords || !coords_x || !coords_y) {
        fprintf(stderr, "Invalid or missing DIMENSION/NODE_COORD_SECTION\n");
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    if (!have_vehicles || !have_capacity || !have_demands) {
        fprintf(stderr, "Missing VEHICLES/CAPACITY/DEMAND_SECTION in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    if (demands[0] != 0) {
        fprintf(stderr, "Depot demand must be 0 in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }
    for (int i = 1; i <= n; ++i) {
        if (demands[i] != 1) {
            fprintf(stderr, "Customer demand must be 1 for all customers in %s\n", path);
            free(coords_x);
            free(coords_y);
            free(demands);
            fclose(f);
            return 1;
        }
    }

    if (capacity <= 0) {
        fprintf(stderr, "CAPACITY must be positive in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    double **c = matrix_alloc(n);
    if (!c) {
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            c[i][j] = euc_2d(coords_x[i], coords_y[i], coords_x[j], coords_y[j]);
        }
    }

    free(coords_x);
    free(coords_y);
    free(demands);
    fclose(f);

    *n_out = n;
    *c_out = c;
    if (meta_out) {
        meta_out->vehicles = K;
        meta_out->capacity = capacity;
    }
    return 0;
}

int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out) {
    return vrp_load_tsplib_euc2d_matrix_ex(path, n_out, c_out, NULL);
}

int vrp_load_tsplib_euc2d_coords(const char *path, int *n_out,
                                 float **coords_x_out, float **coords_y_out,
                                 VrpInstanceMeta *meta_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int n = 0;
    int K = 0;
    int capacity = 0;
    char line[256];
    float *coords_x = NULL;
    float *coords_y = NULL;
    int *demands = NULL;
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
            coords_x = malloc((size_t)(n + 1) * sizeof(float));
            coords_y = malloc((size_t)(n + 1) * sizeof(float));
            if (!coords_x || !coords_y) {
                free(coords_x);
                free(coords_y);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                double x, y;
                if (fscanf(f, "%d %lf %lf", &id, &x, &y) != 3) {
                    fprintf(stderr, "Failed to read coordinates at index %d\n", i);
                    free(coords_x);
                    free(coords_y);
                    fclose(f);
                    return 1;
                }
                coords_x[i] = (float)x;
                coords_y[i] = (float)y;
            }
            have_coords = 1;
        } else if (strncmp(line, "DEMAND_SECTION", 14) == 0) {
            if (n <= 0) {
                fprintf(stderr, "DIMENSION must be defined before DEMAND_SECTION\n");
                fclose(f);
                return 1;
            }
            demands = malloc((size_t)(n + 1) * sizeof(int));
            if (!demands) {
                free(coords_x);
                free(coords_y);
                fclose(f);
                return 1;
            }
            for (int i = 0; i <= n; ++i) {
                int id;
                if (fscanf(f, "%d %d", &id, &demands[i]) != 2) {
                    fprintf(stderr, "Failed to read demands at index %d\n", i);
                    free(coords_x);
                    free(coords_y);
                    free(demands);
                    fclose(f);
                    return 1;
                }
            }
            have_demands = 1;
        }
    }

    if (n <= 0 || !have_coords || !coords_x || !coords_y) {
        fprintf(stderr, "Invalid or missing DIMENSION/NODE_COORD_SECTION\n");
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    if (!have_vehicles || !have_capacity || !have_demands) {
        fprintf(stderr, "Missing VEHICLES/CAPACITY/DEMAND_SECTION in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    if (demands[0] != 0) {
        fprintf(stderr, "Depot demand must be 0 in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }
    for (int i = 1; i <= n; ++i) {
        if (demands[i] != 1) {
            fprintf(stderr, "Customer demand must be 1 for all customers in %s\n", path);
            free(coords_x);
            free(coords_y);
            free(demands);
            fclose(f);
            return 1;
        }
    }

    if (capacity <= 0) {
        fprintf(stderr, "CAPACITY must be positive in %s\n", path);
        free(coords_x);
        free(coords_y);
        free(demands);
        fclose(f);
        return 1;
    }

    free(demands);
    fclose(f);

    *n_out = n;
    *coords_x_out = coords_x;
    *coords_y_out = coords_y;
    if (meta_out) {
        meta_out->vehicles = K;
        meta_out->capacity = capacity;
    }
    return 0;
}
