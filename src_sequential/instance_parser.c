#include "instance_parser.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double euc_2d(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int n = 0;
    char line[256];
    double *coords_x = NULL;
    double *coords_y = NULL;

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
        } else if (strncmp(line, "NODE_COORD_SECTION", 18) == 0) {
            if (n <= 0) {
                fprintf(stderr, "DIMENSION must be defined before NODE_COORD_SECTION\n");
                fclose(f);
                return 1;
            }
            coords_x = malloc((size_t)(n + 1) * sizeof(double));
            coords_y = malloc((size_t)(n + 1) * sizeof(double));
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
            break;
        }
    }

    if (n <= 0 || !coords_x) {
        fprintf(stderr, "Invalid or missing DIMENSION/NODE_COORD_SECTION\n");
        fclose(f);
        return 1;
    }

    double **c = matrix_alloc(n);
    if (!c) {
        free(coords_x);
        free(coords_y);
        fclose(f);
        return 1;
    }

    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            c[i][j] = round(euc_2d(coords_x[i], coords_y[i], coords_x[j], coords_y[j]));
        }
    }

    free(coords_x);
    free(coords_y);
    fclose(f);

    *n_out = n;
    *c_out = c;
    return 0;
}
