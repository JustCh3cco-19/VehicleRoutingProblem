#include "instance_parser.h"

#include "matrix.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double x;
  double y;
  int seen;
} NodeCoord;

static int parse_dimension_line(const char *line, int *dim_out) {
  const char *p = strstr(line, "DIMENSION");
  if (!p) return 0;
  p = strchr(p, ':');
  if (!p) return -1;
  ++p;
  while (*p && isspace((unsigned char)*p)) ++p;
  int dim = atoi(p);
  if (dim <= 0) return -1;
  *dim_out = dim;
  return 1;
}

int vrp_load_tsplib_euc2d_matrix(const char *path, int *n_out, double ***c_out) {
  if (!path || !n_out || !c_out) return -1;

  FILE *fp = fopen(path, "r");
  if (!fp) return -1;

  char line[512];
  int n = 0;
  int in_coords = 0;
  NodeCoord *coords = NULL;
  int parsed = 0;
  int rc = -1;

  while (fgets(line, sizeof(line), fp)) {
    if (!in_coords) {
      int dim_rc = parse_dimension_line(line, &n);
      if (dim_rc < 0) goto cleanup;
      if (dim_rc > 0) {
        coords = calloc((size_t)(n + 1), sizeof(NodeCoord));
        if (!coords) goto cleanup;
      }
      if (strstr(line, "NODE_COORD_SECTION")) {
        if (n <= 0 || !coords) goto cleanup;
        in_coords = 1;
      }
      continue;
    }

    if (strstr(line, "DEMAND_SECTION") || strstr(line, "DEPOT_SECTION") ||
        strstr(line, "EOF")) {
      break;
    }

    int id = 0;
    double x = 0.0;
    double y = 0.0;
    if (sscanf(line, "%d %lf %lf", &id, &x, &y) == 3) {
      if (id < 1 || id > n) goto cleanup;
      coords[id].x = x;
      coords[id].y = y;
      coords[id].seen = 1;
      ++parsed;
    }
  }

  if (n <= 1 || parsed != n) goto cleanup;

  int customers = n - 1;
  double **c = matrix_alloc(customers);
  if (!c) goto cleanup;

  for (int i = 1; i <= n; ++i) {
    if (!coords[i].seen) {
      matrix_free(c);
      goto cleanup;
    }
  }

  c[0][0] = 0.0;
  for (int i = 1; i <= customers; ++i) {
    int node_i = i + 1; /* TSPLIB node 1 is depot; others are customers. */
    double dx0 = coords[node_i].x - coords[1].x;
    double dy0 = coords[node_i].y - coords[1].y;
    double d0 = sqrt(dx0 * dx0 + dy0 * dy0);
    c[0][i] = d0;
    c[i][0] = d0;
  }

  for (int i = 1; i <= customers; ++i) {
    int node_i = i + 1;
    c[i][i] = 0.0;
    for (int j = i + 1; j <= customers; ++j) {
      int node_j = j + 1;
      double dx = coords[node_i].x - coords[node_j].x;
      double dy = coords[node_i].y - coords[node_j].y;
      double d = sqrt(dx * dx + dy * dy);
      c[i][j] = d;
      c[j][i] = d;
    }
  }

  *n_out = customers;
  *c_out = c;
  rc = 0;

cleanup:
  free(coords);
  fclose(fp);
  return rc;
}
