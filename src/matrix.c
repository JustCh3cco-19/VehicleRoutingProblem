#include "matrix.h"

#include <stdlib.h>

double **matrix_alloc(int n) {
  int size = n + 1;
  double *data = calloc((size_t)size * (size_t)size, sizeof(double));
  if (!data) return NULL;

  double **m = malloc((size_t)size * sizeof(double *));
  if (!m) {
    free(data);
    return NULL;
  }

  for (int i = 0; i < size; ++i) {
    m[i] = data + (size_t)i * size;
  }

  return m;
}

void matrix_free(double **m) {
  if (!m) return;
  free(m[0]);
  free(m);
}
