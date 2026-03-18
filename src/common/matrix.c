#include "matrix.h"

#include <stdlib.h>

/*
 * Function:  matrix_alloc
 * -----------------------
 * allocates a contiguous (n+1) x (n+1) matrix and initializes all entries to
 * zero. a secondary array of row pointers is built for m[i][j] access.
 *
 *  n: highest node index; effective matrix side length is n+1
 *
 *  returns: matrix pointer on success
 *           NULL on allocation failure
 */
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

/*
 * Function:  matrix_free
 * ----------------------
 * frees a matrix allocated by matrix_alloc by releasing contiguous data first
 * and row-pointer array second.
 *
 *  m: matrix pointer returned by matrix_alloc; NULL is accepted
 *
 *  returns: nothing
 */
void matrix_free(double **m) {
  if (!m) return;
  free(m[0]);
  free(m);
}
