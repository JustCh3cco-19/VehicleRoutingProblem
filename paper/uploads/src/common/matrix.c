#include "matrix.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MATRIX_ALIGNMENT 64u

/*
 * Function:  align_up_size
 * ------------------------
 * rounds a size value up to the next multiple of alignment.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}

/*
 * Function:  matrix_alloc
 * -----------------------
 * allocates a contiguous (n+1) x (n+1) matrix with:
 * - 64-byte aligned base pointer
 * - padded row stride so each row starts on a cache-line boundary
 * - zero-initialized storage
 * a secondary array of row pointers is built for m[i][j] access.
 *
 *  n: highest node index; effective matrix side length is n+1
 *
 *  returns: matrix pointer on success
 *           NULL on allocation failure
 */
double **matrix_alloc(int n) {
  if (n < 0) {
    return NULL;
  }

  int size = n + 1;
  size_t row_bytes = (size_t)size * sizeof(double);
  size_t padded_row_bytes = align_up_size(row_bytes, MATRIX_ALIGNMENT);
  size_t stride = padded_row_bytes / sizeof(double);
  size_t total_elems = (size_t)size * stride;
  size_t total_bytes = total_elems * sizeof(double);
  size_t aligned_total_bytes = align_up_size(total_bytes, MATRIX_ALIGNMENT);

  double *data = aligned_alloc(MATRIX_ALIGNMENT, aligned_total_bytes);
  if (!data) {
    return NULL;
  }
  memset(data, 0, aligned_total_bytes);

  double **m = malloc((size_t)size * sizeof(double *));
  if (!m) {
    free(data);
    return NULL;
  }

  for (int i = 0; i < size; ++i) {
    m[i] = data + (size_t)i * stride;
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
