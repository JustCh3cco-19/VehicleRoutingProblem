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
Matrix *matrix_create(int n) {
  if (n < 0) return NULL;

  Matrix *m = malloc(sizeof(Matrix));
  if (!m) return NULL;

  int size = n + 1;
  size_t row_bytes = (size_t)size * sizeof(double);
  size_t padded_row_bytes = align_up_size(row_bytes, MATRIX_ALIGNMENT);
  m->stride = (int)(padded_row_bytes / sizeof(double));
  m->n = n;

  size_t total_elems = (size_t)size * (size_t)m->stride;
  size_t total_bytes = total_elems * sizeof(double);
  size_t aligned_total_bytes = align_up_size(total_bytes, MATRIX_ALIGNMENT);

  m->data = aligned_alloc(MATRIX_ALIGNMENT, aligned_total_bytes);
  if (!m->data) {
    free(m);
    return NULL;
  }
  memset(m->data, 0, aligned_total_bytes);

  m->rows = malloc((size_t)size * sizeof(double *));
  if (!m->rows) {
    free(m->data);
    free(m);
    return NULL;
  }

  for (int i = 0; i < size; ++i) {
    m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
  }

  return m;
}

void matrix_free_handle(Matrix *m) {
  if (!m) return;
  free(m->rows);
  free(m->data);
  free(m);
}

double **matrix_alloc(int n) {
  Matrix *m = matrix_create(n);
  if (!m) return NULL;
  double **rows = m->rows;
  // We lose the handle here in legacy mode, but this matches old behavior
  // Actually, to keep it compatible with matrix_free(m[0], m), we'd need to leak m.
  // Let's just keep the old implementation for matrix_alloc to be safe.
  free(m); // but m->rows and m->data are still valid
  return rows;
}

void matrix_free(double **m) {
  if (!m) return;
  free(m[0]);
  free(m);
}
