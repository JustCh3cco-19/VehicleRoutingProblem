#include "matrix.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int matrix_mul_size(size_t a, size_t b, size_t *out) {
  if (!out) {
    return 0;
  }
  if (a != 0u && b > SIZE_MAX / a) {
    return 0;
  }
  *out = a * b;
  return 1;
}

int matrix_align_up(size_t value, size_t alignment, size_t *out) {
  if (!out || alignment == 0u) {
    return 0;
  }
  size_t rem = value % alignment;
  if (rem == 0u) {
    *out = value;
    return 1;
  }

  size_t padding = alignment - rem;
  if (value > SIZE_MAX - padding) {
    return 0;
  }
  *out = value + padding;
  return 1;
}

void *matrix_aligned_calloc(size_t bytes, size_t alignment) {
  size_t alloc_bytes = 0;
  if (!matrix_align_up(bytes, alignment, &alloc_bytes)) {
    return NULL;
  }

  void *ptr = aligned_alloc(alignment, alloc_bytes);
  if (!ptr) {
    return NULL;
  }
  memset(ptr, 0, alloc_bytes);
  return ptr;
}


Matrix *matrix_create(int n) {
  if (n < 0) {
    return NULL;
  }

  size_t size = (size_t)n + 1u;
  if (size > (size_t)INT32_MAX) {
    return NULL;
  }

  size_t row_bytes = 0;
  size_t padded_row_bytes = 0;
  size_t total_elems = 0;
  size_t total_bytes = 0;
  if (!matrix_mul_size(size, sizeof(double), &row_bytes) ||
      !matrix_align_up(row_bytes, kMatrixDefaultAlignment,
                       &padded_row_bytes)) {
    return NULL;
  }
  size_t stride = padded_row_bytes / sizeof(double);
  if (stride > (size_t)INT32_MAX ||
      !matrix_mul_size(size, stride, &total_elems) ||
      !matrix_mul_size(total_elems, sizeof(double), &total_bytes)) {
    return NULL;
  }

  Matrix *m = malloc(sizeof(*m));
  if (!m) {
    return NULL;
  }

  m->stride = (int)stride;
  m->n = n;
  m->data = matrix_aligned_calloc(total_bytes, kMatrixDefaultAlignment);
  if (!m->data) {
    free(m);
    return NULL;
  }

  m->rows = malloc(size * sizeof(double *));
  if (!m->rows) {
    free(m->data);
    free(m);
    return NULL;
  }

  for (int i = 0; i < (int)size; ++i) {
    m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
  }

  return m;
}

void matrix_free_handle(Matrix *m) {
  if (!m) {
    return;
  }
  free(m->rows);
  free(m->data);
  free(m);
}

double **matrix_alloc(int n) {
  Matrix *m = matrix_create(n);
  if (!m) {
    return NULL;
  }
  double **rows = m->rows;

  free(m);
  return rows;
}

void matrix_free(double **m) {
  if (!m) {
    return;
  }
  free(m[0]);
  free(m);
}
