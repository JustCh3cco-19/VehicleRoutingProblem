#include "matrix.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MATRIX_ALIGNMENT 64u


/**
 * @brief Executes `align_up_size`.
 * @param value Function parameter.
 * @param alignment Function parameter.
 * @return Function result.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}


/**
 * @brief Executes `matrix_create`.
 * @param n Function parameter.
 * @return Function result.
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

/**
 * @brief Executes `matrix_free_handle`.
 * @param m Function parameter.
 */
void matrix_free_handle(Matrix *m) {
  if (!m) return;
  free(m->rows);
  free(m->data);
  free(m);
}

/**
 * @brief Executes `matrix_alloc`.
 * @param n Function parameter.
 * @return Function result.
 */
double **matrix_alloc(int n) {
  Matrix *m = matrix_create(n);
  if (!m) return NULL;
  double **rows = m->rows;



  free(m);
  return rows;
}

/**
 * @brief Executes `matrix_free`.
 * @param m Function parameter.
 */
void matrix_free(double **m) {
  if (!m) return;
  free(m[0]);
  free(m);
}
