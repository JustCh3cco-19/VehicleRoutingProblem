#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
  int n;
  int stride;
  double *data;
  double **rows;
} Matrix;

/*
 * Matrix type policy:
 * - use Matrix/double for common CPU data, TSPLIB distances and public APIs;
 * - use backend-local float matrices when memory bandwidth/cache pressure is
 *   more important than double precision, as in the MPI/OpenMP hot loop;
 * - use quantized uint8_t buffers only for backend-private representations
 *   whose scaling and precision loss are explicitly owned by that backend.
 *
 * Keep stride, alignment and overflow-checked allocation through the helpers
 * below so matrix backends share the same memory safety rules.
 */
enum {
  kMatrixDefaultAlignment = 64,
};

int matrix_mul_size(size_t a, size_t b, size_t *out);
int matrix_align_up(size_t value, size_t alignment, size_t *out);
void *matrix_aligned_calloc(size_t bytes, size_t alignment);

/**
 * @brief Allocates a square (n+1) x (n+1) zero-initialized matrix.
 * @param n Number of customers.
 * @return Allocated matrix handle or NULL on failure.
 */
Matrix *matrix_create(int n);

/**
 * @brief Releases a matrix allocated with matrix_create().
 * @param m Matrix handle.
 */
void matrix_free_handle(Matrix *m);

/**
 * @brief Legacy matrix allocation API retained for compatibility.
 * @param n Number of customers.
 * @return Distance matrix pointer.
 */
double **matrix_alloc(int n);

/**
 * @brief Legacy matrix free API retained for compatibility.
 * @param m Distance matrix pointer.
 */
void matrix_free(double **m);

#endif
