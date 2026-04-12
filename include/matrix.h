#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

typedef struct {
  int n;
  int stride;
  double *data;
  double **rows;
} Matrix;

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
