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
 * Function:  matrix_create
 * -----------------------
 * allocates a square (n+1) x (n+1) matrix of doubles initialized to zero.
 */
Matrix *matrix_create(int n);

/*
 * Function:  matrix_free_handle
 * ----------------------
 * releases a matrix previously returned by matrix_create.
 */
void matrix_free_handle(Matrix *m);

/* Legacy support (to be phased out) */
double **matrix_alloc(int n);
void matrix_free(double **m);

#endif
