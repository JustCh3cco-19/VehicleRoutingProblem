#ifndef MATRIX_H
#define MATRIX_H

/* Allocate a (n+1)x(n+1) contiguous matrix with row pointers. */
double **matrix_alloc(int n);
void matrix_free(double **m);

#endif
