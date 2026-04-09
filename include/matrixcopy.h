#ifndef MATRIX_H
#define MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate a (n+1)x(n+1) contiguous matrix with row pointers. */
double **matrix_alloc(int n);
void matrix_free(double **m);

#ifdef __cplusplus
}
#endif

#endif
