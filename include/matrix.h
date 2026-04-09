#ifndef MATRIX_H
#define MATRIX_H

/*
 * Function:  matrix_alloc
 * -----------------------
 * allocates a square (n+1) x (n+1) matrix of doubles initialized to zero.
 */
double **matrix_alloc(int n);

/*
 * Function:  matrix_free
 * ----------------------
 * releases a matrix previously returned by matrix_alloc.
 */
void matrix_free(double **m);

#endif
