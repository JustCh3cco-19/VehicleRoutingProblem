#ifndef MATRIX_H
#define MATRIX_H

/*
 * Function:  matrix_alloc
 * -----------------------
 * allocates a square (n+1) x (n+1) matrix of doubles initialized to zero.
 * the matrix is stored as one contiguous aligned data block plus an array
 * of row pointers for O(1) indexing with m[i][j]. rows are padded so each
 * row starts on a 64-byte boundary for cache/SIMD-friendly traversal.
 *
 *  n: highest node index; the allocated matrix size is (n+1)
 *
 *  returns: a valid matrix pointer on success
 *           NULL on allocation error
 */
double **matrix_alloc(int n);

/*
 * Function:  matrix_free
 * ----------------------
 * releases a matrix previously returned by matrix_alloc.
 *
 *  m: matrix pointer to free; NULL is allowed
 *
 *  returns: nothing
 */
void matrix_free(double **m);

#endif
