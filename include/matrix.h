#ifndef MATRIX_H
# define MATRIX_H

# include <stddef.h>

struct s_matrix
{
	int						n;
	int						stride;
	double					*data;
	double					**rows;
};
typedef struct s_matrix		t_matrix;

enum e_matrix_constants
{
	matrix_default_alignment = 64
};

int							matrix_mul_size(size_t a, size_t b, size_t *out);
int							matrix_align_up(size_t value, size_t alignment,
								size_t *out);
void						*matrix_aligned_calloc(size_t bytes,
								size_t alignment);
t_matrix					*matrix_create(int n);
void						matrix_free_handle(t_matrix *m);
double						**matrix_alloc(int n);
void						matrix_free(double **m);

#endif
