#include "aco_mpi_internal.h"
#include "matrix.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t aco_mpi_align_up_64(size_t value) {
  size_t aligned = 0;
  if (!matrix_align_up(value, kAcoMpiAlignment, &aligned)) {
    return 0;
  }
  return aligned;
}

void *aco_mpi_aligned_calloc_64(size_t bytes) {
  return matrix_aligned_calloc(bytes, kAcoMpiAlignment);
}

aco_mpi_matrix_float_t *aco_mpi_matrix_create_float(int n) {
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
  if (!matrix_mul_size(size, sizeof(float), &row_bytes) ||
      !matrix_align_up(row_bytes, kAcoMpiAlignment, &padded_row_bytes)) {
    return NULL;
  }
  size_t stride = padded_row_bytes / sizeof(float);
  if (stride > (size_t)INT32_MAX ||
      !matrix_mul_size(size, stride, &total_elems) ||
      !matrix_mul_size(total_elems, sizeof(float), &total_bytes)) {
    return NULL;
  }

  aco_mpi_matrix_float_t *m = malloc(sizeof(*m));
  if (!m) {
    return NULL;
  }

  m->n = n;
  m->stride = (int)stride;
  m->data = matrix_aligned_calloc(total_bytes, kAcoMpiAlignment);
  m->rows = malloc(size * sizeof(float *));
  if (!m->data || !m->rows) {
    free(m->data);
    free(m->rows);
    free(m);
    return NULL;
  }

  for (int i = 0; i <= n; i++) {
    m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
  }
  return m;
}

void aco_mpi_matrix_free_float(aco_mpi_matrix_float_t *m) {
  if (!m) {
    return;
  }
  free(m->data);
  free(m->rows);
  free(m);
}

static long aco_mpi_get_l3_cache_size(void) {
  FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
  if (!f) {
    return 0;
  }

  char buf[64];
  if (!fgets(buf, sizeof(buf), f)) {
    fclose(f);
    return 0;
  }
  fclose(f);

  long size = atol(buf);
  char *unit = strpbrk(buf, "KMGTkmgt");
  if (unit) {
    if (*unit == 'K' || *unit == 'k') {
      size *= 1024;
    } else if (*unit == 'M' || *unit == 'm') {
      size *= 1024 * 1024;
    } else if (*unit == 'G' || *unit == 'g') {
      size *= 1024 * 1024 * 1024;
    }
  }
  return size;
}

int aco_mpi_choose_candidate_count(int n, int requested_candidate_k) {
  if (requested_candidate_k > 0) {
    return (requested_candidate_k > n) ? n : requested_candidate_k;
  }

  long l3_size = aco_mpi_get_l3_cache_size();
  if (l3_size <= 0) {
    return (n < 32) ? n : 32;
  }

  double target_bytes = (double)l3_size * 0.7;
  int k = (int)(target_bytes / ((double)(n + 1) * 8.0));
  if (k < 16) {
    k = 16;
  }
  if (k > kAcoMpiMaxCandidates) {
    k = kAcoMpiMaxCandidates;
  }
  return (k > n) ? n : k;
}

static float aco_mpi_fast_powf(float base, float exp) {
  if (exp == 1.0f) {
    return base;
  }
  if (exp == 2.0f) {
    return base * base;
  }
  if (exp == 0.5f) {
    return sqrtf(base);
  }
  return powf(base, exp);
}

void aco_mpi_shared_free(aco_mpi_rank_shared_t *s) {
  if (!s) {
    return;
  }
  free(s->eta_beta);
  free(s->cand_idx);
}

int aco_mpi_shared_init(aco_mpi_rank_shared_t *s, int n, int cand_k,
                        const aco_mpi_matrix_float_t *c_mat, double beta) {
  memset(s, 0, sizeof(*s));
  s->n = n;
  s->cand_k = cand_k;
  size_t row_bytes = 0;
  size_t padded_row_bytes = 0;
  if (n < 0 || n == INT32_MAX || cand_k <= 0 ||
      !matrix_mul_size((size_t)cand_k, sizeof(float), &row_bytes) ||
      !matrix_align_up(row_bytes, kAcoMpiAlignment, &padded_row_bytes)) {
    return 0;
  }
  size_t stride = padded_row_bytes / sizeof(float);
  if (stride > (size_t)INT32_MAX) {
    return 0;
  }
  s->stride = (int)stride;
  s->visited_words = (n / 64) + 1;
  s->meta_words = (s->visited_words / 64) + 1;
  size_t total_elems = 0;
  size_t cand_bytes = 0;
  size_t eta_bytes = 0;
  if (!matrix_mul_size((size_t)(n + 1), (size_t)s->stride, &total_elems) ||
      !matrix_mul_size(total_elems, sizeof(int), &cand_bytes) ||
      !matrix_mul_size(total_elems, sizeof(float), &eta_bytes)) {
    return 0;
  }
  s->cand_idx = aco_mpi_aligned_calloc_64(cand_bytes);
  s->eta_beta = aco_mpi_aligned_calloc_64(eta_bytes);
  if (!s->cand_idx || !s->eta_beta) {
    aco_mpi_shared_free(s);
    return 0;
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i <= n; i++) {
    int nodes[kAcoMpiMaxCandidates];
    float dists[kAcoMpiMaxCandidates];
    for (int t = 0; t < s->cand_k; t++) {
      nodes[t] = -1;
      dists[t] = FLT_MAX;
    }

    const float *row = c_mat->rows[i];
    for (int node = 1; node <= n; node++) {
      if (node == i) {
        continue;
      }
      float d = row[node];
      int pos = -1;
      for (int t = 0; t < s->cand_k; t++) {
        if (d < dists[t]) {
          pos = t;
          break;
        }
      }
      if (pos >= 0) {
        for (int m = s->cand_k - 1; m > pos; m--) {
          dists[m] = dists[m - 1];
          nodes[m] = nodes[m - 1];
        }
        dists[pos] = d;
        nodes[pos] = node;
      }
    }

    int *c_row = s->cand_idx + (size_t)i * (size_t)s->stride;
    float *e_row = s->eta_beta + (size_t)i * (size_t)s->stride;
    for (int t = 0; t < s->cand_k; t++) {
      c_row[t] = (nodes[t] > 0) ? nodes[t] : 0;
      e_row[t] = (nodes[t] > 0)
                     ? aco_mpi_fast_powf(1.0f / (dists[t] + 1e-7f),
                                         (float)beta)
                     : 0.0f;
    }
  }
  return 1;
}
