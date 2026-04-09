#include "local_search.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ACO_ALIGNMENT 64u
#define ACO_MAX_CANDIDATES 64

static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}

static void *aligned_calloc_bytes(size_t bytes) {
  size_t alloc_bytes = align_up_size(bytes, ACO_ALIGNMENT);
  void *ptr = aligned_alloc(ACO_ALIGNMENT, alloc_bytes);
  if (!ptr) {
    return NULL;
  }
  memset(ptr, 0, alloc_bytes);
  return ptr;
}

static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

static double fast_pow_nonneg(double base, double exponent) {
  if (exponent == 1.0) {
    return base;
  }
  if (exponent == 2.0) {
    return base * base;
  }
  if (exponent == 0.5) {
    return sqrt(base);
  }
  return pow(base, exponent);
}

static int aligned_row_stride(int cols, size_t elem_size) {
  size_t row_bytes = (size_t)cols * elem_size;
  size_t padded = align_up_size(row_bytes, ACO_ALIGNMENT);
  return (int)(padded / elem_size);
}

static int choose_candidate_count(int n) {
  if (n <= 8) {
    return n;
  }
  if (n <= 256) {
    return 16;
  }
  if (n <= 4096) {
    return 24;
  }
  return 32;
}

static void seq_shared_build_candidates_common(SeqShared *shared, double **c,
                                               double beta) {
  int n = shared->n;
  int k = shared->candidate_k;
  int stride = shared->stride;

  for (int i = 0; i <= n; ++i) {
    int best_nodes[ACO_MAX_CANDIDATES];
    double best_dists[ACO_MAX_CANDIDATES];

    for (int t = 0; t < k; ++t) {
      best_nodes[t] = 0;
      best_dists[t] = DBL_MAX;
    }

    for (int node = 1; node <= n; ++node) {
      int pos = -1;
      double d;
      if (node == i) {
        continue;
      }

      d = c[i][node];
      for (int t = 0; t < k; ++t) {
        if (d < best_dists[t]) {
          pos = t;
          break;
        }
      }

      if (pos < 0) {
        continue;
      }

      for (int s = k - 1; s > pos; --s) {
        best_dists[s] = best_dists[s - 1];
        best_nodes[s] = best_nodes[s - 1];
      }
      best_dists[pos] = d;
      best_nodes[pos] = node;
    }

    {
      int *cand_row = shared->candidate_idx + (size_t)i * (size_t)stride;
      float *eta_row = shared->eta_beta + (size_t)i * (size_t)stride;
      for (int t = 0; t < k; ++t) {
        int node = best_nodes[t];
        cand_row[t] = node;
        if (node > 0) {
          double eta = 1.0 / (c[i][node] + ACO_EPS);
          eta_row[t] = (float)fast_pow_nonneg(eta, beta);
        } else {
          eta_row[t] = 0.0f;
        }
      }

      for (int t = k; t < stride; ++t) {
        cand_row[t] = 0;
        eta_row[t] = 0.0f;
      }
    }
  }
}

int local_search_shared_init(SeqShared *shared, int n, double **c, double beta) {
  int rows;
  size_t elems;

  if (!shared || n < 1 || !c) {
    return 0;
  }

  memset(shared, 0, sizeof(*shared));
  shared->n = n;
  shared->candidate_k = choose_candidate_count(n);
  shared->candidate_k = clamp_int(shared->candidate_k, 1, ACO_MAX_CANDIDATES);
  if (shared->candidate_k > n) {
    shared->candidate_k = n;
  }
  shared->stride = aligned_row_stride(shared->candidate_k, sizeof(float));
  if (shared->stride < shared->candidate_k) {
    shared->stride = shared->candidate_k;
  }
  shared->visited_words = (n / 64) + 1;

  rows = n + 1;
  elems = (size_t)rows * (size_t)shared->stride;

  shared->candidate_idx = aligned_calloc_bytes(elems * sizeof(int));
  shared->eta_beta = aligned_calloc_bytes(elems * sizeof(float));
  shared->score = NULL;
  shared->ls_pos = (int *)malloc((size_t)(n + 1) * sizeof(int));
  shared->ls_node_route = (int *)malloc((size_t)(n + 1) * sizeof(int));
  shared->ls_node_pos = (int *)malloc((size_t)(n + 1) * sizeof(int));
  if (!shared->candidate_idx || !shared->eta_beta || !shared->ls_pos ||
      !shared->ls_node_route || !shared->ls_node_pos) {
    local_search_shared_free(shared);
    return 0;
  }

  seq_shared_build_candidates_common(shared, c, beta);
  return 1;
}

void local_search_shared_free(SeqShared *shared) {
  if (!shared) {
    return;
  }
  free(shared->ls_node_pos);
  free(shared->ls_node_route);
  free(shared->ls_pos);
  free(shared->score);
  free(shared->eta_beta);
  free(shared->candidate_idx);
  memset(shared, 0, sizeof(*shared));
}

static int candidate_contains(const SeqShared *shared, int u, int v) {
  if (u <= 0 || v <= 0 || u > shared->n || v > shared->n) {
    return 0;
  }

  {
    const int *row =
        shared->candidate_idx + (size_t)u * (size_t)shared->stride;
    for (int t = 0; t < shared->candidate_k; ++t) {
      if (row[t] == v) {
        return 1;
      }
    }
  }
  return 0;
}

static int candidate_pair_allowed(const SeqShared *shared, int a, int b) {
  if (a == 0 || b == 0) {
    return 1;
  }
  if (candidate_contains(shared, a, b)) {
    return 1;
  }
  if (candidate_contains(shared, b, a)) {
    return 1;
  }
  return 0;
}

static void route_reverse_segment(Route *r, int i, int k) {
  while (i < k) {
    int tmp = r->nodes[i];
    r->nodes[i] = r->nodes[k];
    r->nodes[k] = tmp;
    ++i;
    --k;
  }
}

static void route_remove_at(Route *r, int pos) {
  if (pos < 0 || pos >= r->len) {
    return;
  }
  for (int i = pos; i + 1 < r->len; ++i) {
    r->nodes[i] = r->nodes[i + 1];
  }
  --r->len;
}

static void route_insert_at(Route *r, int pos, int node) {
  if (pos < 0 || pos > r->len || r->len >= r->cap) {
    return;
  }
  for (int i = r->len; i > pos; --i) {
    r->nodes[i] = r->nodes[i - 1];
  }
  r->nodes[pos] = node;
  ++r->len;
}

static int route_customer_count(const Route *r) {
  return (r->len >= 2) ? (r->len - 2) : 0;
}

static void local_search_two_opt_intra(Solution *sol, int K, double **c,
                                       const SeqShared *shared) {
  int n = shared->n;
  int *pos = shared->ls_pos;
  if (!pos) {
    return;
  }

  for (int ri = 0; ri < K; ++ri) {
    Route *r = &sol->routes[ri];
    if (r->len < 5) {
      continue;
    }

    for (int node = 0; node <= n; ++node) {
      pos[node] = -1;
    }
    for (int t = 1; t + 1 < r->len; ++t) {
      int node = r->nodes[t];
      if (node > 0 && node <= n) {
        pos[node] = t;
      }
    }

    {
      int improved = 1;
      int pass = 0;
      while (improved && pass < 2) {
        improved = 0;
        ++pass;

        for (int i = 1; i + 2 < r->len; ++i) {
          int a = r->nodes[i - 1];
          int b = r->nodes[i];
          if (b <= 0) {
            continue;
          }

          {
            const int *cand_row =
                shared->candidate_idx + (size_t)b * (size_t)shared->stride;
            for (int ct = 0; ct < shared->candidate_k; ++ct) {
              int cc = cand_row[ct];
              int kpos;
              int d;
              double delta;
              if (cc <= 0 || cc > n) {
                continue;
              }

              kpos = pos[cc];
              if (kpos <= i || kpos + 1 >= r->len) {
                continue;
              }

              d = r->nodes[kpos + 1];
              if (!candidate_pair_allowed(shared, a, cc) ||
                  !candidate_pair_allowed(shared, b, d)) {
                continue;
              }

              delta = c[a][cc] + c[b][d] - c[a][b] - c[cc][d];
              if (delta < -ACO_EPS) {
                route_reverse_segment(r, i, kpos);
                for (int p = i; p <= kpos; ++p) {
                  int node = r->nodes[p];
                  if (node > 0 && node <= n) {
                    pos[node] = p;
                  }
                }
                improved = 1;
              }
            }
          }
        }
      }
    }
  }
}

static void local_search_relocate(Solution *sol, int K,
                                  int vehicle_capacity_customers, double **c,
                                  const SeqShared *shared) {
  int n = shared->n;
  int *node_route = shared->ls_node_route;
  int *node_pos = shared->ls_node_pos;
  if (!node_route || !node_pos) {
    return;
  }

  {
    int move_budget = 48;

    for (int moves = 0; moves < move_budget; ++moves) {
      double best_delta = -ACO_EPS;
      int best_from_r = -1;
      int best_from_pos = -1;
      int best_to_r = -1;
      int best_to_pos = -1;

      for (int node = 0; node <= n; ++node) {
        node_route[node] = -1;
        node_pos[node] = -1;
      }

      for (int ri = 0; ri < K; ++ri) {
        Route *r = &sol->routes[ri];
        for (int pos = 1; pos + 1 < r->len; ++pos) {
          int node = r->nodes[pos];
          if (node > 0 && node <= n) {
            node_route[node] = ri;
            node_pos[node] = pos;
          }
        }
      }

      for (int ri = 0; ri < K; ++ri) {
        Route *from = &sol->routes[ri];
        if (from->len <= 2) {
          continue;
        }

        for (int pos = 1; pos + 1 < from->len; ++pos) {
          int x = from->nodes[pos];
          int prev = from->nodes[pos - 1];
          int next = from->nodes[pos + 1];
          double remove_delta = c[prev][next] - c[prev][x] - c[x][next];
          const int *cand_row =
              shared->candidate_idx + (size_t)x * (size_t)shared->stride;

          for (int ct = 0; ct < shared->candidate_k; ++ct) {
            int y = cand_row[ct];
            int rj;
            int y_pos;
            Route *to;
            int ins_candidates[2];
            if (y <= 0 || y > n) {
              continue;
            }

            rj = node_route[y];
            y_pos = node_pos[y];
            if (rj < 0 || y_pos < 1) {
              continue;
            }

            to = &sol->routes[rj];
            if (ri != rj && vehicle_capacity_customers > 0 &&
                route_customer_count(to) >= vehicle_capacity_customers) {
              continue;
            }

            ins_candidates[0] = y_pos;
            ins_candidates[1] = y_pos + 1;
            for (int q = 0; q < 2; ++q) {
              int ins = ins_candidates[q];
              int u;
              int v;
              double insert_delta;
              double delta;

              if (ins <= 0 || ins >= to->len) {
                continue;
              }
              if (ri == rj && (ins == pos || ins == pos + 1)) {
                continue;
              }

              u = to->nodes[ins - 1];
              v = to->nodes[ins];
              if (!candidate_pair_allowed(shared, u, x) ||
                  !candidate_pair_allowed(shared, x, v)) {
                continue;
              }

              insert_delta = c[u][x] + c[x][v] - c[u][v];
              delta = remove_delta + insert_delta;
              if (delta < best_delta) {
                best_delta = delta;
                best_from_r = ri;
                best_from_pos = pos;
                best_to_r = rj;
                best_to_pos = ins;
              }
            }
          }
        }
      }

      if (best_from_r < 0) {
        break;
      }

      {
        Route *from = &sol->routes[best_from_r];
        Route *to = &sol->routes[best_to_r];
        int node = from->nodes[best_from_pos];
        route_remove_at(from, best_from_pos);

        if (best_from_r == best_to_r && best_to_pos > best_from_pos) {
          --best_to_pos;
        }

        route_insert_at(to, best_to_pos, node);
      }
    }
  }
}

void local_search_refine_solution_common(Solution *sol, int K,
                                         int vehicle_capacity_customers,
                                         double **c,
                                         const SeqShared *shared) {
  local_search_two_opt_intra(sol, K, c, shared);
  local_search_relocate(sol, K, vehicle_capacity_customers, c, shared);
  local_search_two_opt_intra(sol, K, c, shared);
}
