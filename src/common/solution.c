#include "solution.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOLUTION_ALIGNMENT 64u

/*
 * Function:  align_up_size
 * ------------------------
 * rounds a byte count up to the next alignment boundary.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}

/*
 * Function:  route_append
 * -----------------------
 * appends one node to a route if there is free capacity.
 *
 *  r: destination route
 *  node: node id to append
 *
 *  returns: nothing; if capacity is exhausted, the route is unchanged
 */
void route_append(Route *r, int node) {
  if (r->len >= r->cap) {
    return;
  }
  r->nodes[r->len++] = node;
}

/*
 * Function:  solution_create
 * --------------------------
 * allocates a Solution with K routes. each route gets capacity n+2 to hold
 * depot start/end and up to all customers. all route node arrays are slices
 * of one aligned contiguous storage buffer.
 *
 *  K: number of routes
 *  n: number of customers
 *
 *  returns: allocated Solution pointer on success
 *           NULL on allocation failure
 */
Solution *solution_create(int K, int n) {
  if (K <= 0 || n < 0) {
    return NULL;
  }

  Solution *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->K = K;
  s->route_cap = n + 2;
  s->routes = calloc((size_t)K, sizeof(Route));
  if (!s->routes) {
    free(s);
    return NULL;
  }

  size_t total_nodes = (size_t)K * (size_t)s->route_cap;
  size_t bytes = total_nodes * sizeof(int);
  size_t aligned_bytes = align_up_size(bytes, SOLUTION_ALIGNMENT);
  s->nodes_storage = aligned_alloc(SOLUTION_ALIGNMENT, aligned_bytes);
  if (!s->nodes_storage) {
    free(s->routes);
    free(s);
    return NULL;
  }
  memset(s->nodes_storage, 0, aligned_bytes);

  for (int i = 0; i < K; ++i) {
    Route *r = &s->routes[i];
    r->nodes = s->nodes_storage + (size_t)i * (size_t)s->route_cap;
    r->cap = s->route_cap;
    r->len = 0;
  }

  return s;
}

/*
 * Function:  solution_reset
 * -------------------------
 * clears all route lengths without reallocating memory.
 *
 *  s: solution to reset
 *
 *  returns: nothing
 */
void solution_reset(Solution *s) {
  for (int i = 0; i < s->K; ++i) {
    s->routes[i].len = 0;
  }
}

/*
 * Function:  solution_free
 * ------------------------
 * frees contiguous route storage, route metadata, and solution header.
 *
 *  s: solution to free; NULL is accepted
 *
 *  returns: nothing
 */
void solution_free(Solution *s) {
  if (!s) return;
  free(s->nodes_storage);
  free(s->routes);
  free(s);
}

/*
 * Function:  solution_copy
 * ------------------------
 * copies route lengths and node sequences from src to dst.
 *
 *  dst: destination solution
 *  src: source solution
 *
 *  returns: nothing
 */
void solution_copy(Solution *dst, const Solution *src) {
  for (int i = 0; i < src->K; ++i) {
    const Route *r_src = &src->routes[i];
    Route *r_dst = &dst->routes[i];
    r_dst->len = r_src->len;
    memcpy(r_dst->nodes, r_src->nodes, (size_t)r_src->len * sizeof(int));
  }
}

/*
 * Function:  solution_cost
 * ------------------------
 * computes sum of arc costs over all consecutive node pairs in all routes.
 *
 *  s: solution to evaluate
 *  c: cost matrix
 *
 *  returns: total route traversal cost
 */
double solution_cost(const Solution *s, double **c) {
  double cost = 0.0;
  for (int i = 0; i < s->K; ++i) {
    const Route *r = &s->routes[i];
    for (int t = 0; t + 1 < r->len; ++t) {
      int u = r->nodes[t];
      int v = r->nodes[t + 1];
      cost += c[u][v];
    }
  }
  return cost;
}

/*
 * Function:  set_err
 * ------------------
 * writes an error message into the output buffer if it is valid.
 *
 *  err: destination buffer
 *  err_len: buffer size in bytes
 *  msg: message to write
 *
 *  returns: nothing
 */
static void set_err(char *err, size_t err_len, const char *msg) {
  if (!err || err_len == 0) return;
  snprintf(err, err_len, "%s", msg);
}

/*
 * Function:  solution_validate
 * ----------------------------
 * checks structural validity of a VRP solution:
 * 1) solution metadata and route pointers are coherent
 * 2) each route starts/ends at depot and never uses depot internally
 * 3) node ids are in range
 * 4) each customer is visited exactly once globally
 *
 *  s: solution to validate
 *  n: number of customers
 *  K: expected route count
 *  err: optional buffer for validation error text
 *  err_len: size of err buffer
 *
 *  returns: true when solution is valid
 *           false when any constraint is violated or on allocation failure
 */
bool solution_validate(const Solution *s, int n, int K, char *err,
                       size_t err_len) {
  if (!s) {
    set_err(err, err_len, "solution is NULL");
    return false;
  }
  if (s->K != K) {
    set_err(err, err_len, "solution K does not match expected K");
    return false;
  }
  if (!s->routes) {
    set_err(err, err_len, "solution routes are NULL");
    return false;
  }
  if (n < 0 || K < 0) {
    set_err(err, err_len, "invalid n or K");
    return false;
  }

  bool *seen = calloc((size_t)(n + 1), sizeof(bool));
  if (!seen) {
    set_err(err, err_len, "allocation failure in solution_validate");
    return false;
  }

  for (int i = 0; i < s->K; ++i) {
    const Route *r = &s->routes[i];
    if (r->len < 2) {
      free(seen);
      set_err(err, err_len, "route length < 2");
      return false;
    }
    if (r->nodes[0] != 0 || r->nodes[r->len - 1] != 0) {
      free(seen);
      set_err(err, err_len, "route must start/end at depot 0");
      return false;
    }

    for (int t = 0; t < r->len; ++t) {
      int node = r->nodes[t];
      if (node < 0 || node > n) {
        free(seen);
        set_err(err, err_len, "route contains out-of-range node");
        return false;
      }
      if (node == 0) {
        if (t != 0 && t != r->len - 1) {
          free(seen);
          set_err(err, err_len, "route contains depot inside the path");
          return false;
        }
        continue;
      }
      if (seen[node]) {
        free(seen);
        set_err(err, err_len, "customer visited more than once");
        return false;
      }
      seen[node] = true;
    }
  }

  for (int node = 1; node <= n; ++node) {
    if (!seen[node]) {
      free(seen);
      set_err(err, err_len, "customer not visited");
      return false;
    }
  }

  free(seen);
  set_err(err, err_len, "");
  return true;
}
