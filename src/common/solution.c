#include "solution.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SOLUTION_ALIGNMENT 64u


/**
 * @brief Executes `align_up_size`.
 * @param value Function parameter.
 * @param alignment Function parameter.
 * @return Function result.
 */
static size_t align_up_size(size_t value, size_t alignment) {
  size_t rem = value % alignment;
  if (rem == 0u) {
    return value;
  }
  return value + (alignment - rem);
}


/**
 * @brief Executes `route_append`.
 * @param r Function parameter.
 * @param node Function parameter.
 */
void route_append(Route *r, int node) {
  if (r->len >= r->cap) {
    return;
  }
  r->nodes[r->len++] = node;
}


/**
 * @brief Executes `solution_create`.
 * @param K Function parameter.
 * @param n Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `solution_reset`.
 * @param s Function parameter.
 */
void solution_reset(Solution *s) {
  for (int i = 0; i < s->K; ++i) {
    s->routes[i].len = 0;
  }
}


/**
 * @brief Executes `solution_free`.
 * @param s Function parameter.
 */
void solution_free(Solution *s) {
  if (!s) return;
  free(s->nodes_storage);
  free(s->routes);
  free(s);
}


/**
 * @brief Executes `solution_copy`.
 * @param dst Function parameter.
 * @param src Function parameter.
 */
void solution_copy(Solution *dst, const Solution *src) {
  for (int i = 0; i < src->K; ++i) {
    const Route *r_src = &src->routes[i];
    Route *r_dst = &dst->routes[i];
    r_dst->len = r_src->len;
    memcpy(r_dst->nodes, r_src->nodes, (size_t)r_src->len * sizeof(int));
  }
}


/**
 * @brief Executes `solution_cost`.
 * @param s Function parameter.
 * @param c Function parameter.
 * @return Function result.
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


/**
 * @brief Executes `set_err`.
 * @param err Function parameter.
 * @param err_len Function parameter.
 * @param msg Function parameter.
 */
static void set_err(char *err, size_t err_len, const char *msg) {
  if (!err || err_len == 0) return;
  snprintf(err, err_len, "%s", msg);
}


/**
 * @brief Executes `solution_validate`.
 * @param s Function parameter.
 * @param n Function parameter.
 * @param K Function parameter.
 * @param err Function parameter.
 * @param err_len Function parameter.
 * @return Function result.
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

bool solution_validate_cvrp(const Solution *s, int n, int K, const int *demands,
                            int vehicle_capacity, char *err, size_t err_len) {
  if (!solution_validate(s, n, K, err, err_len)) {
    return false;
  }
  if (!demands) {
    set_err(err, err_len, "demands are NULL");
    return false;
  }
  if (vehicle_capacity <= 0) {
    set_err(err, err_len, "vehicle capacity must be positive");
    return false;
  }
  if (demands[0] != 0) {
    set_err(err, err_len, "depot demand must be 0");
    return false;
  }

  for (int i = 0; i < s->K; ++i) {
    const Route *r = &s->routes[i];
    int load = 0;
    for (int t = 1; t + 1 < r->len; ++t) {
      int node = r->nodes[t];
      if (demands[node] < 0) {
        set_err(err, err_len, "route contains customer with negative demand");
        return false;
      }
      load += demands[node];
      if (load > vehicle_capacity) {
        set_err(err, err_len, "route exceeds vehicle capacity");
        return false;
      }
    }
  }

  set_err(err, err_len, "");
  return true;
}
