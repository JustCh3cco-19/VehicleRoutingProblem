#include "solution.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void route_init(Route *r, int cap) {
  r->nodes = malloc((size_t)cap * sizeof(int));
  r->cap = cap;
  r->len = 0;
}

static void route_free(Route *r) {
  free(r->nodes);
  r->nodes = NULL;
  r->cap = 0;
  r->len = 0;
}

void route_append(Route *r, int node) {
  if (r->len >= r->cap) {
    return;
  }
  r->nodes[r->len++] = node;
}

Solution *solution_create(int K, int n) {
  Solution *s = malloc(sizeof(*s));
  if (!s) return NULL;

  s->K = K;
  s->routes = calloc((size_t)K, sizeof(Route));
  if (!s->routes) {
    free(s);
    return NULL;
  }

  int cap = n + 2;
  for (int i = 0; i < K; ++i) {
    route_init(&s->routes[i], cap);
    if (!s->routes[i].nodes) {
      for (int j = 0; j < i; ++j) {
        route_free(&s->routes[j]);
      }
      free(s->routes);
      free(s);
      return NULL;
    }
  }

  return s;
}

void solution_reset(Solution *s) {
  for (int i = 0; i < s->K; ++i) {
    s->routes[i].len = 0;
  }
}

void solution_free(Solution *s) {
  if (!s) return;
  for (int i = 0; i < s->K; ++i) {
    route_free(&s->routes[i]);
  }
  free(s->routes);
  free(s);
}

void solution_copy(Solution *dst, const Solution *src) {
  for (int i = 0; i < src->K; ++i) {
    const Route *r_src = &src->routes[i];
    Route *r_dst = &dst->routes[i];
    r_dst->len = r_src->len;
    memcpy(r_dst->nodes, r_src->nodes, (size_t)r_src->len * sizeof(int));
  }
}

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

static void set_err(char *err, size_t err_len, const char *msg) {
  if (!err || err_len == 0) return;
  snprintf(err, err_len, "%s", msg);
}

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
