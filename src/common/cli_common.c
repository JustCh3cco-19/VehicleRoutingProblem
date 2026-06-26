#include "cli_common.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int cli_parse_int_arg(const char *s, int *out) {
  char *end = NULL;
  long v;

  if (!s || !out) {
    return 0;
  }

  errno = 0;
  v = strtol(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0') {
    return 0;
  }
  if (v < 0 || v > 100000000L) {
    return 0;
  }
  *out = (int)v;
  return 1;
}

unsigned int cli_parse_uint_arg(const char *s, int *ok) {
  char *end = NULL;
  unsigned long v;

  if (!s || !ok) {
    if (ok) {
      *ok = 0;
    }
    return 0u;
  }

  errno = 0;
  v = strtoul(s, &end, 10);
  if (errno != 0 || end == s || *end != '\0' || v > 0xFFFFFFFFUL) {
    *ok = 0;
    return 0u;
  }
  *ok = 1;
  return (unsigned int)v;
}

void cli_print_solution_routes(const Solution *best, int K) {
  for (int i = 0; i < K; ++i) {
    const Route *r = &best->routes[i];
    int printed = 0;
    printf("Route %d:", i + 1);
    for (int t = 0; t < r->len; ++t) {
      int node = r->nodes[t];
      if (node != 0) {
        printf("%s%d", printed ? " " : " ", node);
        printed = 1;
      }
    }
    printf("\n");
  }
}

void cli_print_solution_cost(double best_cost) {
  printf("Cost: %.3f\n", best_cost);
  printf("best cost: %.3f\n", best_cost);
}

int cli_validate_solution_or_report(const Solution *best, int n, int K,
                                    double best_cost) {
  char err[160];

  if (!isfinite(best_cost) || best_cost <= 0.0) {
    fprintf(stderr, "invalid solution cost: %.12g\n", best_cost);
    return 0;
  }

  if (!solution_validate(best, n, K, err, sizeof(err))) {
    fprintf(stderr, "invalid solution: %s\n", err);
    return 0;
  }

  return 1;
}
