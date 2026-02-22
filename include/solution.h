#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  int *nodes;
  int len;
  int cap;
} Route;

typedef struct {
  Route *routes;
  int K;
} Solution;

/* Allocate K routes with per-route capacity n + 2 (0 ... customers ... 0). */
Solution *solution_create(int K, int n);
void solution_free(Solution *s);
void solution_reset(Solution *s);
/* dst and src must have the same K and route capacities. */
void solution_copy(Solution *dst, const Solution *src);
double solution_cost(const Solution *s, double **c);
void route_append(Route *r, int node);
bool solution_validate(const Solution *s, int n, int K, char *err, size_t err_len);

#endif
