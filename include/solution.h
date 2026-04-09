#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Struct:  Route
 * --------------
 * stores a single vehicle route as a dynamic sequence of node ids.
 */
typedef struct {
  int *nodes;
  int len;
  int cap;
} Route;

/*
 * Struct:  Solution
 * -----------------
 * stores a complete VRP solution with K routes and backing contiguous storage.
 */
typedef struct {
  Route *routes;
  int K;
  int route_cap;
  int *nodes_storage;
} Solution;

/*
 * Function:  solution_create
 * --------------------------
 * allocates and initializes a solution with K routes.
 */
Solution *solution_create(int K, int n);

/*
 * Function:  solution_free
 * ------------------------
 * releases all memory associated with a solution.
 */
void solution_free(Solution *s);

/*
 * Function:  solution_reset
 * -------------------------
 * clears all routes in a solution.
 */
void solution_reset(Solution *s);

/*
 * Function:  solution_copy
 * ------------------------
 * copies route contents from src to dst.
 */
void solution_copy(Solution *dst, const Solution *src);

/*
 * Function:  solution_cost
 * ------------------------
 * computes total traversal cost of all arcs.
 */
double solution_cost(const Solution *s, double **c);

/*
 * Function:  route_append
 * -----------------------
 * appends one node to a route if capacity is available.
 */
void route_append(Route *r, int node);

/*
 * Function:  solution_validate
 * ----------------------------
 * validates VRP structural constraints.
 */
bool solution_validate(const Solution *s, int n, int K, char *err, size_t err_len);

#endif
