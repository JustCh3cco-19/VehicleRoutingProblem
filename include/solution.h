#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdbool.h>
#include <stddef.h>

/*
 * Struct:  Route
 * --------------
 * stores a single vehicle route as a dynamic sequence of node ids.
 *
 *  nodes: contiguous buffer containing route node ids
 *  len: current number of valid nodes in nodes
 *  cap: maximum number of nodes storable in nodes
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
 *
 *  routes: array of K Route entries
 *  K: number of routes/vehicles
 *  route_cap: per-route capacity (same for each route)
 *  nodes_storage: flat backing storage used by all route node buffers
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
 * allocates and initializes a solution with K routes. each route capacity is
 * set to n+2, enough for depot start/end plus all customers. route node data
 * is stored in one contiguous flat buffer for cache-friendly traversal.
 *
 *  K: number of routes/vehicles
 *  n: number of customers
 *
 *  returns: a valid Solution pointer on success
 *           NULL on allocation error
 */
Solution *solution_create(int K, int n);

/*
 * Function:  solution_free
 * ------------------------
 * releases all memory associated with a solution.
 *
 *  s: solution to release; NULL is allowed
 *
 *  returns: nothing
 */
void solution_free(Solution *s);

/*
 * Function:  solution_reset
 * -------------------------
 * clears all routes in a solution by resetting route lengths to zero.
 *
 *  s: solution to reset
 *
 *  returns: nothing
 */
void solution_reset(Solution *s);

/*
 * Function:  solution_copy
 * ------------------------
 * copies route contents from src to dst. both solutions must be compatible
 * (same K and route capacities).
 *
 *  dst: destination solution
 *  src: source solution
 *
 *  returns: nothing
 */
void solution_copy(Solution *dst, const Solution *src);

/*
 * Function:  solution_cost
 * ------------------------
 * computes total traversal cost of all arcs in all routes of a solution.
 *
 *  s: solution to evaluate
 *  c: cost matrix
 *
 *  returns: total route cost
 */
double solution_cost(const Solution *s, double **c);

/*
 * Function:  route_append
 * -----------------------
 * appends one node to a route if capacity is available.
 *
 *  r: target route
 *  node: node id to append
 *
 *  returns: nothing; if the route is full, the function leaves it unchanged
 */
void route_append(Route *r, int node);

/*
 * Function:  solution_validate
 * ----------------------------
 * validates VRP structural constraints:
 * route start/end at depot, no depot in the middle, all customers visited
 * exactly once, and node ids in range.
 *
 *  s: solution to validate
 *  n: number of customers
 *  K: expected number of routes
 *  err: optional output buffer for an error message
 *  err_len: length of err buffer
 *
 *  returns: true if the solution is valid
 *           false otherwise (err is populated when provided)
 */
bool solution_validate(const Solution *s, int n, int K, char *err, size_t err_len);

#endif
