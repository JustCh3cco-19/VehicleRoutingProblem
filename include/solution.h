#ifndef SOLUTION_H
#define SOLUTION_H

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief A single vehicle route represented as a dynamic list of node ids.
 */
typedef struct {
  int *nodes;
  int len;
  int cap;
} Route;

/**
 * @brief A complete VRP solution containing K routes.
 */
typedef struct {
  Route *routes;
  int K;
  int route_cap;
  int *nodes_storage;
} Solution;

/**
 * @brief Allocates and initializes a solution with K routes.
 * @param K Number of vehicles.
 * @param n Number of customers.
 * @return Allocated solution handle or NULL on failure.
 */
Solution *solution_create(int K, int n);

/**
 * @brief Releases all memory associated with a solution.
 * @param s Solution handle.
 */
void solution_free(Solution *s);

/**
 * @brief Clears all routes in a solution.
 * @param s Solution handle.
 */
void solution_reset(Solution *s);

/**
 * @brief Copies route content from source to destination.
 * @param dst Destination solution.
 * @param src Source solution.
 */
void solution_copy(Solution *dst, const Solution *src);

/**
 * @brief Computes the total traversal cost of all arcs in a solution.
 * @param s Solution handle.
 * @param c Distance matrix.
 * @return Total route cost.
 */
double solution_cost(const Solution *s, double **c);

/**
 * @brief Appends a node to a route.
 * @param r Route handle.
 * @param node Node id to append.
 * @return true if the node was appended, false if the route is full.
 */
bool route_append(Route *r, int node);

/**
 * @brief Validates core VRP structural constraints for a solution.
 * @param s Solution handle.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param err Output error message buffer.
 * @param err_len Size of output error buffer.
 * @return true if valid, false otherwise.
 */
bool solution_validate(const Solution *s, int n, int K, char *err, size_t err_len);

bool solution_validate_cvrp(const Solution *s, int n, int K, const int *demands,
                            int vehicle_capacity, char *err, size_t err_len);

#endif
