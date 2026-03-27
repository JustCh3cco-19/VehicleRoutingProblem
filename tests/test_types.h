#ifndef TEST_TYPES_H
#define TEST_TYPES_H

/*
 * Struct:  Scenario
 * -----------------
 * describes one benchmark/test configuration used by comparison drivers.
 *
 *  n: number of customers
 *  K: number of vehicles/routes
 *  m: ants per iteration
 *  T: number of iterations
 *  solver_seed: seed for solver stochastic components
 *  instance_seed: seed for random instance generation
 *  layout_id: geometry/layout selector for point generation
 */
typedef struct {
  int n;
  int K;
  int m;
  int T;
  unsigned int solver_seed;
  unsigned int instance_seed;
  int layout_id;
} Scenario;

/*
 * Struct:  Point
 * --------------
 * Cartesian coordinate for generated or loaded test instances.
 *
 *  x: horizontal coordinate
 *  y: vertical coordinate
 */
typedef struct {
  double x;
  double y;
} Point;

#endif
