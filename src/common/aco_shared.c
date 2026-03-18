#include "aco.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>

/*
 * Function:  fast_pow_nonneg
 * --------------------------
 * evaluates base^exponent for non-negative base, with fast paths for common
 * exponents used in ACO.
 *
 *  base: non-negative input value
 *  exponent: power exponent
 *
 *  returns: base raised to exponent
 */
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

/*
 * Function:  rand01
 * -----------------
 * returns a pseudo-random value in [0, 1] using the C runtime RNG.
 *
 *  returns: random double in [0, 1]
 */
static double rand01(void) {
  return (double)rand() / (double)RAND_MAX;
}

/*
 * Function:  aco_rand01_state
 * ---------------------------
 * generates random values in [0, 1] from a local xorshift RNG state when
 * provided; otherwise falls back to rand01().
 *
 *  state: optional RNG state pointer
 *
 *  returns: pseudo-random double in [0, 1]
 */
double aco_rand01_state(unsigned int *state) {
  if (!state) {
    return rand01();
  }

  unsigned int x = *state;
  if (x == 0u) {
    x = 1u;
  }

  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;

  return (double)x / (double)UINT_MAX;
}

/*
 * Function:  aco_make_ant_seed
 * ----------------------------
 * deterministically mixes base seed, iteration index, and ant index to build
 * a non-zero per-ant seed.
 *
 *  base_seed: user-provided base seed
 *  iter: iteration index
 *  ant_index: ant index
 *
 *  returns: non-zero seed value
 */
unsigned int aco_make_ant_seed(unsigned int base_seed, int iter,
                               int ant_index) {
  unsigned int x = base_seed ? base_seed : 1u;
  x ^= 0xa511e9b3u;
  x ^= (unsigned int)(iter + 1) * 0x9e3779b1u;
  x ^= (unsigned int)(ant_index + 1) * 0xc2b2ae3du;
  return x ? x : 1u;
}

/*
 * Function:  aco_select_next
 * --------------------------
 * performs roulette-wheel selection among unvisited nodes using scores:
 * score(node) = tau[current][node]^alpha * eta[current][node]^beta.
 *
 *  current: current node id
 *  unvisited_nodes: compact list of currently unvisited customers
 *  unvisited_count: number of valid entries in unvisited_nodes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  roulette_r: random value in [0, 1] for thresholding
 *  candidate_scores: scratch array to store per-candidate scores
 *  selected_index: optional output index in unvisited_nodes
 *
 *  returns: selected customer id on success
 *           0 if unvisited_count is non-positive
 */
int aco_select_next(int current, const int *unvisited_nodes,
                    int unvisited_count, double **tau, double **eta,
                    double alpha, double beta, double roulette_r,
                    double *candidate_scores, int *selected_index) {
  if (unvisited_count <= 0) {
    if (selected_index) {
      *selected_index = -1;
    }
    return 0;
  }

  double *tau_row = tau[current];
  double *eta_row = eta[current];
  double denom = 0.0;

  for (int idx = 0; idx < unvisited_count; ++idx) {
    int node = unvisited_nodes[idx];
    double tau_term = fast_pow_nonneg(tau_row[node], alpha);
    double eta_term = fast_pow_nonneg(eta_row[node], beta);
    double score = tau_term * eta_term;
    candidate_scores[idx] = score;
    denom += score;
  }

  double threshold = roulette_r * denom;
  double cumulative = 0.0;
  int chosen_idx = unvisited_count - 1;
  for (int idx = 0; idx < unvisited_count; ++idx) {
    cumulative += candidate_scores[idx];
    if (cumulative >= threshold) {
      chosen_idx = idx;
      break;
    }
  }

  if (selected_index) {
    *selected_index = chosen_idx;
  }
  return unvisited_nodes[chosen_idx];
}

/*
 * Function:  aco_build_ant_solution
 * ---------------------------------
 * builds one ant solution route-by-route.
 * algorithm outline:
 * 1) initialize unvisited list [1..n] and pre-generate random draws
 * 2) for each route, repeatedly select next customer with aco_select_next
 * 3) remove selected customer from unvisited list with swap-remove
 * 4) after K routes, append any remaining customers to the last route
 *
 *  sol: solution object to fill
 *  n: number of customers
 *  K: number of routes
 *  tau: pheromone matrix
 *  eta: heuristic matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  rng_state: RNG state used to draw random numbers
 *  unvisited_nodes: scratch array of size >= n
 *  candidate_scores: scratch array of size >= n
 *  random_draws: scratch array of size >= n
 *
 *  returns: nothing
 */
void aco_build_ant_solution(Solution *sol, int n, int K, double **tau,
                            double **eta, double alpha, double beta,
                            unsigned int *rng_state, int *unvisited_nodes,
                            double *candidate_scores, double *random_draws) {
  solution_reset(sol);

  for (int i = 0; i < n; ++i) {
    unvisited_nodes[i] = i + 1;
    random_draws[i] = aco_rand01_state(rng_state);
  }

  int unvisited_count = n;
  int draw_index = 0;

  for (int vehicle = 1; vehicle <= K; ++vehicle) {
    Route *r = &sol->routes[vehicle - 1];
    route_append(r, 0);
    int current = 0;

    while (unvisited_count > 0 && unvisited_count > (K - vehicle)) {
      double roulette_r = random_draws[draw_index++];
      int selected_idx = -1;
      int next = aco_select_next(current, unvisited_nodes, unvisited_count,
                                 tau, eta, alpha, beta, roulette_r,
                                 candidate_scores, &selected_idx);
      if (next <= 0) {
        break;
      }

      route_append(r, next);
      --unvisited_count;
      unvisited_nodes[selected_idx] = unvisited_nodes[unvisited_count];
      current = next;
    }

    route_append(r, 0);
  }

  if (unvisited_count > 0) {
    Route *last = &sol->routes[K - 1];
    if (last->len > 0 && last->nodes[last->len - 1] == 0) {
      --last->len;
    }

    while (unvisited_count > 0) {
      route_append(last, unvisited_nodes[--unvisited_count]);
    }

    route_append(last, 0);
  }
}
