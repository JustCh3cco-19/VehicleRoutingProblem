#include "aco_common.h"

#include <float.h>
#include <math.h>
#include <string.h>

/* Small, fast RNG for deterministic and backend-independent sampling. */
static unsigned int xorshift32(unsigned int *state) {
  unsigned int x = *state;
  if (x == 0u) {
    x = 0x9e3779b9u;
  }
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static double rand01(unsigned int *state) {
  return (double)(xorshift32(state) & 0x00FFFFFFu) / 16777216.0;
}

unsigned int aco_ant_seed(unsigned int seed, int iter, int ant) {
  return seed ^ (0x9e3779b9u * (unsigned int)(iter + 1)) ^
         (0x85ebca6bu * (unsigned int)(ant + 1));
}

int aco_better_candidate(double cost_a, int ant_a, double cost_b, int ant_b) {
  if (cost_a < cost_b) {
    return 1;
  }
  if (fabs(cost_a - cost_b) <= 1e-12 && ant_a < ant_b) {
    return 1;
  }
  return 0;
}

void aco_init_eta_tau(int n, double **c, double **eta, double **tau,
                      double tau0) {
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i == j) {
        eta[i][j] = 0.0;
        tau[i][j] = 0.0;
      } else {
        eta[i][j] = 1.0 / (c[i][j] + ACO_EPS);
        tau[i][j] = tau0;
      }
    }
  }
}

static int select_next(int current, const bool *visited, int n,
                       double **tau, double **eta,
                       double alpha, double beta,
                       unsigned int *rng_state) {
  double denom = 0.0;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      denom += score;
    }
  }

  if (denom <= 0.0) {
    /* Defensive fallback: pick first feasible customer. */
    for (int j = 1; j <= n; ++j) {
      if (!visited[j]) {
        return j;
      }
    }
    return 0;
  }

  double r = rand01(rng_state);
  double cumulative = 0.0;
  int last = 1;
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      double score = pow(tau[current][j], alpha) * pow(eta[current][j], beta);
      double p = score / denom;
      cumulative += p;
      last = j;
      if (cumulative >= r) {
        return j;
      }
    }
  }
  return last;
}

static void append_unvisited_to_last_route(Solution *sol, bool *visited,
                                           int n, int K) {
  Route *last = &sol->routes[K - 1];
  /* Remove closing depot, append leftovers, then close again. */
  if (last->len > 0) {
    --last->len;
  }
  for (int j = 1; j <= n; ++j) {
    if (!visited[j]) {
      route_append(last, j);
      visited[j] = true;
    }
  }
  route_append(last, 0);
}

void aco_construct_ant_solution(Solution *sol, bool *visited,
                                int n, int K,
                                double **tau, double **eta,
                                double alpha, double beta,
                                unsigned int seed) {
  /* Per-ant state is caller-owned so this function stays allocation-free. */
  memset(visited, 0, (size_t)(n + 1) * sizeof(bool));
  solution_reset(sol);

  int unvisited_count = n;

  for (int vehicle = 1; vehicle <= K; ++vehicle) {
    Route *r = &sol->routes[vehicle - 1];
    route_append(r, 0);
    int current = 0;

    /* Keep at least one customer for each remaining vehicle. */
    while (unvisited_count > 0 && unvisited_count > (K - vehicle)) {
      int next = select_next(current, visited, n, tau, eta, alpha, beta,
                             &seed);
      route_append(r, next);
      visited[next] = true;
      --unvisited_count;
      current = next;
    }

    route_append(r, 0);
  }

  if (unvisited_count > 0) {
    append_unvisited_to_last_route(sol, visited, n, K);
  }
}

void aco_update_pheromones(double **tau, const Solution *iter_best,
                           double iter_best_cost,
                           int n, int K, double rho, double Q) {
  /* Global evaporation applied to all non-diagonal entries. */
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      if (i != j) {
        tau[i][j] *= (1.0 - rho);
      }
    }
  }

  if (iter_best_cost >= DBL_MAX) {
    return;
  }

  double deposit = Q / iter_best_cost;
  /* Symmetric deposit (undirected/symmetric VRP cost matrix). */
  for (int i = 0; i < K; ++i) {
    const Route *r = &iter_best->routes[i];
    for (int t = 0; t + 1 < r->len; ++t) {
      int u = r->nodes[t];
      int v = r->nodes[t + 1];
      tau[u][v] += deposit;
      tau[v][u] += deposit;
    }
  }
}
