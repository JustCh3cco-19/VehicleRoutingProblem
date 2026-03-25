#include "aco.h"

#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

/*
 * Function:  choose_l1_lines
 * --------------------------
 * picks L1 cache line count for desirability-row cache.
 */
static int choose_l1_lines(int n) {
  int side = n + 1;
  return (side < 8) ? side : 8;
}

/*
 * Function:  choose_l2_lines
 * --------------------------
 * picks L2 cache line count for desirability-row cache.
 */
static int choose_l2_lines(int n) {
  int side = n + 1;
  return (side < 64) ? side : 64;
}

/*
 * Function:  clamp_int
 * --------------------
 * Clamps an integer value between a lower and an upper bound.
 *
 *  x:  the value to clamp
 *  lo: the lower bound
 *  hi: the upper bound
 *
 *  returns: the clamped value
 */
static int clamp_int(int x, int lo, int hi) {
  if (x < lo) {
    return lo;
  }
  if (x > hi) {
    return hi;
  }
  return x;
}

/*
 * Function:  choose_auto_total_ants
 * ---------------------------------
 * estimates a robust ant count from problem size and total worker count
 * (MPI ranks x OpenMP threads).
 *
 *  n: number of customers
 *  mpi_size: number of MPI ranks
 *
 *  returns: total ants per global iteration
 */
static int choose_auto_total_ants(int n, int mpi_size) {
  int omp_threads = 1;
#ifdef _OPENMP
  omp_threads = omp_get_max_threads();
#endif
  if (omp_threads < 1) {
    omp_threads = 1;
  }

  int workers = mpi_size * omp_threads;
  if (workers < 1) {
    workers = 1;
  }

  int ants_per_worker;
  if (n <= 2000) {
    ants_per_worker = 4;
  } else if (n <= 8000) {
    ants_per_worker = 3;
  } else if (n <= 16000) {
    ants_per_worker = 2;
  } else {
    ants_per_worker = 1;
  }

  int total_ants = workers * ants_per_worker;
  total_ants = clamp_int(total_ants, workers, workers * 16);
  total_ants = clamp_int(total_ants, 8, n > 8 ? n : 8);
  return total_ants;
}

/*
 * Function:  choose_stagnation_level
 * ----------------------------------
 * maps the global stagnation counter to a discrete level:
 * 0 = low, 1 = medium, 2 = high.
 *
 *  stagnation_iters: consecutive iterations without global-best improvement
 *  stagnation_trigger: base trigger used for medium/high transitions
 *
 *  returns: stagnation level in [0, 2]
 */
static int choose_stagnation_level(int stagnation_iters,
                                   int stagnation_trigger) {
  if (stagnation_iters >= 2 * stagnation_trigger) {
    return 2;
  }
  if (stagnation_iters >= stagnation_trigger) {
    return 1;
  }
  return 0;
}

/*
 * Function:  choose_ant_batch_size
 * --------------------------------
 * picks a batch size for adaptive in-iteration ant evaluation. Higher
 * stagnation uses smaller batches to check marginal gains more frequently.
 *
 *  local_ants: ants assigned to this rank for one iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: ants processed per batch (>= 1)
 */
static int choose_ant_batch_size(int local_ants, int stagnation_level) {
  if (local_ants <= 0) {
    return 0;
  }

  int divisor = 8;
  if (stagnation_level <= 0) {
    divisor = 6;
  } else if (stagnation_level >= 2) {
    divisor = 12;
  }

  int batch = local_ants / divisor;
  return clamp_int(batch, 1, local_ants);
}

/*
 * Function:  choose_min_ants_before_stop
 * --------------------------------------
 * picks the minimum ants to evaluate before allowing early stop. Higher
 * stagnation enforces deeper per-iteration search.
 *
 *  local_ants: ants assigned to this rank for one iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: minimum ants to evaluate (>= 1, <= local_ants)
 */
static int choose_min_ants_before_stop(int local_ants, int stagnation_level) {
  if (local_ants <= 0) {
    return 0;
  }

  int min_ants = local_ants / 4;
  if (stagnation_level == 1) {
    min_ants = local_ants / 2;
  } else if (stagnation_level >= 2) {
    min_ants = (3 * local_ants) / 4;
  }
  return clamp_int(min_ants, 1, local_ants);
}

/*
 * Function:  choose_no_improve_patience
 * -------------------------------------
 * picks how many consecutive non-improving batches are tolerated before
 * stopping the current iteration early. Higher stagnation increases patience.
 *
 *  local_ants: ants assigned to this rank for one iteration
 *  stagnation_level: 0 (low), 1 (medium), 2 (high)
 *
 *  returns: patience in number of batches (>= 1)
 */
static int choose_no_improve_patience(int local_ants, int stagnation_level) {
  if (local_ants <= 0) {
    return 1;
  }

  int patience = (local_ants >= 128) ? 3 : 2;
  if (stagnation_level == 1) {
    ++patience;
  } else if (stagnation_level >= 2) {
    patience += 2;
  }
  return clamp_int(patience, 1, 6);
}

/*
 * Function:  route_reverse_segment
 * --------------------------------
 * reverses an in-place route segment between inclusive endpoints.
 *
 *  r: route to modify
 *  i: segment start index
 *  k: segment end index
 *
 *  returns: nothing
 */
static void route_reverse_segment(Route *r, int i, int k) {
  while (i < k) {
    int tmp = r->nodes[i];
    r->nodes[i] = r->nodes[k];
    r->nodes[k] = tmp;
    ++i;
    --k;
  }
}

/*
 * Function:  route_remove_at
 * --------------------------
 * removes one node at a valid position by shifting tail elements left.
 *
 *  r: route to modify
 *  pos: position to remove
 *
 *  returns: nothing; invalid positions are ignored
 */
static void route_remove_at(Route *r, int pos) {
  if (pos < 0 || pos >= r->len) {
    return;
  }
  for (int i = pos; i + 1 < r->len; ++i) {
    r->nodes[i] = r->nodes[i + 1];
  }
  --r->len;
}

/*
 * Function:  route_insert_at
 * --------------------------
 * inserts a node at a valid position by shifting tail elements right.
 *
 *  r: route to modify
 *  pos: insertion index
 *  node: node id to insert
 *
 *  returns: nothing; invalid positions/full route are ignored
 */
static void route_insert_at(Route *r, int pos, int node) {
  if (pos < 0 || pos > r->len || r->len >= r->cap) {
    return;
  }
  for (int i = r->len; i > pos; --i) {
    r->nodes[i] = r->nodes[i - 1];
  }
  r->nodes[pos] = node;
  ++r->len;
}

/*
 * Function:  route_customer_count
 * -------------------------------
 * counts customers in a route excluding start/end depot nodes.
 *
 *  r: route to inspect
 *
 *  returns: number of customer nodes in the route
 */
static int route_customer_count(const Route *r) { return (r->len >= 2) ? (r->len - 2) : 0; }

/*
 * Function:  local_search_two_opt_intra
 * -------------------------------------
 * applies bounded 2-opt improvement independently within each route.
 *
 *  sol: solution to refine
 *  K: route count
 *  c: cost matrix
 *
 *  returns: nothing
 */
static void local_search_two_opt_intra(Solution *sol, int K, double **c) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (K > 4)
#endif
  for (int ri = 0; ri < K; ++ri) {
    Route *r = &sol->routes[ri];
    if (r->len < 5) {
      continue;
    }

    int improved = 1;
    int pass = 0;
    while (improved && pass < 3) {
      improved = 0;
      ++pass;

      for (int i = 1; i + 2 < r->len; ++i) {
        for (int k = i + 1; k + 1 < r->len; ++k) {
          int a = r->nodes[i - 1];
          int b = r->nodes[i];
          int cc = r->nodes[k];
          int d = r->nodes[k + 1];
          double delta = c[a][cc] + c[b][d] - c[a][b] - c[cc][d];
          if (delta < -ACO_EPS) {
            route_reverse_segment(r, i, k);
            improved = 1;
          }
        }
      }
    }
  }
}

/*
 * Function:  local_search_relocate
 * --------------------------------
 * performs best-improving single-customer relocations across routes under
 * simple per-vehicle customer-capacity constraints.
 *
 *  sol: solution to refine
 *  K: route count
 *  vehicle_capacity_customers: max customers allowed in a route when > 0
 *  c: cost matrix
 *
 *  returns: nothing
 */
static void local_search_relocate(Solution *sol, int K, int vehicle_capacity_customers,
                                  double **c) {
  int move_budget = 64;
  for (int moves = 0; moves < move_budget; ++moves) {
    double best_delta = -ACO_EPS;
    int best_from_r = -1;
    int best_from_pos = -1;
    int best_to_r = -1;
    int best_to_pos = -1;

    for (int ri = 0; ri < K; ++ri) {
      Route *from = &sol->routes[ri];
      if (from->len <= 2) {
        continue;
      }

      for (int pos = 1; pos + 1 < from->len; ++pos) {
        int x = from->nodes[pos];
        int prev = from->nodes[pos - 1];
        int next = from->nodes[pos + 1];
        double remove_delta = c[prev][next] - c[prev][x] - c[x][next];

        for (int rj = 0; rj < K; ++rj) {
          Route *to = &sol->routes[rj];
          if (ri != rj && vehicle_capacity_customers > 0 &&
              route_customer_count(to) >= vehicle_capacity_customers) {
            continue;
          }

          for (int ins = 1; ins < to->len; ++ins) {
            if (ri == rj && (ins == pos || ins == pos + 1)) {
              continue;
            }

            int u = to->nodes[ins - 1];
            int v = to->nodes[ins];
            double insert_delta = c[u][x] + c[x][v] - c[u][v];
            double delta = remove_delta + insert_delta;

            if (delta < best_delta) {
              best_delta = delta;
              best_from_r = ri;
              best_from_pos = pos;
              best_to_r = rj;
              best_to_pos = ins;
            }
          }
        }
      }
    }

    if (best_from_r < 0) {
      break;
    }

    Route *from = &sol->routes[best_from_r];
    Route *to = &sol->routes[best_to_r];
    int node = from->nodes[best_from_pos];
    route_remove_at(from, best_from_pos);

    if (best_from_r == best_to_r && best_to_pos > best_from_pos) {
      --best_to_pos;
    }

    route_insert_at(to, best_to_pos, node);
  }
}

/*
 * Function:  local_search_refine_solution
 * ---------------------------------------
 * runs a lightweight local-search pipeline: 2-opt, relocate, then 2-opt.
 *
 *  sol: solution to refine
 *  K: route count
 *  vehicle_capacity_customers: max customers per route when > 0
 *  c: cost matrix
 *
 *  returns: nothing
 */
static void local_search_refine_solution(Solution *sol, int K,
                                         int vehicle_capacity_customers,
                                         double **c) {
  local_search_two_opt_intra(sol, K, c);
  local_search_relocate(sol, K, vehicle_capacity_customers, c);
  local_search_two_opt_intra(sol, K, c);
}

#ifdef USE_MPI
/*
 * Function:  solution_pack_size_ints
 * ----------------------------------
 * computes the integer buffer length required to serialize a Solution with K
 * routes and per-route capacity n+2.
 *
 *  K: number of routes
 *  n: number of customers
 *
 *  returns: number of ints needed in packed representation
 */
static int solution_pack_size_ints(int K, int n) {
  return 1 + K + K * (n + 2);
}

/*
 * Function:  solution_pack
 * ------------------------
 * serializes a Solution into an integer buffer for MPI broadcast.
 *
 *  sol: source solution
 *  n: number of customers
 *  buf: destination integer buffer with size solution_pack_size_ints(K, n)
 *
 *  returns: nothing
 */
static void solution_pack(const Solution *sol, int n, int *buf) {
  int route_cap = n + 2;
  buf[0] = sol->K;

  for (int i = 0; i < sol->K; ++i) {
    const Route *src = &sol->routes[i];
    int len = src->len;
    if (len < 0) {
      len = 0;
    }
    if (len > route_cap) {
      len = route_cap;
    }

    buf[1 + i] = len;

    int *dst_nodes = &buf[1 + sol->K + i * route_cap];
    memset(dst_nodes, 0, (size_t)route_cap * sizeof(int));
    if (len > 0) {
      memcpy(dst_nodes, src->nodes, (size_t)len * sizeof(int));
    }
  }
}

/*
 * Function:  solution_unpack
 * --------------------------
 * deserializes a packed solution buffer into an existing Solution object.
 *
 *  sol: destination solution (already allocated with matching K/capacities)
 *  n: number of customers
 *  buf: serialized integer buffer
 *
 *  returns: true on successful unpack
 *           false if packed metadata is inconsistent
 */
static bool solution_unpack(Solution *sol, int n, const int *buf) {
  int route_cap = n + 2;
  if (buf[0] != sol->K) {
    return false;
  }

  for (int i = 0; i < sol->K; ++i) {
    int len = buf[1 + i];
    if (len < 0 || len > route_cap) {
      return false;
    }

    Route *dst = &sol->routes[i];
    dst->len = len;
    if (len > 0) {
      memcpy(dst->nodes, &buf[1 + sol->K + i * route_cap],
             (size_t)len * sizeof(int));
    }
  }

  return true;
}
#endif

/*
 * Function:  aco_vrp
 * ------------------
 * runs the hybrid OpenMP+MPI ACO solver for VRP.
 * algorithm outline:
 * 1) split ants across MPI ranks
 * 2) initialize shared eta/tau matrices
 * 3) for each iteration, evaluate local ants (OpenMP in-rank parallelism)
 *    in adaptive batches with early stop on weak marginal gains
 * 4) reduce local iteration-best to global best with MPI collectives
 * 5) synchronize winning solution across ranks
 * 6) update pheromone matrix from global iteration-best solution
 *
 *  n: number of customers (1..n), with depot at 0
 *  K: number of routes/vehicles
 *  m: total ants per iteration across all ranks
 *  T: number of iterations
 *  c: cost matrix
 *  alpha: pheromone exponent
 *  beta: heuristic exponent
 *  rho: evaporation factor
 *  tau0: initial pheromone value
 *  Q: deposit scale
 *  seed: base random seed
 *  best_solution: output container for best route set
 *  best_cost: output best objective value
 *
 *  returns: nothing; on allocation or synchronization failure prints an error
 *           and returns early
 */
void aco_vrp(int n, int K, int m, int T, double **c, double alpha,
             double beta, double rho, double tau0, double Q,
             unsigned int seed, Solution *best_solution, double *best_cost) {
  int mpi_rank = 0;
  int mpi_size = 1;
  bool mpi_enabled = false;

#ifdef USE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized) {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    mpi_enabled = (mpi_size > 1);
  }
#endif

  int total_m = m;
  if (total_m <= 0) {
    total_m = choose_auto_total_ants(n, mpi_size);
  }

  int local_m = total_m;
  int ant_offset = 0;
  if (mpi_enabled) {
    int base = (mpi_size > 0) ? (total_m / mpi_size) : total_m;
    int rem = (mpi_size > 0) ? (total_m % mpi_size) : 0;
    local_m = base + ((mpi_rank < rem) ? 1 : 0);
    ant_offset = mpi_rank * base + ((mpi_rank < rem) ? mpi_rank : rem);
  }

  double **eta = matrix_alloc(n);
  double **tau = matrix_alloc(n);
  Solution *iter_best = solution_create(K, n);

#ifdef USE_MPI
  int packed_len = mpi_enabled ? solution_pack_size_ints(K, n) : 0;
  int *packed_solution =
      mpi_enabled ? calloc((size_t)packed_len, sizeof(int)) : NULL;
#endif

  int local_ok = (eta && tau && iter_best) ? 1 : 0;
#ifdef USE_MPI
  if (mpi_enabled && !packed_solution) {
    local_ok = 0;
  }
  if (mpi_enabled) {
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    local_ok = global_ok;
  }
#endif

  if (!local_ok) {
    fprintf(stderr, "allocation failure in aco_vrp\n");
    matrix_free(eta);
    matrix_free(tau);
    solution_free(iter_best);
#ifdef USE_MPI
    free(packed_solution);
#endif
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (n > 16)
#endif
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

  solution_reset(best_solution);
  *best_cost = DBL_MAX;

  int vehicle_capacity_customers = (K > 0) ? ((n + K - 1) / K) : n;
  if (vehicle_capacity_customers < 1) {
    vehicle_capacity_customers = 1;
  }

#ifdef _OPENMP
  bool omp_enabled = (omp_get_max_threads() > 1 && local_m > 1);
#else
  bool omp_enabled = false;
#endif

  AcoScoreCache *serial_cache = NULL;
  if (!omp_enabled) {
    serial_cache =
        aco_score_cache_create(n, choose_l1_lines(n), choose_l2_lines(n));
  }

  size_t scratch_len = (n > 0) ? (size_t)n : 1u;
  int stagnation_iters = 0;
  int stagnation_trigger = T / 4;
  if (stagnation_trigger < 4) {
    stagnation_trigger = 4;
  }
  const double iter_deposit_weight = 0.3;
  const double global_deposit_weight = 0.7;
  double tau_max = tau0;
  double tau_min = tau0 * 0.05;

  for (int iter = 0; iter < T; ++iter) {
    int stagnation_level =
        choose_stagnation_level(stagnation_iters, stagnation_trigger);
    int batch_size = choose_ant_batch_size(local_m, stagnation_level);
    int min_ants_before_stop =
        choose_min_ants_before_stop(local_m, stagnation_level);
    int no_improve_patience =
        choose_no_improve_patience(local_m, stagnation_level);

    double iter_best_cost = DBL_MAX;
    int iter_best_ant = INT_MAX;
    int iter_failed = 0;

    if (omp_enabled) {
#ifdef _OPENMP
      int ants_evaluated = 0;
      int consecutive_no_improve_batches = 0;

      for (int batch_start = 0; batch_start < local_m;
           batch_start += batch_size) {
        int batch_end = batch_start + batch_size;
        if (batch_end > local_m) {
          batch_end = local_m;
        }

        double prev_best_cost = iter_best_cost;
        int prev_best_ant = iter_best_ant;
        double batch_best_cost = DBL_MAX;
        int batch_best_ant = INT_MAX;
        Solution *batch_best_solution = solution_create(K, n);

        if (!batch_best_solution) {
          iter_failed = 1;
          break;
        }

#pragma omp parallel default(shared)
        {
          double thread_best_cost = DBL_MAX;
          int thread_best_ant = INT_MAX;
          Solution *sol = solution_create(K, n);
          Solution *thread_best = solution_create(K, n);
          AcoScoreCache *score_cache =
              aco_score_cache_create(n, choose_l1_lines(n), choose_l2_lines(n));
          int *unvisited_nodes = malloc(scratch_len * sizeof(int));
          double *candidate_scores = malloc(scratch_len * sizeof(double));
          double *random_draws = malloc(scratch_len * sizeof(double));

          if (!sol || !thread_best || !score_cache || !unvisited_nodes ||
              !candidate_scores || !random_draws) {
#pragma omp critical
            { iter_failed = 1; }
          } else {
#pragma omp for schedule(static)
            for (int ant = batch_start; ant < batch_end; ++ant) {
              int global_ant = ant_offset + ant;
              unsigned int rng_state =
                  aco_make_ant_seed(seed, iter, global_ant);

              aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                                     vehicle_capacity_customers, score_cache,
                                     &rng_state, unvisited_nodes,
                                     candidate_scores, random_draws);
              double cost = solution_cost(sol, c);

              if (cost < thread_best_cost ||
                  (fabs(cost - thread_best_cost) <= ACO_EPS &&
                   global_ant < thread_best_ant)) {
                thread_best_cost = cost;
                thread_best_ant = global_ant;
                solution_copy(thread_best, sol);
              }
            }

            if (thread_best_cost < DBL_MAX) {
#pragma omp critical
              {
                if (thread_best_cost < batch_best_cost ||
                    (fabs(thread_best_cost - batch_best_cost) <= ACO_EPS &&
                     thread_best_ant < batch_best_ant)) {
                  batch_best_cost = thread_best_cost;
                  batch_best_ant = thread_best_ant;
                  solution_copy(batch_best_solution, thread_best);
                }
              }
            }
          }

          free(random_draws);
          free(candidate_scores);
          free(unvisited_nodes);
          aco_score_cache_free(score_cache);
          solution_free(thread_best);
          solution_free(sol);
        }

        if (iter_failed) {
          solution_free(batch_best_solution);
          break;
        }

        if (batch_best_cost < iter_best_cost ||
            (fabs(batch_best_cost - iter_best_cost) <= ACO_EPS &&
             batch_best_ant < iter_best_ant)) {
          iter_best_cost = batch_best_cost;
          iter_best_ant = batch_best_ant;
          solution_copy(iter_best, batch_best_solution);
        }

        solution_free(batch_best_solution);

        ants_evaluated += (batch_end - batch_start);
        if (iter_best_cost < prev_best_cost - ACO_EPS ||
            (fabs(iter_best_cost - prev_best_cost) <= ACO_EPS &&
             iter_best_ant < prev_best_ant)) {
          consecutive_no_improve_batches = 0;
        } else {
          ++consecutive_no_improve_batches;
        }

        if (ants_evaluated >= min_ants_before_stop &&
            consecutive_no_improve_batches >= no_improve_patience) {
          break;
        }
      }
#endif
    } else {
      aco_score_cache_invalidate(serial_cache);
      Solution *sol = solution_create(K, n);
      int *unvisited_nodes = malloc(scratch_len * sizeof(int));
      double *candidate_scores = malloc(scratch_len * sizeof(double));
      double *random_draws = malloc(scratch_len * sizeof(double));

      if (!sol || !unvisited_nodes || !candidate_scores ||
          !random_draws) {
        iter_failed = 1;
      } else {
        int ants_evaluated = 0;
        int consecutive_no_improve_batches = 0;

        for (int batch_start = 0; batch_start < local_m;
             batch_start += batch_size) {
          int batch_end = batch_start + batch_size;
          if (batch_end > local_m) {
            batch_end = local_m;
          }

          double prev_best_cost = iter_best_cost;
          int prev_best_ant = iter_best_ant;

          for (int ant = batch_start; ant < batch_end; ++ant) {
            int global_ant = ant_offset + ant;
            unsigned int rng_state = aco_make_ant_seed(seed, iter, global_ant);

            aco_build_ant_solution(sol, n, K, tau, eta, alpha, beta,
                                   vehicle_capacity_customers, serial_cache,
                                   &rng_state, unvisited_nodes,
                                   candidate_scores, random_draws);
            double cost = solution_cost(sol, c);

            if (cost < iter_best_cost ||
                (fabs(cost - iter_best_cost) <= ACO_EPS &&
                 global_ant < iter_best_ant)) {
              iter_best_cost = cost;
              iter_best_ant = global_ant;
              solution_copy(iter_best, sol);
            }
          }

          ants_evaluated += (batch_end - batch_start);
          if (iter_best_cost < prev_best_cost - ACO_EPS ||
              (fabs(iter_best_cost - prev_best_cost) <= ACO_EPS &&
               iter_best_ant < prev_best_ant)) {
            consecutive_no_improve_batches = 0;
          } else {
            ++consecutive_no_improve_batches;
          }

          if (ants_evaluated >= min_ants_before_stop &&
              consecutive_no_improve_batches >= no_improve_patience) {
            break;
          }
        }
      }

      free(random_draws);
      free(candidate_scores);
      free(unvisited_nodes);
      solution_free(sol);
    }

#ifdef USE_MPI
    if (mpi_enabled) {
      int global_failed = 0;
      MPI_Allreduce(&iter_failed, &global_failed, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD);
      if (global_failed) {
        fprintf(stderr, "allocation failure during iteration\n");
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
        aco_score_cache_free(serial_cache);
        free(packed_solution);
        return;
      }

      int local_has_solution = (iter_best_cost < DBL_MAX) ? 1 : 0;
      int global_has_solution = 0;
      MPI_Allreduce(&local_has_solution, &global_has_solution, 1, MPI_INT,
                    MPI_MAX, MPI_COMM_WORLD);
      if (!global_has_solution) {
        continue;
      }

      double global_best_cost = DBL_MAX;
      MPI_Allreduce(&iter_best_cost, &global_best_cost, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);

      int local_ant = INT_MAX;
      if (local_has_solution && fabs(iter_best_cost - global_best_cost) <= ACO_EPS) {
        local_ant = iter_best_ant;
      }

      int global_best_ant = INT_MAX;
      MPI_Allreduce(&local_ant, &global_best_ant, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      int owner_rank = mpi_size;
      if (global_best_ant >= ant_offset && global_best_ant < ant_offset + local_m) {
        owner_rank = mpi_rank;
      }

      int winner_rank = mpi_size;
      MPI_Allreduce(&owner_rank, &winner_rank, 1, MPI_INT, MPI_MIN,
                    MPI_COMM_WORLD);

      if (mpi_rank == winner_rank) {
        solution_pack(iter_best, n, packed_solution);
      }

      MPI_Bcast(packed_solution, packed_len, MPI_INT, winner_rank,
                MPI_COMM_WORLD);

      if (mpi_rank != winner_rank &&
          !solution_unpack(iter_best, n, packed_solution)) {
        fprintf(stderr, "solution synchronization failure\n");
        matrix_free(eta);
        matrix_free(tau);
        solution_free(iter_best);
        aco_score_cache_free(serial_cache);
        free(packed_solution);
        return;
      }

      iter_best_cost = global_best_cost;
      iter_best_ant = global_best_ant;
    }
#endif

    if (iter_failed) {
      fprintf(stderr, "allocation failure during iteration\n");
      matrix_free(eta);
      matrix_free(tau);
      solution_free(iter_best);
      aco_score_cache_free(serial_cache);
#ifdef USE_MPI
      free(packed_solution);
#endif
      return;
    }

    if (iter_best_cost < DBL_MAX) {
      local_search_refine_solution(iter_best, K, vehicle_capacity_customers, c);
      iter_best_cost = solution_cost(iter_best, c);
    }

    if (iter_best_cost < *best_cost) {
      *best_cost = iter_best_cost;
      solution_copy(best_solution, iter_best);
      stagnation_iters = 0;

      if (*best_cost > ACO_EPS) {
        tau_max = 1.0 / ((1.0 - rho) * (*best_cost));
        tau_min = tau_max * 0.05;
      }
    } else {
      ++stagnation_iters;
    }

    if (iter_best_cost < DBL_MAX) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (n > 16)
#endif
      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i != j) {
            tau[i][j] *= (1.0 - rho);
          }
        }
      }

      double iter_deposit = (iter_deposit_weight * Q) / iter_best_cost;
      for (int i = 0; i < K; ++i) {
        Route *r = &iter_best->routes[i];
        for (int t = 0; t + 1 < r->len; ++t) {
          int u = r->nodes[t];
          int v = r->nodes[t + 1];
          tau[u][v] += iter_deposit;
          tau[v][u] += iter_deposit;
        }
      }

      if (*best_cost < DBL_MAX) {
        double global_deposit = (global_deposit_weight * Q) / (*best_cost);
        for (int i = 0; i < K; ++i) {
          Route *r = &best_solution->routes[i];
          for (int t = 0; t + 1 < r->len; ++t) {
            int u = r->nodes[t];
            int v = r->nodes[t + 1];
            tau[u][v] += global_deposit;
            tau[v][u] += global_deposit;
          }
        }
      }

      for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
          if (i == j) {
            continue;
          }
          if (tau[i][j] < tau_min) {
            tau[i][j] = tau_min;
          } else if (tau[i][j] > tau_max) {
            tau[i][j] = tau_max;
          }
        }
      }

      if (stagnation_iters >= stagnation_trigger) {
        for (int i = 0; i <= n; ++i) {
          for (int j = 0; j <= n; ++j) {
            if (i != j) {
              tau[i][j] = 0.5 * tau[i][j] + 0.5 * tau0;
            }
          }
        }
        stagnation_iters = 0;
      }
    }
  }

  solution_free(iter_best);
  matrix_free(eta);
  matrix_free(tau);
  aco_score_cache_free(serial_cache);
#ifdef USE_MPI
  free(packed_solution);
#endif
}
