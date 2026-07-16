#include "aco_v2.h"
#include "aco.h"
#include "matrix.h"
#include "solution.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#define V2_ALIGNMENT 64u
#define V2_MAX_CANDS 512
#define V2_EPS 1e-7f

/* -- Matrices in FLOAT (HPC Optimized) -- */
typedef struct {
    int n;
    int stride;
    float *data;
    float **rows;
} MatrixFloat;

static size_t align_up_size(size_t value, size_t alignment) {
    size_t rem = value % alignment;
    return (rem == 0u) ? value : value + (alignment - rem);
}

static size_t align_up_64(size_t value) {
    return align_up_size(value, V2_ALIGNMENT);
}

static MatrixFloat *matrix_create_float(int n) {
    MatrixFloat *m = malloc(sizeof(*m));
    m->n = n;
    size_t row_bytes = (size_t)(n + 1) * sizeof(float);
    m->stride = (int)(align_up_64(row_bytes) / sizeof(float));
    m->data = aligned_alloc(V2_ALIGNMENT, (size_t)(n + 1) * (size_t)m->stride * sizeof(float));
    m->rows = malloc((size_t)(n + 1) * sizeof(float *));
    for (int i = 0; i <= n; i++) m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
    memset(m->data, 0, (size_t)(n + 1) * (size_t)m->stride * sizeof(float));
    return m;
}

static void matrix_free_float(MatrixFloat *m) {
    if (!m) return;
    free(m->data); free(m->rows); free(m);
}

/* -- Helper Statici (Float) -- */

static void *aligned_calloc_64(size_t bytes) {
    size_t alloc_bytes = align_up_64(bytes);
    void *ptr = aligned_alloc(V2_ALIGNMENT, alloc_bytes);
    if (!ptr) return NULL;
    memset(ptr, 0, alloc_bytes);
    return ptr;
}

static double wall_time(void) {
#ifdef USE_MPI
    int init = 0; MPI_Initialized(&init);
    if (init) return MPI_Wtime();
#endif
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static long get_l3_cache_size(void) {
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (!f) return 0;
    char buf[64];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);
    long size = atol(buf);
    char *unit = strpbrk(buf, "KMGTkmgt");
    if (unit) {
        if (*unit == 'K' || *unit == 'k') size *= 1024;
        else if (*unit == 'M' || *unit == 'm') size *= 1024 * 1024;
        else if (*unit == 'G' || *unit == 'g') size *= 1024 * 1024 * 1024;
    }
    return size;
}

static int tune_candidate_k(int n, long l3_size) {
    if (l3_size <= 0) return 32;
    double target_bytes = (double)l3_size * 0.6;
    int k = (int)(target_bytes / ((double)(n + 1) * 4.0));
    if (k < 16) return 16;
    if (k > V2_MAX_CANDS) return V2_MAX_CANDS;
    return k;
}

static float fast_powf(float base, float exp) {
    if (exp == 1.0f) return base;
    if (exp == 2.0f) return base * base;
    if (exp == 0.5f) return sqrtf(base);
    return powf(base, exp);
}

static void load_v2_directives(double *max_runtime_sec, int *max_stagnation_epochs, double *min_rel_improvement) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  const char *s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");
  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_epochs = (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = (s_rel && *s_rel) ? atof(s_rel) : 1e-3;
}

static int is_sig_imp(double prev_best, double new_best, double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) return (new_best < prev_best);
  if (new_best >= prev_best - V2_EPS) return 0;
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, V2_EPS);
  return rel_gain + V2_EPS >= min_rel_improvement;
}

typedef struct {
    int n;
    int cand_k;
    int stride;
    int visited_words;
    int *cand_idx;
    float *eta_beta;
} V2RankShared;

static void v2_shared_free(V2RankShared *s) {
    if (!s) return;
    free(s->eta_beta); free(s->cand_idx);
}

static int v2_shared_init(V2RankShared *s, int n, int cand_k, const MatrixFloat *c_mat, double beta) {
    s->n = n; s->cand_k = cand_k;
    size_t row_bytes = (size_t)s->cand_k * sizeof(float);
    s->stride = (int)(align_up_64(row_bytes) / sizeof(float));
    s->visited_words = (n / 64) + 1;
    size_t total_elems = (size_t)(n + 1) * (size_t)s->stride;
    s->cand_idx = aligned_calloc_64(total_elems * sizeof(int));
    s->eta_beta = aligned_calloc_64(total_elems * sizeof(float));
    if (!s->cand_idx || !s->eta_beta) { v2_shared_free(s); return 0; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) {
        int nodes[V2_MAX_CANDS]; float dists[V2_MAX_CANDS];
        for (int t = 0; t < s->cand_k; t++) { nodes[t] = -1; dists[t] = FLT_MAX; }
        const float *row = c_mat->rows[i];
        for (int node = 1; node <= n; node++) {
            if (node == i) continue;
            float d = row[node]; int pos = -1;
            for (int t = 0; t < s->cand_k; t++) { if (d < dists[t]) { pos = t; break; } }
            if (pos >= 0) {
                for (int m = s->cand_k - 1; m > pos; m--) { dists[m] = dists[m-1]; nodes[m] = nodes[m-1]; }
                dists[pos] = d; nodes[pos] = node;
            }
        }
        int *c_row = s->cand_idx + (size_t)i * (size_t)s->stride;
        float *e_row = s->eta_beta + (size_t)i * (size_t)s->stride;
        for (int t = 0; t < s->cand_k; t++) {
            c_row[t] = (nodes[t] > 0) ? nodes[t] : 0;
            e_row[t] = (nodes[t] > 0) ? fast_powf(1.0f / (dists[t] + 1e-7f), (float)beta) : 0.0f;
        }
    }
    return 1;
}

static void v2_ws_free(AcoThreadWorkspace *ws) {
  if (!ws) return;
  free(ws->route_loads); free(ws->visited);
  solution_free(ws->thread_best); solution_free(ws->sol);
}

static int v2_ws_init(AcoThreadWorkspace *ws, int K, int n, int words) {
  ws->sol = solution_create(K, n); ws->thread_best = solution_create(K, n);
  ws->visited = aligned_calloc_64((size_t)words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->route_loads) { v2_ws_free(ws); return 0; }
  return 1;
}

static int find_nearest_unvisited(const V2RankShared *s, int curr, const uint64_t *visited, const MatrixFloat *c) {
    int best = 0;
    float best_d = FLT_MAX;
    const float *row = c->rows[curr];

    #if defined(__AVX2__)
    // Prova a scansionare a blocchi di 8 con SIMD se il nodo non è visitato
    // Nota: La bitmask è ancora la guida primaria per evitare caricamenti inutili
    #endif

    for (int w = 0; w < s->visited_words; w++) {
        uint64_t v = visited[w];
        if (v == 0xFFFFFFFFFFFFFFFFull) continue;
        uint64_t mask = ~v;
        int base = w << 6;
        if (w == s->visited_words - 1) {
            int bits = (s->n % 64) + 1;
            if (bits < 64) mask &= (1ull << bits) - 1;
        }
        if (w == 0) mask &= ~1ull;

        #if defined(__AVX2__)
        // Ottimizzazione: se il blocco da 64 ha molti bit liberi, la scansione SIMD è più veloce
        // del loop ctzll se fatta con intelligenza. Per ora usiamo ctzll ma pre-carichiamo i dati.
        #endif

        while (mask != 0) {
            int bit = __builtin_ctzll(mask);
            int node = base + bit;
            
            // Prefetching manuale del prossimo nodo possibile
            uint64_t next_mask = mask & (mask - 1);
            if (next_mask != 0) {
                int next_bit = __builtin_ctzll(next_mask);
                __builtin_prefetch(&row[base + next_bit], 0, 3);
            }

            float d = row[node];
            if (d < best_d) {
                best_d = d;
                best = node;
            }
            mask &= mask - 1;
        }
    }
    return best;
}

static void build_ant_v2(AcoThreadWorkspace *ws, const V2RankShared *s, int K, int cap, const MatrixFloat *c, const float *scores) {
    solution_reset(ws->sol);
    memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
    memset(ws->route_loads, 0, (size_t)K * sizeof(int));
    int rem = s->n;
    for (int v = 0; v < K; v++) {
        Route *r = &ws->sol->routes[v]; route_append(r, 0);
        int curr = 0; int rem_v = K - v - 1, fut_cap = rem_v * cap;
        while (rem > 0 && rem > fut_cap && ws->route_loads[v] < cap) {
            int next = 0; float denom = 0.0f;
            const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
            const float *sc = scores + (size_t)curr * (size_t)s->stride;
            float weights[V2_MAX_CANDS];
            for (int t = 0; t < s->cand_k; t++) {
                int node = cands[t];
                if (node > 0 && !((ws->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) {
                    weights[t] = sc[t]; denom += sc[t];
                } else weights[t] = 0.0f;
            }
            if (denom > 0.0f) {
                float thres = (float)aco_rand01_state(&ws->rng_state) * denom, cum = 0.0f;
                for (int t = 0; t < s->cand_k; t++) {
                    if (weights[t] <= 0.0f) continue;
                    cum += weights[t]; if (cum >= thres) { next = cands[t]; break; }
                }
            }
            if (next <= 0) next = find_nearest_unvisited(s, curr, ws->visited, c);
            if (next <= 0) break;
            route_append(r, next); ws->visited[(unsigned)next >> 6] |= (1ull << ((unsigned)next & 63u));
            rem--; ws->route_loads[v]++; curr = next;
        }
        route_append(r, 0);
    }
}

#ifdef USE_MPI
static void sync_tau_v2(MatrixFloat *tau, int mpi_size) {
    size_t total = (size_t)(tau->n + 1) * (size_t)tau->stride;
    MPI_Allreduce(MPI_IN_PLACE, tau->data, (int)total, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float inv = 1.0f / (float)mpi_size;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= tau->n; i++) {
        for (int j = 0; j <= tau->n; j++) {
            if (i == j) tau->rows[i][j] = 0.0f;
            else tau->rows[i][j] *= inv;
        }
    }
}
#endif

static double solution_cost_f(const Solution *s, float **c) {
    double total = 0.0;
    for (int i = 0; i < s->K; i++) {
        const Route *r = &s->routes[i];
        for (int t = 0; t + 1 < r->len; t++) total += (double)c[r->nodes[t]][r->nodes[t+1]];
    }
    return total;
}

void aco_vrp_v2_run(int n, int K, int cap, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_sol, double *best_cost) {
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    int mpi_init = 0; MPI_Initialized(&mpi_init);
    if (mpi_init) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
#endif

    double max_runtime_sec; int max_stagnation_epochs; double min_rel_improvement;
    load_v2_directives(&max_runtime_sec, &max_stagnation_epochs, &min_rel_improvement);
    long l3_size = get_l3_cache_size(); int cand_k = tune_candidate_k(n, l3_size);
    int total_m = (m <= 0) ? (n / 2) * mpi_size : m;
    int local_m = total_m / mpi_size + (mpi_rank < (total_m % mpi_size));
    int ant_off = mpi_rank * (total_m / mpi_size) + ((mpi_rank < (total_m % mpi_size)) ? mpi_rank : (total_m % mpi_size));

    MatrixFloat *tau_mat = matrix_create_float(n); MatrixFloat *c_mat = matrix_create_float(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            c_mat->rows[i][j] = (float)c[i][j];
            tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0;
        }
    }

    V2RankShared shared; v2_shared_init(&shared, n, cand_k, c_mat, beta);
    float *score_mat = aligned_calloc_64((size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    Solution *iter_best = solution_create(K, n);
    double iter_best_cost = DBL_MAX; *best_cost = DBL_MAX; double start_time = wall_time();
    int iter_since_best = 0, total_iters = 0; bool stop_now = false;
    const char *s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS");
    int fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 0;

    if (mpi_rank == 0) {
        printf("Hybrid V2 Solver (FLOAT Optimized) starting... (N=%d, K=%d, M=%d)\n", n, K, total_m);
        if (fixed_epochs > 0) printf("  FIXED EPOCHS MODE: %d iterations\n", fixed_epochs);
        if (l3_size > 0) printf("  Detected L3 Cache: %.2f MB. Auto-tuned K_cand = %d\n", (double)l3_size / (1024*1024), cand_k);
        const char *omp_sched = getenv("OMP_SCHEDULE");
        printf("  OMP_SCHEDULE: %s\n", omp_sched ? omp_sched : "default (static)");
    }

    #pragma omp parallel default(shared) proc_bind(close)
    {
        AcoThreadWorkspace ws; v2_ws_init(&ws, K, n, shared.visited_words);
        float tau_max = (float)tau0, tau_min = (float)tau0*0.05f;
        for (int iter = 0;; iter++) {
            #pragma omp master
            { total_iters++; double ct = wall_time();
              if (fixed_epochs > 0) { if (iter >= fixed_epochs) stop_now = true; }
              else { if (max_runtime_sec > 0.0 && (ct - start_time) > max_runtime_sec) stop_now = true;
                     if (max_stagnation_epochs > 0 && iter_since_best >= max_stagnation_epochs) stop_now = true; }
            }
            #pragma omp barrier
            if (stop_now) break;

            #pragma omp for schedule(runtime)
            for (int i = 0; i <= n; i++) {
                int *cands = shared.cand_idx + (size_t)i * (size_t)shared.stride;
                float *etas = shared.eta_beta + (size_t)i * (size_t)shared.stride;
                float *sc = score_mat + (size_t)i * (size_t)shared.stride;
                const float *tau_row = tau_mat->rows[i];
                for (int t = 0; t < shared.cand_k; t++) {
                    int node = cands[t];
                    sc[t] = (node > 0) ? (fast_powf(tau_row[node], (float)alpha) * etas[t]) : 0.0f;
                }
            }
            #pragma omp barrier

            double t_best_c = DBL_MAX;
            #pragma omp single
            { iter_best_cost = DBL_MAX; }
            #pragma omp for schedule(runtime) nowait
            for (int a = 0; a < local_m; a++) {
                ws.rng_state = aco_make_ant_seed(seed, iter, ant_off + a);
                build_ant_v2(&ws, &shared, K, cap, c_mat, score_mat);
                double cost = solution_cost_f(ws.sol, c_mat->rows);
                if (cost < t_best_c) { t_best_c = cost; solution_copy(ws.thread_best, ws.sol); }
            }
            #pragma omp critical
            { if (t_best_c < iter_best_cost) { iter_best_cost = t_best_c; solution_copy(iter_best, ws.thread_best); } }
            #pragma omp barrier

            #pragma omp master
            {
#ifdef USE_MPI
                if (mpi_size > 1) { double g_min = iter_best_cost; MPI_Allreduce(MPI_IN_PLACE, &g_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD); iter_best_cost = g_min; }
#endif
                if (iter_best_cost < *best_cost) {
                    if (is_sig_imp(*best_cost, iter_best_cost, min_rel_improvement)) iter_since_best = 0; else iter_since_best++;
                    *best_cost = iter_best_cost; solution_copy(best_sol, iter_best);
                    tau_max = (float)(1.0 / ((1.0 - rho) * (*best_cost))); tau_min = tau_max * 0.05f;
                    if (mpi_rank == 0) printf("Epoch %d: New Best Cost = %.3f (T = %.2fs)\n", iter, *best_cost, wall_time()-start_time);
                } else iter_since_best++;
            }
            #pragma omp barrier
            if (stop_now) break;

            float rho_f = (float)rho;
            #pragma omp for collapse(2) schedule(runtime)
            for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) if (i != j) tau_mat->rows[i][j] *= (1.0f - rho_f);

            float dep = (float)(0.3 * Q / fmax(iter_best_cost, 1e-9));
            #pragma omp for schedule(runtime)
            for (int v = 0; v < K; v++) {
                Route *r = &iter_best->routes[v];
                for (int t = 0; t + 1 < r->len; t++) {
                    int from = r->nodes[t], to = r->nodes[t+1];
                    #pragma omp atomic
                    tau_mat->rows[from][to] += dep;
                    #pragma omp atomic
                    tau_mat->rows[to][from] += dep;
                }
            }
            if (*best_cost < DBL_MAX) {
              float g_dep = (float)(0.7 * Q / (*best_cost));
              #pragma omp for schedule(runtime)
              for (int v = 0; v < K; v++) {
                  Route *r = &best_sol->routes[v];
                  for (int t = 0; t + 1 < r->len; t++) {
                      int from = r->nodes[t], to = r->nodes[t+1];
                      #pragma omp atomic
                      tau_mat->rows[from][to] += g_dep;
                      #pragma omp atomic
                      tau_mat->rows[to][from] += g_dep;
                  }
              }
            }

            #pragma omp for collapse(2) schedule(runtime)
            for (int i = 0; i <= n; i++) {
                for (int j = 0; j <= n; j++) {
                    if (i == j) continue;
                    if (tau_mat->rows[i][j] < tau_min) tau_mat->rows[i][j] = tau_min;
                    else if (tau_mat->rows[i][j] > tau_max) tau_mat->rows[i][j] = tau_max;
                }
            }

            if (iter_since_best > 0 && (iter_since_best % 32 == 0)) {
              #pragma omp for schedule(runtime)
              for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) if (i != j) tau_mat->rows[i][j] = 0.5f * tau_mat->rows[i][j] + 0.5f * (float)tau0;
            }

#ifdef USE_MPI
            #pragma omp master
            if (mpi_size > 1) sync_tau_v2(tau_mat, mpi_size);
            #pragma omp barrier
#endif
        }
        v2_ws_free(&ws);
    }

    if (mpi_rank == 0) {
        printf("Executed Epochs: %d\n", total_iters);
        printf("V2 Completion. Final Best: %.3f. Time: %.3fs\n", *best_cost, wall_time() - start_time);
    }
    matrix_free_float(tau_mat); matrix_free_float(c_mat);
    free(score_mat); solution_free(iter_best); v2_shared_free(&shared);
}

void aco_vrp_v2(int n, int K, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    int cap = (K > 0) ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K)) : n;
    aco_vrp_v2_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}

void aco_vrp_v2_with_capacity(int n, int K, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    aco_vrp_v2_run(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}
