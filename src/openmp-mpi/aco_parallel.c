#include "aco.h"
#include "aco_config.h"
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

#define ACO_MPI_EPS 1e-7f

enum {
  kAcoMpiAlignment = 64,
  kAcoMpiMaxCandidates = 512,
};

typedef struct {
    uint32_t edge_idx;
    float increment;
} aco_mpi_sparse_delta_t;

typedef struct {
    int n;
    int stride;
    float *data;
    float **rows;
} aco_mpi_matrix_float_t;

/**
 * @brief Executes `align_up_64`.
 * @param value Function parameter.
 * @return Function result.
 */
static size_t align_up_64(size_t value) {
  size_t rem = value % kAcoMpiAlignment;
  return (rem == 0u) ? value : value + (kAcoMpiAlignment - rem);
}

/**
 * @brief Executes `matrix_create_float`.
 * @param n Function parameter.
 * @return Function result.
 */
static aco_mpi_matrix_float_t *matrix_create_float(int n) {
  aco_mpi_matrix_float_t *m = malloc(sizeof(*m));
  if (!m) {
    return NULL;
  }

  m->n = n;
  size_t row_bytes = (size_t)(n + 1) * sizeof(float);
  m->stride = (int)(align_up_64(row_bytes) / sizeof(float));
  size_t total_elems = (size_t)(n + 1) * (size_t)m->stride;
  m->data = aligned_alloc(kAcoMpiAlignment, total_elems * sizeof(float));
  m->rows = malloc((size_t)(n + 1) * sizeof(float *));
  if (!m->data || !m->rows) {
    free(m->data);
    free(m->rows);
    free(m);
    return NULL;
  }

  for (int i = 0; i <= n; i++) {
    m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
  }
  memset(m->data, 0, total_elems * sizeof(float));
  return m;
}

/**
 * @brief Executes `matrix_free_float`.
 * @param m Function parameter.
 */
static void matrix_free_float(aco_mpi_matrix_float_t *m) {
  if (!m) {
    return;
  }
  free(m->data);
  free(m->rows);
  free(m);
}

/**
 * @brief Executes `aligned_calloc_64`.
 * @param bytes Function parameter.
 * @return Function result.
 */
static void *aligned_calloc_64(size_t bytes) {
  size_t alloc_bytes = align_up_64(bytes);
  void *ptr = aligned_alloc(kAcoMpiAlignment, alloc_bytes);
  if (!ptr) {
    return NULL;
  }
  memset(ptr, 0, alloc_bytes);
  return ptr;
}

/**
 * @brief Executes `wall_time`.
 * @return Function result.
 */
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

/**
 * @brief Executes `get_l3_cache_size`.
 * @return Function result.
 */
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

/**
 * @brief Executes `tune_candidate_k`.
 * @param n Function parameter.
 * @param l3_size Function parameter.
 * @return Function result.
 */
static int tune_candidate_k(int n, long l3_size) {
    if (l3_size <= 0) return 32;
    double target_bytes = (double)l3_size * 0.7;
    int k = (int)(target_bytes / ((double)(n + 1) * 8.0));
    if (k < 16) k = 16;
    if (k > kAcoMpiMaxCandidates) k = kAcoMpiMaxCandidates;
    return k;
}

/**
 * @brief Executes `fast_powf`.
 * @param base Function parameter.
 * @param exp Function parameter.
 * @return Function result.
 */
static float fast_powf(float base, float exp) {
    if (exp == 1.0f) return base;
    if (exp == 2.0f) return base * base;
    if (exp == 0.5f) return sqrtf(base);
    return powf(base, exp);
}

typedef struct {
    int n;
    int cand_k;
    int stride;
    int visited_words;
    int meta_words;
    int *cand_idx;
    float *eta_beta;
} aco_mpi_rank_shared_t;

/**
 * @brief Executes `aco_mpi_shared_free`.
 * @param s Function parameter.
 */
static void aco_mpi_shared_free(aco_mpi_rank_shared_t *s) {
    if (!s) return;
    if (s->eta_beta) free(s->eta_beta);
    if (s->cand_idx) free(s->cand_idx);
}

/**
 * @brief Executes `aco_mpi_shared_init`.
 * @param s Function parameter.
 * @param n Function parameter.
 * @param cand_k Function parameter.
 * @param c_mat Function parameter.
 * @param beta Function parameter.
 * @return Function result.
 */
static int aco_mpi_shared_init(aco_mpi_rank_shared_t *s, int n, int cand_k, const aco_mpi_matrix_float_t *c_mat, double beta) {
    s->n = n; s->cand_k = cand_k;
    size_t row_bytes = (size_t)s->cand_k * sizeof(float);
    s->stride = (int)(align_up_64(row_bytes) / sizeof(float));
    s->visited_words = (n / 64) + 1;
    s->meta_words = (s->visited_words / 64) + 1;
    size_t total_elems = (size_t)(n + 1) * (size_t)s->stride;
    s->cand_idx = aligned_calloc_64(total_elems * sizeof(int));
    s->eta_beta = aligned_calloc_64(total_elems * sizeof(float));
    if (!s->cand_idx || !s->eta_beta) return 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) {
        int nodes[kAcoMpiMaxCandidates]; float dists[kAcoMpiMaxCandidates];
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

typedef struct {
    Solution *sol;
    uint64_t *visited;
    uint64_t *meta_active;
    int *route_loads;
    unsigned int rng_state;
    Solution *thread_best;
} aco_mpi_workspace_t;

/**
 * @brief Executes `aco_mpi_ws_free`.
 * @param ws Function parameter.
 */
static void aco_mpi_ws_free(aco_mpi_workspace_t *ws) {
  if (!ws) return;
  if (ws->route_loads) free(ws->route_loads);
  if (ws->visited) free(ws->visited);
  if (ws->meta_active) free(ws->meta_active);
  if (ws->thread_best) solution_free(ws->thread_best);
  if (ws->sol) solution_free(ws->sol);
}

/**
 * @brief Executes `aco_mpi_ws_init`.
 * @param ws Function parameter.
 * @param K Function parameter.
 * @param n Function parameter.
 * @param words Function parameter.
 * @param meta_words Function parameter.
 * @return Function result.
 */
static int aco_mpi_ws_init(aco_mpi_workspace_t *ws, int K, int n, int words, int meta_words) {
  ws->sol = solution_create(K, n); ws->thread_best = solution_create(K, n);
  ws->visited = aligned_calloc_64((size_t)words * sizeof(uint64_t));
  ws->meta_active = aligned_calloc_64((size_t)meta_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->meta_active || !ws->route_loads) { aco_mpi_ws_free(ws); return 0; }
  return 1;
}

/**
 * @brief Executes `find_nearest_unvisited_h`.
 * @param s Function parameter.
 * @param curr Function parameter.
 * @param ws Function parameter.
 * @param c Function parameter.
 * @return Function result.
 */
static int find_nearest_unvisited_h(const aco_mpi_rank_shared_t *s, int curr, const aco_mpi_workspace_t *ws, const aco_mpi_matrix_float_t *c) {
    int best = 0; float best_d = FLT_MAX;
    const float *row = c->rows[curr];
    for (int mw = 0; mw < s->meta_words; mw++) {
        uint64_t m_mask = ws->meta_active[mw];
        while (m_mask != 0) {
            int w_off = __builtin_ctzll(m_mask); int w = (mw << 6) + w_off;
            if (w >= s->visited_words) break;
            uint64_t v = ws->visited[w]; uint64_t mask = ~v; int base = w << 6;
            if (w == s->visited_words - 1) { int bits = (s->n % 64) + 1; if (bits < 64) mask &= (1ull << bits) - 1; }
            if (w == 0) mask &= ~1ull;
            while (mask != 0) {
                int bit = __builtin_ctzll(mask); int node = base + bit;
                float d = row[node]; if (d < best_d) { best_d = d; best = node; }
                mask &= mask - 1;
            }
            m_mask &= m_mask - 1;
        }
    }
    return best;
}

/**
 * @brief Executes `build_ant_h`.
 * @param ws Function parameter.
 * @param s Function parameter.
 * @param K Function parameter.
 * @param cap Function parameter.
 * @param c Function parameter.
 * @param scores Function parameter.
 */
static void build_ant_h(aco_mpi_workspace_t *ws, const aco_mpi_rank_shared_t *s, int K, int cap, const aco_mpi_matrix_float_t *c, const float *scores) {
    solution_reset(ws->sol); memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
    memset(ws->meta_active, 0xFF, (size_t)s->meta_words * sizeof(uint64_t));
    memset(ws->route_loads, 0, (size_t)K * sizeof(int));
    int rem = s->n;
    for (int v = 0; v < K; v++) {
        route_append(&ws->sol->routes[v], 0); int curr = 0;
        while (true) {
            int rem_v = K - v - 1, fut_cap = rem_v * cap;
            if (!(rem > 0 && rem > fut_cap && ws->route_loads[v] < cap)) break;
            int next = 0; float denom = 0.0f;
            const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
            const float *sc = scores + (size_t)curr * (size_t)s->stride;
            for (int t = 0; t < s->cand_k; t++) {
                int node = cands[t];
                if (node > 0 && !((ws->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) { denom += sc[t]; }
            }
            if (denom > 0.0f) {
                float thres = (float)aco_rand01_state(&ws->rng_state) * denom, cum = 0.0f;
                for (int t = 0; t < s->cand_k; t++) {
                    int node = cands[t];
                    if (node > 0 && !((ws->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) {
                        cum += sc[t]; if (cum >= thres) { next = node; break; }
                    }
                }
            }
            if (next <= 0) next = find_nearest_unvisited_h(s, curr, ws, c);
            if (next <= 0) break;
            route_append(&ws->sol->routes[v], next);
            int word_idx = (unsigned)next >> 6;
            ws->visited[word_idx] |= (1ull << ((unsigned)next & 63u));
            if (ws->visited[word_idx] == 0xFFFFFFFFFFFFFFFFull) ws->meta_active[word_idx >> 6] &= ~(1ull << (word_idx & 63u));
            ws->route_loads[v]++; rem--; curr = next;
        }
        route_append(&ws->sol->routes[v], 0);
    }
}

/**
 * @brief Executes `solution_cost_f`.
 * @param s Function parameter.
 * @param c Function parameter.
 * @return Function result.
 */
static double solution_cost_f(const Solution *s, float **c) {
    double total = 0.0;
    for (int i = 0; i < s->K; i++) {
        const Route *r = &s->routes[i];
        if (r->len <= 2) continue;
        for (int t = 0; t + 1 < r->len; t++) total += (double)c[r->nodes[t]][r->nodes[t+1]];
    }
    return total;
}

#ifdef USE_MPI
typedef struct {
    MPI_Request req;
    aco_mpi_sparse_delta_t *recv_buf;
    int *counts;
    int *displs;
    MPI_Datatype type;
    int active;
} aco_mpi_async_sparse_context_t;

/**
 * @brief Executes `async_sparse_init`.
 * @param ctx Function parameter.
 * @param mpi_size Function parameter.
 */
static void async_sparse_init(aco_mpi_async_sparse_context_t *ctx, int mpi_size) {
    ctx->req = MPI_REQUEST_NULL; ctx->recv_buf = NULL;
    ctx->counts = calloc((size_t)mpi_size, sizeof(int));
    ctx->displs = calloc((size_t)mpi_size, sizeof(int));
    int blocklengths[2] = {
        [0] = 1,
        [1] = 1,
    };
    MPI_Aint offsets[2];
    offsets[0] = offsetof(aco_mpi_sparse_delta_t, edge_idx);
    offsets[1] = offsetof(aco_mpi_sparse_delta_t, increment);
    MPI_Datatype types[2] = {
        [0] = MPI_UINT32_T,
        [1] = MPI_FLOAT,
    };
    MPI_Type_create_struct(2, blocklengths, offsets, types, &ctx->type);
    MPI_Type_commit(&ctx->type);
    ctx->active = 0;
}

/**
 * @brief Executes `async_sparse_cleanup`.
 * @param ctx Function parameter.
 */
static void async_sparse_cleanup(aco_mpi_async_sparse_context_t *ctx) {
    if (!ctx) return;
    if (ctx->active) MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
    if (ctx->counts) free(ctx->counts);
    if (ctx->displs) free(ctx->displs);
    if (ctx->recv_buf) free(ctx->recv_buf);
    MPI_Type_free(&ctx->type); ctx->active = 0;
}

/**
 * @brief Executes `async_sparse_wait_and_apply`.
 * @param ctx Function parameter.
 * @param tau Function parameter.
 * @param mpi_rank Function parameter.
 * @param mpi_size Function parameter.
 */
static void async_sparse_wait_and_apply(aco_mpi_async_sparse_context_t *ctx, aco_mpi_matrix_float_t *tau, int mpi_rank, int mpi_size) {
    if (!ctx->active) return;
    MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
    float inv = 1.0f / (float)mpi_size;
    for (int r = 0; r < mpi_size; r++) {
        if (r == mpi_rank) continue;
        for (int i = ctx->displs[r]; i < ctx->displs[r] + ctx->counts[r]; i++) {
            tau->data[ctx->recv_buf[i].edge_idx] += ctx->recv_buf[i].increment * inv;
        }
    }
    free(ctx->recv_buf); ctx->recv_buf = NULL; ctx->active = 0;
}

/**
 * @brief Executes `async_sparse_start`.
 * @param ctx Function parameter.
 * @param local_deltas Function parameter.
 * @param local_count Function parameter.
 * @param mpi_size Function parameter.
 */
static void async_sparse_start(aco_mpi_async_sparse_context_t *ctx, aco_mpi_sparse_delta_t *local_deltas, int local_count, int mpi_size) {
    int my_c = local_count;
    MPI_Allgather(&my_c, 1, MPI_INT, ctx->counts, 1, MPI_INT, MPI_COMM_WORLD);
    int total_recv = 0;
    for (int i = 0; i < mpi_size; i++) { ctx->displs[i] = total_recv; total_recv += ctx->counts[i]; }
    if (total_recv > 0) {
        ctx->recv_buf = malloc((size_t)total_recv * sizeof(aco_mpi_sparse_delta_t));
        MPI_Iallgatherv(local_deltas, local_count, ctx->type, ctx->recv_buf, ctx->counts, ctx->displs, ctx->type, MPI_COMM_WORLD, &ctx->req);
        ctx->active = 1;
    } else { ctx->active = 0; ctx->req = MPI_REQUEST_NULL; }
}
#endif

/**
 * @brief Executes `is_sig_imp`.
 * @param prev_best Function parameter.
 * @param new_best Function parameter.
 * @param min_rel_improvement Function parameter.
 * @return Function result.
 */
static int is_sig_imp(double prev_best, double new_best, double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) return (new_best < prev_best);
  if (new_best >= prev_best - ACO_MPI_EPS) return 0;
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, (double)ACO_MPI_EPS);
  return rel_gain + (double)ACO_MPI_EPS >= min_rel_improvement;
}

/**
 * @brief Core MPI/OpenMP ACO execution loop.
 * @param n Number of customers.
 * @param K Number of vehicles.
 * @param cap Per-vehicle customer capacity.
 * @param m Number of ants (0 enables backend auto-tuning).
 * @param c Distance matrix.
 * @param alpha Pheromone exponent.
 * @param beta Heuristic exponent.
 * @param rho Evaporation factor.
 * @param tau0 Initial pheromone value.
 * @param Q Deposit scaling factor.
 * @param seed RNG seed.
 * @param best_sol Output best solution.
 * @param best_cost Output best cost.
 */
AcoStatus aco_vrp_run(int n, int K, int cap, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_sol, double *best_cost, const AcoRuntimeConfig *config) {
    if (n <= 0 || K <= 0 || !c || !best_sol || !best_cost) return ACO_ERR_INVALID_INPUT;
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    int mpi_init = 0; MPI_Initialized(&mpi_init);
    if (mpi_init) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
#endif
    long l3_size = get_l3_cache_size(); int cand_k = tune_candidate_k(n, l3_size);
    int total_m = (m <= 0) ? ((config && config->ants > 0) ? config->ants : (n / 2) * mpi_size) : m;
    int ant_off = mpi_rank * (total_m / mpi_size) + ((mpi_rank < (total_m % mpi_size)) ? mpi_rank : (total_m % mpi_size));
    int local_m = total_m / mpi_size + (mpi_rank < (total_m % mpi_size));
    double max_runtime_sec = config ? config->timeout_seconds : 0.0; int max_stagnation_epochs = config ? config->stagnation_epochs : 0; double min_rel_improvement = config ? config->min_rel_improvement : 1e-3; double progress_interval_sec = config ? config->progress_interval_seconds : 10.0;
    if (max_stagnation_epochs <= 0) max_stagnation_epochs = 100;
    aco_mpi_matrix_float_t *tau_mat = matrix_create_float(n); aco_mpi_matrix_float_t *c_mat = matrix_create_float(n);
    if (!tau_mat || !c_mat) { matrix_free_float(tau_mat); matrix_free_float(c_mat); return ACO_ERR_ALLOCATION; }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { c_mat->rows[i][j] = (float)c[i][j]; tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0; }
    aco_mpi_rank_shared_t shared; if (!aco_mpi_shared_init(&shared, n, cand_k, c_mat, beta)) { matrix_free_float(tau_mat); matrix_free_float(c_mat); return ACO_ERR_ALLOCATION; }
    float *score_mat = aligned_alloc(kAcoMpiAlignment, (size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    Solution *iter_best_sol_rank = solution_create(K, n); double iter_best_cost_g = DBL_MAX; *best_cost = DBL_MAX; double start_time = wall_time();
    double next_progress_time = (progress_interval_sec > 0.0) ? start_time + progress_interval_sec : 0.0;
    if (!score_mat || !iter_best_sol_rank) { free(score_mat); solution_free(iter_best_sol_rank); aco_mpi_shared_free(&shared); matrix_free_float(tau_mat); matrix_free_float(c_mat); return ACO_ERR_ALLOCATION; }
    int fixed_epochs = config ? config->fixed_epochs : 0;
#ifdef USE_MPI
    aco_mpi_async_sparse_context_t async_ctx; async_sparse_init(&async_ctx, mpi_size);
#endif
    int iter_since_best = 0; aco_mpi_sparse_delta_t *rank_deltas = malloc((size_t)(n + K + 500) * 2 * sizeof(aco_mpi_sparse_delta_t)); int rank_delta_count = 0;
    if (!rank_deltas) { solution_free(iter_best_sol_rank); aco_mpi_shared_free(&shared); matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat); return ACO_ERR_ALLOCATION; }
    #pragma omp parallel default(shared) proc_bind(close)
    {
        aco_mpi_workspace_t ws; aco_mpi_ws_init(&ws, K, n, shared.visited_words, shared.meta_words);
        if (mpi_rank == 0 && omp_get_thread_num() == 0) printf("ACO Parallel Starting with %d threads. N=%d K=%d Cap=%d\n", omp_get_num_threads(), n, K, cap);
        for (int iter = 0; (fixed_epochs > 0 && iter < fixed_epochs) || (fixed_epochs <= 0 && iter_since_best < max_stagnation_epochs); iter++) {
            if (max_runtime_sec > 0.0 && (wall_time() - start_time) > max_runtime_sec) break;
#ifdef USE_MPI
            #pragma omp master
            async_sparse_wait_and_apply(&async_ctx, tau_mat, mpi_rank, mpi_size);
            #pragma omp barrier
#endif
            #pragma omp for schedule(static) nowait
            for (int i = 0; i <= n; i++) {
                int *cands = shared.cand_idx + (size_t)i * (size_t)shared.stride; float *etas = shared.eta_beta + (size_t)i * (size_t)shared.stride;
                float *sc = score_mat + (size_t)i * (size_t)shared.stride; const float *tau_row = tau_mat->rows[i];
                for (int t = 0; t < shared.cand_k; t++) { int node = cands[t]; sc[t] = (node > 0) ? (fast_powf(tau_row[node], (float)alpha) * etas[t]) : 0.0f; }
            }
            #pragma omp barrier
            double t_best_c = DBL_MAX;
            #pragma omp for schedule(runtime) nowait
            for (int a = 0; a < local_m; a++) {
                ws.rng_state = aco_make_ant_seed(seed, iter, ant_off + a);
                build_ant_h(&ws, &shared, K, cap, c_mat, score_mat);
                double cost = solution_cost_f(ws.sol, c_mat->rows);
                if (cost < t_best_c) { t_best_c = cost; solution_copy(ws.thread_best, ws.sol); }
            }
            #pragma omp critical
            { if (t_best_c < iter_best_cost_g) { iter_best_cost_g = t_best_c; solution_copy(iter_best_sol_rank, ws.thread_best); } }
            #pragma omp barrier
            #pragma omp master
            {
                double g_min = iter_best_cost_g;
#ifdef USE_MPI
                if (mpi_size > 1) MPI_Allreduce(MPI_IN_PLACE, &g_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
                if (is_sig_imp(*best_cost, g_min, min_rel_improvement)) iter_since_best = 0; else iter_since_best++;
                if (g_min < *best_cost) { *best_cost = g_min; solution_copy(best_sol, iter_best_sol_rank);
                }
                double now = wall_time();
                if (mpi_rank == 0 && progress_interval_sec > 0.0 && now >= next_progress_time) {
                    double elapsed = now - start_time;
                    if (max_runtime_sec > 0.0) {
                        double remaining = max_runtime_sec - elapsed;
                        if (remaining < 0.0) remaining = 0.0;
                        fprintf(stderr, "[mpi] elapsed %.1fs, remaining %.1fs, iter %d, best %.3f\n", elapsed, remaining, iter + 1, *best_cost);
                    } else {
                        fprintf(stderr, "[mpi] elapsed %.1fs, iter %d, best %.3f\n", elapsed, iter + 1, *best_cost);
                    }
                    next_progress_time = now + progress_interval_sec;
                }
                iter_best_cost_g = DBL_MAX; rank_delta_count = 0;
            }
            #pragma omp barrier
            float rho_f = (float)rho;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < (size_t)(n+1)*tau_mat->stride; i++) tau_mat->data[i] *= (1.0f - rho_f);
            float dep = (float)(Q / fmax(*best_cost, 1e-9)), weighted_dep = dep / (float)mpi_size;
            #pragma omp for schedule(static)
            for (int v = 0; v < K; v++) {
                Route *r = &best_sol->routes[v];
                if (r->len <= 2) continue;
                for (int t = 0; t + 1 < r->len; t++) {
                    int from = r->nodes[t], to = r->nodes[t+1];
                    uint32_t idx1 = (uint32_t)((size_t)from * tau_mat->stride + to);
                    uint32_t idx2 = (uint32_t)((size_t)to * tau_mat->stride + from);
                    #pragma omp atomic
                    tau_mat->data[idx1] += weighted_dep;
                    #pragma omp atomic
                    tau_mat->data[idx2] += weighted_dep;
                    int d_idx;
                    #pragma omp atomic capture
                    d_idx = rank_delta_count++;
                    rank_deltas[d_idx] = (aco_mpi_sparse_delta_t){
                        .edge_idx = idx1,
                        .increment = weighted_dep,
                    };
                    #pragma omp atomic capture
                    d_idx = rank_delta_count++;
                    rank_deltas[d_idx] = (aco_mpi_sparse_delta_t){
                        .edge_idx = idx2,
                        .increment = weighted_dep,
                    };
                }
            }
            #pragma omp barrier
            #pragma omp master
            {
#ifdef USE_MPI
                if (mpi_size > 1) async_sparse_start(&async_ctx, rank_deltas, rank_delta_count, mpi_size);
#endif
            }
            #pragma omp barrier
        }
#ifdef USE_MPI
        #pragma omp master
        async_sparse_cleanup(&async_ctx);
#endif
        aco_mpi_ws_free(&ws);
    }
    free(rank_deltas); if (mpi_rank == 0) printf("ACO Parallel Ultimate Completion. Best: %.3f. Time: %.3fs\n", *best_cost, wall_time() - start_time);
    matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat); solution_free(iter_best_sol_rank); aco_mpi_shared_free(&shared);
    return (*best_cost < DBL_MAX) ? ACO_OK : ACO_ERR_NO_SOLUTION;
}

/**
 * @brief Runs the MPI/OpenMP ACO solver with explicit vehicle capacity.
 */
AcoStatus aco_vrp_with_capacity(int n, int K, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    AcoRuntimeConfig config;
    aco_runtime_config_load_env(&config);
    config.ants = m;
    config.seed = seed;
    return aco_vrp_run(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost, &config);
}

/**
 * @brief Runs the MPI/OpenMP ACO solver with auto-derived vehicle capacity.
 */
AcoStatus aco_vrp(int n, int K, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    int cap = (K > 0) ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K)) : n;
    return aco_vrp_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}
