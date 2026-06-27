#include "aco_v2.h"
# include "solver.h"
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

#define V2_ALIGNMENT 64u
#define V2_MAX_CANDS 512
#define V2_EPS 1e-7f

/* -- Sparse Sync Structure -- */
typedef struct {
    uint32_t edge_idx;
    float increment;
} SparseDelta;

/* -- Matrices in FLOAT -- */
typedef struct {
    int n;
    int stride;
    float *data;
    float **rows;
} MatrixFloat;

static size_t align_up_64(size_t value) {
    size_t rem = value % V2_ALIGNMENT;
    return (rem == 0u) ? value : value + (V2_ALIGNMENT - rem);
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
        if (*unit == 'k' || *unit == 'k') size *= 1024;
        else if (*unit == 'M' || *unit == 'm') size *= 1024 * 1024;
        else if (*unit == 'G' || *unit == 'g') size *= 1024 * 1024 * 1024;
    }
    return size;
}

static int tune_candidate_k(int n, long l3_size) {
    if (l3_size <= 0) return 32;
    double target_bytes = (double)l3_size * 0.7;
    int k = (int)(target_bytes / ((double)(n + 1) * 8.0));
    if (k < 16) k = 16;
    if (k > V2_MAX_CANDS) k = V2_MAX_CANDS;
    return k;
}

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
    s->meta_words = (s->visited_words / 64) + 1;
    size_t total_elems = (size_t)(n + 1) * (size_t)s->stride;
    s->cand_idx = aligned_calloc_64(total_elems * sizeof(int));
    s->eta_beta = aligned_calloc_64(total_elems * sizeof(float));
    if (!s->cand_idx || !s->eta_beta) return 0;
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

typedef struct {
    t_solution *sol;
    uint64_t *visited;
    uint64_t *meta_active;
    int *route_loads;
    unsigned int rng_state;
    t_solution *thread_best;
} HierarchicalWorkspace;

static void v2_ws_free(HierarchicalWorkspace *ws) {
  if (!ws) return;
  free(ws->route_loads); free(ws->visited); free(ws->meta_active);
  solution_free(ws->thread_best); solution_free(ws->sol);
}

static int v2_ws_init(HierarchicalWorkspace *ws, int k, int n, int words, int meta_words) {
  ws->sol = solution_create(k, n); ws->thread_best = solution_create(k, n);
  ws->visited = aligned_calloc_64((size_t)words * sizeof(uint64_t));
  ws->meta_active = aligned_calloc_64((size_t)meta_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)k, sizeof(int));
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->meta_active || !ws->route_loads) { v2_ws_free(ws); return 0; }
  return 1;
}

static int find_nearest_unvisited_h(const V2RankShared *s, int curr, const HierarchicalWorkspace *ws, const MatrixFloat *c) {
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

static void build_ant_v2_h(HierarchicalWorkspace *ws, const V2RankShared *s, int k, int cap, const MatrixFloat *c, const float *scores) {
    solution_reset(ws->sol); memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
    memset(ws->meta_active, 0xFF, (size_t)s->meta_words * sizeof(uint64_t));
    memset(ws->route_loads, 0, (size_t)k * sizeof(int));
    int rem = s->n;
    for (int v = 0; v < k; v++) {
        route_append(&ws->sol->routes[v], 0); int curr = 0;
        while (true) {
            int rem_v = k - v - 1, fut_cap = rem_v * cap;
            if (!(rem > 0 && rem > fut_cap && ws->route_loads[v] < cap)) break;
            int next = 0; float denom = 0.0f;
            const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
            const float *sc = scores + (size_t)curr * (size_t)s->stride;
            for (int t = 0; t < s->cand_k; t++) {
                int node = cands[t];
                if (node > 0 && !((ws->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) { denom += sc[t]; }
            }
            if (denom > 0.0f) {
                float thres = (float)rand01_state(&ws->rng_state) * denom, cum = 0.0f;
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

static double solution_cost_f(const t_solution *s, float **c) {
    double total = 0.0;
    for (int i = 0; i < s->k; i++) {
        const t_route *r = &s->routes[i];
        for (int t = 0; t + 1 < r->len; t++) total += (double)c[r->nodes[t]][r->nodes[t+1]];
    }
    return total;
}

#ifdef USE_MPI
typedef struct {
    MPI_Request req;
    SparseDelta *recv_buf;
    int *counts;
    int *displs;
    MPI_Datatype type;
    int active;
} AsyncSparseContext;

static void async_sparse_init(AsyncSparseContext *ctx, int mpi_size) {
    ctx->req = MPI_REQUEST_NULL; ctx->recv_buf = NULL;
    ctx->counts = calloc((size_t)mpi_size, sizeof(int));
    ctx->displs = calloc((size_t)mpi_size, sizeof(int));
    
    // MPI Struct Type per portabilità
    int blocklengths[2] = {1, 1};
    MPI_Aint offsets[2];
    offsets[0] = offsetof(SparseDelta, edge_idx);
    offsets[1] = offsetof(SparseDelta, increment);
    MPI_Datatype types[2] = {MPI_UINT32_T, MPI_FLOAT};
    MPI_Type_create_struct(2, blocklengths, offsets, types, &ctx->type);
    MPI_Type_commit(&ctx->type);
    ctx->active = 0;
}

static void async_sparse_cleanup(AsyncSparseContext *ctx) {
    if (!ctx) return;
    if (ctx->active) MPI_Wait(&ctx->req, MPI_STATUS_IGNORE);
    if (ctx->counts) free(ctx->counts); if (ctx->displs) free(ctx->displs);
    if (ctx->recv_buf) free(ctx->recv_buf);
    MPI_Type_free(&ctx->type); ctx->active = 0;
}

static void async_sparse_wait_and_apply(AsyncSparseContext *ctx, MatrixFloat *tau, int mpi_rank, int mpi_size) {
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

static void async_sparse_start(AsyncSparseContext *ctx, SparseDelta *local_deltas, int local_count, int mpi_size) {
    int my_c = local_count;
    MPI_Allgather(&my_c, 1, MPI_INT, ctx->counts, 1, MPI_INT, MPI_COMM_WORLD);
    int total_recv = 0;
    for (int i = 0; i < mpi_size; i++) { ctx->displs[i] = total_recv; total_recv += ctx->counts[i]; }
    if (total_recv > 0) {
        ctx->recv_buf = malloc((size_t)total_recv * sizeof(SparseDelta));
        MPI_Iallgatherv(local_deltas, local_count, ctx->type, ctx->recv_buf, ctx->counts, ctx->displs, ctx->type, MPI_COMM_WORLD, &ctx->req);
        ctx->active = 1;
    } else { ctx->active = 0; ctx->req = MPI_REQUEST_NULL; }
}
#endif

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
  double rel_gain = abs_gain / fmax(prev_best, (double)V2_EPS);
  return rel_gain + (double)V2_EPS >= min_rel_improvement;
}

void aco_vrp_v2_run(int n, int k, int cap, int m, double **c, double alpha, double beta, double rho, double tau0, double q, unsigned int seed, t_solution *best_sol, double *best_cost) {
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    int mpi_init = 0; MPI_Initialized(&mpi_init);
    if (mpi_init) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
#endif
    long l3_size = get_l3_cache_size(); int cand_k = tune_candidate_k(n, l3_size);
    int total_m = (m <= 0) ? (n / 2) * mpi_size : m;
    int ant_off = mpi_rank * (total_m / mpi_size) + ((mpi_rank < (total_m % mpi_size)) ? mpi_rank : (total_m % mpi_size));
    int local_m = total_m / mpi_size + (mpi_rank < (total_m % mpi_size));
    double max_runtime_sec; int max_stagnation_epochs; double min_rel_improvement;
    load_v2_directives(&max_runtime_sec, &max_stagnation_epochs, &min_rel_improvement);
    MatrixFloat *tau_mat = matrix_create_float(n); MatrixFloat *c_mat = matrix_create_float(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { c_mat->rows[i][j] = (float)c[i][j]; tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0; }
    V2RankShared shared; v2_shared_init(&shared, n, cand_k, c_mat, beta);
    float *score_mat = aligned_alloc(V2_ALIGNMENT, (size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    t_solution *iter_best = solution_create(k, n); double iter_best_cost_g = DBL_MAX; *best_cost = DBL_MAX; double start_time = wall_time();
    const char *s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS"); int fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 100;
#ifdef USE_MPI
    AsyncSparseContext async_ctx; async_sparse_init(&async_ctx, mpi_size);
#endif
    int iter_since_best = 0; SparseDelta *rank_deltas = malloc((size_t)(n + k + 500) * 2 * sizeof(SparseDelta)); int rank_delta_count = 0;
    #pragma omp parallel default(shared) proc_bind(close)
    {
        HierarchicalWorkspace ws; v2_ws_init(&ws, k, n, shared.visited_words, shared.meta_words);
        float tau_max = (float)tau0, tau_min = (float)tau0*0.05f;
        for (int iter = 0; iter < fixed_epochs; iter++) {
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
                ws.rng_state = make_ant_seed(seed, iter, ant_off + a);
                build_ant_v2_h(&ws, &shared, k, cap, c_mat, score_mat);
                double cost = solution_cost_f(ws.sol, c_mat->rows);
                if (cost < t_best_c) { t_best_c = cost; solution_copy(ws.thread_best, ws.sol); }
            }
            #pragma omp critical
            { if (t_best_c < iter_best_cost_g) { iter_best_cost_g = t_best_c; solution_copy(iter_best, ws.thread_best); } }
            #pragma omp barrier
            #pragma omp master
            {
                double g_min = iter_best_cost_g;
#ifdef USE_MPI
                if (mpi_size > 1) MPI_Allreduce(MPI_IN_PLACE, &g_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
                if (is_sig_imp(*best_cost, g_min, min_rel_improvement)) iter_since_best = 0; else iter_since_best++;
                if (g_min < *best_cost) { *best_cost = g_min; solution_copy(best_sol, iter_best);
                    tau_max = (float)(1.0 / ((1.0 - rho) * (*best_cost))); tau_min = tau_max * 0.05f; }
                if (mpi_rank == 0 && iter % 10 == 0) printf("Epoch %d: Best Cost = %.3f\n", iter, *best_cost);
                iter_best_cost_g = DBL_MAX; rank_delta_count = 0;
            }
            #pragma omp barrier
            float rho_f = (float)rho;
            #pragma omp for schedule(static)
            for (size_t i = 0; i < (size_t)(n+1)*tau_mat->stride; i++) tau_mat->data[i] *= (1.0f - rho_f);
            float dep = (float)(q / fmax(*best_cost, 1e-9)), weighted_dep = dep / (float)mpi_size;
            #pragma omp for schedule(static)
            for (int v = 0; v < k; v++) {
                t_route *r = &best_sol->routes[v];
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
                    rank_deltas[d_idx] = (SparseDelta){idx1, weighted_dep};
                    #pragma omp atomic capture
                    d_idx = rank_delta_count++;
                    rank_deltas[d_idx] = (SparseDelta){idx2, weighted_dep};
                }
            }
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
        v2_ws_free(&ws);
    }
    free(rank_deltas); if (mpi_rank == 0) printf("V2 Ultimate Completion. Best: %.3f. Time: %.3fs\n", *best_cost, wall_time() - start_time);
    matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat); solution_free(iter_best); v2_shared_free(&shared);
}

void aco_vrp_v2(int n, int k, int m, double **c, double alpha, double beta, double rho, double tau0, double q, unsigned int seed, t_solution *best_solution, double *best_cost) {
    int cap = (k > 0) ? (int)(((long long)120 * n + 100 * k - 1) / (100 * k)) : n;
    aco_vrp_v2_with_capacity(n, k, cap, m, c, alpha, beta, rho, tau0, q, seed, best_solution, best_cost);
}

void aco_vrp_v2_with_capacity(int n, int k, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double q, unsigned int seed, t_solution *best_solution, double *best_cost) {
    aco_vrp_v2_run(n, k, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, q, seed, best_solution, best_cost);
}
