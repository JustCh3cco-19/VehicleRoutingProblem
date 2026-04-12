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

#define V2_ALIGNMENT 64u
#define V2_MAX_CANDS 512
#define V2_EPS 1e-7f

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
        if (*unit == 'K' || *unit == 'k') size *= 1024;
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
    Solution *sol;
    uint64_t *visited;
    uint64_t *meta_active;
    int *route_loads;
    unsigned int rng_state;
    Solution *thread_best;
} HierarchicalWorkspace;

static void v2_ws_free(HierarchicalWorkspace *ws) {
  if (!ws) return;
  free(ws->route_loads); free(ws->visited); free(ws->meta_active);
  solution_free(ws->thread_best); solution_free(ws->sol);
}

static int v2_ws_init(HierarchicalWorkspace *ws, int K, int n, int words, int meta_words) {
  ws->sol = solution_create(K, n); ws->thread_best = solution_create(K, n);
  ws->visited = aligned_calloc_64((size_t)words * sizeof(uint64_t));
  ws->meta_active = aligned_calloc_64((size_t)meta_words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)K, sizeof(int));
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

static void build_ant_v2_h(HierarchicalWorkspace *ws, const V2RankShared *s, int K, int cap, const MatrixFloat *c, const float *scores) {
    solution_reset(ws->sol);
    memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
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
    long l3_size = get_l3_cache_size(); int cand_k = tune_candidate_k(n, l3_size);
    int total_m = (m <= 0) ? (n / 2) * mpi_size : m;
    int ant_off = mpi_rank * (total_m / mpi_size) + ((mpi_rank < (total_m % mpi_size)) ? mpi_rank : (total_m % mpi_size));
    int local_m = total_m / mpi_size + (mpi_rank < (total_m % mpi_size));

    MatrixFloat *tau_mat = matrix_create_float(n); MatrixFloat *c_mat = matrix_create_float(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { c_mat->rows[i][j] = (float)c[i][j]; tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0; }
    V2RankShared shared; v2_shared_init(&shared, n, cand_k, c_mat, beta);
    float *score_mat = aligned_alloc(V2_ALIGNMENT, (size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    Solution *iter_best = solution_create(K, n); *best_cost = DBL_MAX; double start_time = wall_time();
    const char *s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS"); int fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 100;

    // Diagnostica Sparsità
    uint64_t *dirty_edges = calloc((size_t)(n+1)*(n+1)/64 + 1, sizeof(uint64_t));

    #pragma omp parallel default(shared) proc_bind(close)
    {
        HierarchicalWorkspace ws; v2_ws_init(&ws, K, n, shared.visited_words, shared.meta_words);
        for (int iter = 0; iter < fixed_epochs; iter++) {
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
                build_ant_v2_h(&ws, &shared, K, cap, c_mat, score_mat);
                double cost = solution_cost_f(ws.sol, c_mat->rows);
                if (cost < t_best_c) { t_best_c = cost; solution_copy(ws.thread_best, ws.sol); }
            }
            #pragma omp critical
            { if (t_best_c < *best_cost) { *best_cost = t_best_c; solution_copy(best_sol, ws.thread_best); } }
            #pragma omp barrier

            // Pheromone Deposit & Sparsity Count
            float rho_f = (float)rho;
            #pragma omp for collapse(2) schedule(static)
            for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) if (i != j) tau_mat->rows[i][j] *= (1.0f - rho_f);
            
            float dep = (float)(Q / fmax(*best_cost, 1e-9));
            #pragma omp for schedule(static)
            for (int v = 0; v < K; v++) {
                Route *r = &ws.thread_best->routes[v]; // Usiamo thread_best per ogni thread per contare tutti i link toccati nel rank
                for (int t = 0; t + 1 < r->len; t++) {
                    int from = r->nodes[t], to = r->nodes[t+1];
                    #pragma omp atomic
                    tau_mat->rows[from][to] += dep;
                    #pragma omp atomic
                    tau_mat->rows[to][from] += dep;
                    // Segnamo il bordo come "dirty"
                    size_t idx1 = (size_t)from * (n+1) + to;
                    size_t idx2 = (size_t)to * (n+1) + from;
                    #pragma omp atomic
                    dirty_edges[idx1 >> 6] |= (1ull << (idx1 & 63));
                    #pragma omp atomic
                    dirty_edges[idx2 >> 6] |= (1ull << (idx2 & 63));
                }
            }
            
            #pragma omp master
            {
                if (iter == fixed_epochs - 1) {
                    size_t count = 0;
                    for (size_t i = 0; i < (size_t)(n+1)*(n+1); i++) {
                        if ((dirty_edges[i >> 6] >> (i & 63)) & 1ull) count++;
                    }
                    double density = (double)count / ((double)(n+1)*(n+1)) * 100.0;
                    printf("RANK %d: Epoch %d Sparsity Analysis:\n", mpi_rank, iter);
                    printf("  - Total Edges: %lu\n", (unsigned long)(n+1)*(n+1));
                    printf("  - Dirty Edges: %lu\n", (unsigned long)count);
                    printf("  - Update Density: %.4f%%\n", density);
                    printf("  - Sparse Transfer Size: %.2f MB (vs %.2f MB full)\n", 
                           (double)count * 8.0 / (1024*1024), (double)(n+1)*(n+1)*4.0 / (1024*1024));
                }
            }
            #pragma omp barrier
            // Sincronizzazione MPI standard per validità algoritmica durante il test
#ifdef USE_MPI
            #pragma omp master
            if (mpi_size > 1) {
                size_t total_elements = (size_t)(n + 1) * (size_t)tau_mat->stride;
                MPI_Allreduce(MPI_IN_PLACE, tau_mat->data, (int)total_elements, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                float inv = 1.0f / (float)mpi_size;
                for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) if (i != j) tau_mat->rows[i][j] *= inv;
            }
            #pragma omp barrier
#endif
        }
        v2_ws_free(&ws);
    }
    free(dirty_edges); matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat);
}

void aco_vrp_v2(int n, int K, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    int cap = (K > 0) ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K)) : n;
    aco_vrp_v2_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}

void aco_vrp_v2_with_capacity(int n, int K, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    aco_vrp_v2_run(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}
