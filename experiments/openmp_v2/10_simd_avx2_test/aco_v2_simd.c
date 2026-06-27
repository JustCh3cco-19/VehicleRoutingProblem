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

#include <immintrin.h>

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

static void v2_ws_free(t_thread_workspace *ws) {
  if (!ws) return;
  free(ws->route_loads); free(ws->visited);
  solution_free(ws->thread_best); solution_free(ws->sol);
}

static int v2_ws_init(t_thread_workspace *ws, int k, int n, int words) {
  ws->sol = solution_create(k, n); ws->thread_best = solution_create(k, n);
  ws->visited = aligned_calloc_64((size_t)words * sizeof(uint64_t));
  ws->route_loads = calloc((size_t)k, sizeof(int));
  if (!ws->sol || !ws->thread_best || !ws->visited || !ws->route_loads) { v2_ws_free(ws); return 0; }
  return 1;
}

static int find_nearest_unvisited(const V2RankShared *s, int curr, const uint64_t *visited, const MatrixFloat *c) {
    int best = 0; float best_d = FLT_MAX;
    const float *row = c->rows[curr];
    for (int w = 0; w < s->visited_words; w++) {
        uint64_t v = visited[w]; if (v == 0xFFFFFFFFFFFFFFFFull) continue; 
        int base = w << 6; uint64_t mask = ~v;
        if (w == s->visited_words - 1) { int bits = (s->n % 64) + 1; if (bits < 64) mask &= (1ull << bits) - 1; }
        if (w == 0) mask &= ~1ull;
        for (int i = 0; i < 64; i += 8) {
            uint8_t byte_mask = (uint8_t)(mask >> i); if (byte_mask == 0) continue; 
            __m256 v_dist = _mm256_load_ps(&row[base + i]);
            __m256i v_bits = _mm256_set1_epi32(byte_mask);
            __m256i v_pat  = _mm256_set_epi32(0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01);
            __m256i v_mask_i = _mm256_cmpeq_epi32(_mm256_and_si256(v_bits, v_pat), v_pat);
            __m256 v_final = _mm256_blendv_ps(_mm256_set1_ps(FLT_MAX), v_dist, _mm256_castsi256_ps(v_mask_i));
            __m128 v_low = _mm256_castps256_ps128(v_final); __m128 v_high = _mm256_extractf128_ps(v_final, 1);
            __m128 v_min = _mm_min_ps(v_low, v_high);
            v_min = _mm_min_ps(v_min, _mm_shuffle_ps(v_min, v_min, _MM_SHUFFLE(1, 0, 3, 2)));
            v_min = _mm_min_ps(v_min, _mm_shuffle_ps(v_min, v_min, _MM_SHUFFLE(2, 3, 0, 1)));
            float min_val = _mm_cvtss_f32(v_min);
            if (min_val < best_d) {
                uint64_t sub_mask = (uint64_t)byte_mask;
                while (sub_mask != 0) {
                    int bit = __builtin_ctzll(sub_mask); float d = row[base + i + bit];
                    if (d < best_d) { best_d = d; best = base + i + bit; }
                    sub_mask &= sub_mask - 1;
                }
            }
        }
    }
    return best;
}

static void build_ant_v2(t_thread_workspace *ws, const V2RankShared *s, int k, int cap, const MatrixFloat *c, const float *scores) {
    solution_reset(ws->sol); memset(ws->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
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
            float weights[V2_MAX_CANDS];
            for (int t = 0; t < s->cand_k; t++) {
                int node = cands[t];
                if (node > 0 && !((ws->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) { weights[t] = sc[t]; denom += sc[t]; } else weights[t] = 0.0f;
            }
            if (denom > 0.0f) {
                float thres = (float)rand01_state(&ws->rng_state) * denom, cum = 0.0f;
                for (int t = 0; t < s->cand_k; t++) { if (weights[t] <= 0.0f) continue; cum += weights[t]; if (cum >= thres) { next = cands[t]; break; } }
            }
            if (next <= 0) next = find_nearest_unvisited(s, curr, ws->visited, c);
            if (next <= 0) break;
            route_append(&ws->sol->routes[v], next); ws->visited[(unsigned)next >> 6] |= (1ull << ((unsigned)next & 63u));
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

    MatrixFloat *tau_mat = matrix_create_float(n); MatrixFloat *c_mat = matrix_create_float(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { c_mat->rows[i][j] = (float)c[i][j]; tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0; }
    V2RankShared shared; v2_shared_init(&shared, n, cand_k, c_mat, beta);
    float *score_mat = aligned_alloc(V2_ALIGNMENT, (size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    t_solution *iter_best = solution_create(k, n); *best_cost = DBL_MAX; double start_time = wall_time();
    const char *s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS"); int fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 100;

    #pragma omp parallel default(shared) proc_bind(close)
    {
        t_thread_workspace ws; v2_ws_init(&ws, k, n, shared.visited_words);
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
                ws.rng_state = make_ant_seed(seed, iter, ant_off + a);
                build_ant_v2(&ws, &shared, k, cap, c_mat, score_mat);
                double cost = solution_cost_f(ws.sol, c_mat->rows);
                if (cost < t_best_c) { t_best_c = cost; solution_copy(ws.thread_best, ws.sol); }
            }
            #pragma omp critical
            { if (t_best_c < *best_cost) { *best_cost = t_best_c; solution_copy(best_sol, ws.thread_best); } }
            #pragma omp barrier
        }
        v2_ws_free(&ws);
    }
    if (mpi_rank == 0) printf("V2 SIMD Completion. Time: %.3fs\n", wall_time() - start_time);
    matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat); solution_free(iter_best); v2_shared_free(&shared);
}

void aco_vrp_v2(int n, int k, int m, double **c, double alpha, double beta, double rho, double tau0, double q, unsigned int seed, t_solution *best_solution, double *best_cost) {
    int cap = (k > 0) ? (int)(((long long)120 * n + 100 * k - 1) / (100 * k)) : n;
    aco_vrp_v2_with_capacity(n, k, cap, m, c, alpha, beta, rho, tau0, q, seed, best_solution, best_cost);
}

void aco_vrp_v2_with_capacity(int n, int k, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double q, unsigned int seed, t_solution *best_solution, double *best_cost) {
    aco_vrp_v2_run(n, k, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, q, seed, best_solution, best_cost);
}
