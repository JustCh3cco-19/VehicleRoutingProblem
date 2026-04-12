#include "aco_v3.h"
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
#include <pthread.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif

#define V3_ALIGNMENT 64u
#define V3_MAX_CANDS 512
#define V3_EPS 1e-7f
#define V3_TEAM_SIZE 4

/* -- Matrices in FLOAT -- */
typedef struct {
    int n;
    int stride;
    float *data;
    float **rows;
} MatrixFloat;

static size_t align_up_64(size_t value) {
    size_t rem = value % V3_ALIGNMENT;
    return (rem == 0u) ? value : value + (V3_ALIGNMENT - rem);
}

static MatrixFloat *matrix_create_float(int n) {
    MatrixFloat *m = malloc(sizeof(*m));
    m->n = n;
    size_t row_bytes = (size_t)(n + 1) * sizeof(float);
    m->stride = (int)(align_up_64(row_bytes) / sizeof(float));
    m->data = aligned_alloc(V3_ALIGNMENT, (size_t)(n + 1) * (size_t)m->stride * sizeof(float));
    m->rows = malloc((size_t)(n + 1) * sizeof(float *));
    for (int i = 0; i <= n; i++) m->rows[i] = m->data + (size_t)i * (size_t)m->stride;
    memset(m->data, 0, (size_t)(n + 1) * (size_t)m->stride * sizeof(float));
    return m;
}

static void matrix_free_float(MatrixFloat *m) {
    if (!m) return;
    free(m->data); free(m->rows); free(m);
}

/* -- V3 Wait-Signal Collaborative Structures -- */

typedef struct {
    int node;
    float dist;
    char padding[V3_ALIGNMENT - sizeof(int) - sizeof(float)];
} TeamResult;

typedef struct {
    Solution *sol;
    uint64_t *visited;
    int *route_loads;
    unsigned int rng_state;
    pthread_mutex_t mutex;
    pthread_cond_t  cond_work;   
    pthread_cond_t  cond_done;   
    volatile int work_generation; 
    volatile int workers_done;    
    volatile int current_node;    
    volatile int active;          
    TeamResult results[V3_TEAM_SIZE];
} TeamWorkspace;

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
    if (k > V3_MAX_CANDS) return V3_MAX_CANDS;
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
} V3RankShared;

static void v3_shared_free(V3RankShared *s) {
    if (!s) return;
    free(s->eta_beta); free(s->cand_idx);
}

static int v3_shared_init(V3RankShared *s, int n, int cand_k, const MatrixFloat *c_mat, double beta) {
    s->n = n; s->cand_k = cand_k;
    size_t row_bytes = (size_t)s->cand_k * sizeof(float);
    s->stride = (int)(align_up_64(row_bytes) / sizeof(float));
    s->visited_words = (n / 64) + 1;
    size_t total_elems = (size_t)(n + 1) * (size_t)s->stride;
    s->cand_idx = aligned_alloc(V3_ALIGNMENT, total_elems * sizeof(int));
    s->eta_beta = aligned_alloc(V3_ALIGNMENT, total_elems * sizeof(float));
    if (!s->cand_idx || !s->eta_beta) return 0;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) {
        int nodes[V3_MAX_CANDS]; float dists[V3_MAX_CANDS];
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

static void worker_wait_loop(TeamWorkspace *tw, int team_rank, int team_size, const MatrixFloat *c, int n, int visited_words) {
    int last_gen = 0;
    while (true) {
        pthread_mutex_lock(&tw->mutex);
        while (tw->active && tw->work_generation <= last_gen) {
            pthread_cond_wait(&tw->cond_work, &tw->mutex);
        }
        if (!tw->active) { pthread_mutex_unlock(&tw->mutex); break; }
        int gen = tw->work_generation;
        int curr = tw->current_node;
        pthread_mutex_unlock(&tw->mutex);

        const float *row = c->rows[curr];
        float best_d = FLT_MAX; int best_n = 0;
        int words_per_thread = (visited_words + team_size - 1) / team_size;
        int start_w = team_rank * words_per_thread;
        int end_w = (team_rank + 1) * words_per_thread;
        if (end_w > visited_words) end_w = visited_words;
        for (int w = start_w; w < end_w; w++) {
            uint64_t v = tw->visited[w]; if (v == 0xFFFFFFFFFFFFFFFFull) continue;
            uint64_t mask = ~v; int base = w << 6;
            if (w == visited_words - 1) { int bits = (n % 64) + 1; if (bits < 64) mask &= (1ull << bits) - 1; }
            if (w == 0) mask &= ~1ull;
            while (mask != 0) {
                int bit = __builtin_ctzll(mask); int node = base + bit;
                float d = row[node]; if (d < best_d) { best_d = d; best_n = node; }
                mask &= mask - 1;
            }
        }
        tw->results[team_rank].dist = best_d; tw->results[team_rank].node = best_n;

        pthread_mutex_lock(&tw->mutex);
        tw->workers_done++;
        if (tw->workers_done == team_size - 1) { pthread_cond_signal(&tw->cond_done); }
        pthread_mutex_unlock(&tw->mutex);
        last_gen = gen;
    }
}

static void build_ant_v3_wait_signal(TeamWorkspace *tw, int team_size, const V3RankShared *s, int K, int cap, const MatrixFloat *c_mat, const float *scores) {
    solution_reset(tw->sol);
    memset(tw->visited, 0, (size_t)s->visited_words * sizeof(uint64_t));
    memset(tw->route_loads, 0, (size_t)K * sizeof(int));
    int rem = s->n;
    for (int v = 0; v < K; v++) {
        route_append(&tw->sol->routes[v], 0);
        int curr = 0;
        while (true) {
            int rem_v = K - v - 1, fut_cap = rem_v * cap;
            if (!(rem > 0 && rem > fut_cap && tw->route_loads[v] < cap)) break;
            int next = 0; float denom = 0.0f;
            const int *cands = s->cand_idx + (size_t)curr * (size_t)s->stride;
            const float *sc = scores + (size_t)curr * (size_t)s->stride;
            float weights[V3_MAX_CANDS];
            for (int t = 0; t < s->cand_k; t++) {
                int node = cands[t];
                if (node > 0 && !((tw->visited[(unsigned)node >> 6] >> ((unsigned)node & 63u)) & 1u)) { weights[t] = sc[t]; denom += sc[t]; } else weights[t] = 0.0f;
            }
            if (denom > 0.0f) {
                float thres = (float)aco_rand01_state(&tw->rng_state) * denom, cum = 0.0f;
                for (int t = 0; t < s->cand_k; t++) { if (weights[t] <= 0.0f) continue; cum += weights[t]; if (cum >= thres) { next = cands[t]; break; } }
            }
            if (next <= 0) {
                pthread_mutex_lock(&tw->mutex);
                tw->current_node = curr; tw->workers_done = 0; tw->work_generation++;
                pthread_cond_broadcast(&tw->cond_work);
                pthread_mutex_unlock(&tw->mutex);
                
                const float *row = c_mat->rows[curr];
                float best_d = FLT_MAX; int best_n = 0;
                int words_per_thread = (s->visited_words + team_size - 1) / team_size;
                int end_w = words_per_thread; if (end_w > s->visited_words) end_w = s->visited_words;
                for (int w = 0; w < end_w; w++) {
                    uint64_t v_mask = tw->visited[w]; if (v_mask == 0xFFFFFFFFFFFFFFFFull) continue;
                    uint64_t mask = ~v_mask; int base = w << 6;
                    if (w == s->visited_words - 1) { int bits = (s->n % 64) + 1; if (bits < 64) mask &= (1ull << bits) - 1; }
                    if (w == 0) mask &= ~1ull;
                    while (mask != 0) {
                        int bit = __builtin_ctzll(mask); int node = base + bit;
                        float d = row[node]; if (d < best_d) { best_d = d; best_n = node; }
                        mask &= mask - 1;
                    }
                }
                tw->results[0].dist = best_d; tw->results[0].node = best_n;
                pthread_mutex_lock(&tw->mutex);
                while (tw->workers_done < team_size - 1) pthread_cond_wait(&tw->cond_done, &tw->mutex);
                float g_best_d = tw->results[0].dist; next = tw->results[0].node;
                for (int i = 1; i < team_size; i++) { if (tw->results[i].dist < g_best_d) { g_best_d = tw->results[i].dist; next = tw->results[i].node; } }
                pthread_mutex_unlock(&tw->mutex);
            }
            if (next <= 0) break;
            route_append(&tw->sol->routes[v], next);
            tw->visited[(unsigned)next >> 6] |= (1ull << ((unsigned)next & 63u));
            tw->route_loads[v]++; rem--; curr = next;
        }
        route_append(&tw->sol->routes[v], 0);
    }
}

static void load_v3_directives(double *max_runtime_sec, int *max_stagnation_epochs, double *min_rel_improvement) {
  const char *s_timeout = getenv("ACO_SOLVER_TIMEOUT_SECONDS");
  const char *s_stagnation = getenv("ACO_SOLVER_STAGNATION_EPOCHS");
  const char *s_rel = getenv("ACO_SOLVER_MIN_REL_IMPROVEMENT");
  *max_runtime_sec = (s_timeout && *s_timeout) ? atof(s_timeout) : 0.0;
  *max_stagnation_epochs = (s_stagnation && *s_stagnation) ? atoi(s_stagnation) : 0;
  *min_rel_improvement = (s_rel && *s_rel) ? atof(s_rel) : 1e-3;
}

static int is_sig_imp(double prev_best, double new_best, double min_rel_improvement) {
  if (prev_best >= DBL_MAX || new_best >= DBL_MAX) return (new_best < prev_best);
  if (new_best >= prev_best - V3_EPS) return 0;
  double abs_gain = prev_best - new_best;
  double rel_gain = abs_gain / fmax(prev_best, (double)V3_EPS);
  return rel_gain + (double)V3_EPS >= min_rel_improvement;
}

#ifdef USE_MPI
static void sync_tau_v3(MatrixFloat *tau, int mpi_size) {
    size_t total = (size_t)(tau->n + 1) * (size_t)tau->stride;
    MPI_Allreduce(MPI_IN_PLACE, tau->data, (int)total, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float inv = 1.0f / (float)mpi_size;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i <= tau->n; i++) for (int j = 0; j <= tau->n; j++) if (i == j) tau->rows[i][j] = 0.0f; else tau->rows[i][j] *= inv;
}
#endif

void aco_vrp_v3_run(int n, int K, int cap, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_sol, double *best_cost) {
    int mpi_rank = 0, mpi_size = 1;
#ifdef USE_MPI
    int mpi_init = 0; MPI_Initialized(&mpi_init);
    if (mpi_init) { MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank); MPI_Comm_size(MPI_COMM_WORLD, &mpi_size); }
#endif
    long l3_size = get_l3_cache_size(); int cand_k = tune_candidate_k(n, l3_size);
    int total_m = (m <= 0) ? (n / 2) * mpi_size : m;
    MatrixFloat *tau_mat = matrix_create_float(n); MatrixFloat *c_mat = matrix_create_float(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) { c_mat->rows[i][j] = (float)c[i][j]; tau_mat->rows[i][j] = (i==j) ? 0.0f : (float)tau0; }
    V3RankShared shared; v3_shared_init(&shared, n, cand_k, c_mat, beta);
    float *score_mat = aligned_alloc(V3_ALIGNMENT, (size_t)(n + 1) * (size_t)shared.stride * sizeof(float));
    Solution *iter_best = solution_create(K, n); *best_cost = DBL_MAX; double start_time = wall_time();
    const char *s_fixed = getenv("ACO_SOLVER_FIXED_EPOCHS"); int fixed_epochs = (s_fixed && *s_fixed) ? atoi(s_fixed) : 100;

    #pragma omp parallel default(shared) proc_bind(close)
    {
        int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
        int team_id = tid / V3_TEAM_SIZE, team_rank = tid % V3_TEAM_SIZE, num_teams = nthreads / V3_TEAM_SIZE;
        static TeamWorkspace *team_workspaces = NULL;
        #pragma omp single
        {
            team_workspaces = calloc((size_t)num_teams, sizeof(TeamWorkspace));
            for (int i = 0; i < num_teams; i++) {
                pthread_mutex_init(&team_workspaces[i].mutex, NULL);
                pthread_cond_init(&team_workspaces[i].cond_work, NULL);
                pthread_cond_init(&team_workspaces[i].cond_done, NULL);
                team_workspaces[i].active = 1;
            }
        }
        TeamWorkspace *tw = &team_workspaces[team_id];
        if (team_rank == 0) {
            tw->sol = solution_create(K, n);
            tw->visited = aligned_alloc(V3_ALIGNMENT, (size_t)shared.visited_words * sizeof(uint64_t));
            tw->route_loads = calloc((size_t)K, sizeof(int));
        }
        #pragma omp barrier

        if (team_rank != 0) {
            worker_wait_loop(tw, team_rank, V3_TEAM_SIZE, c_mat, n, shared.visited_words);
        } else {
            double max_runtime_sec; int max_stagnation_epochs; double min_rel_improvement;
            load_v3_directives(&max_runtime_sec, &max_stagnation_epochs, &min_rel_improvement);
            int iter_since_best = 0;
            for (int iter = 0; iter < fixed_epochs; iter++) {
                #pragma omp for schedule(static) nowait
                for (int i = 0; i <= n; i++) {
                    int *cands = shared.cand_idx + (size_t)i * (size_t)shared.stride; float *etas = shared.eta_beta + (size_t)i * (size_t)shared.stride;
                    float *sc = score_mat + (size_t)i * (size_t)shared.stride; const float *tau_row = tau_mat->rows[i];
                    for (int t = 0; t < shared.cand_k; t++) { int node = cands[t]; sc[t] = (node > 0) ? (fast_powf(tau_row[node], (float)alpha) * etas[t]) : 0.0f; }
                }
                #pragma omp barrier 
                double t_team_best = DBL_MAX; int local_m = total_m / mpi_size;
                for (int a = team_id; a < local_m; a += num_teams) {
                    tw->rng_state = aco_make_ant_seed(seed, iter, a);
                    build_ant_v3_wait_signal(tw, V3_TEAM_SIZE, &shared, K, cap, c_mat, score_mat);
                    double cost = solution_cost(tw->sol, c);
                    if (cost < t_team_best) { t_team_best = cost; solution_copy(iter_best, tw->sol); }
                }
                double prev_best = *best_cost;
                #pragma omp critical
                { if (t_team_best < *best_cost) { *best_cost = t_team_best; solution_copy(best_sol, iter_best); } }
                if (is_sig_imp(prev_best, *best_cost, min_rel_improvement)) iter_since_best = 0; else iter_since_best++;
                #pragma omp barrier
                float rho_f = (float)rho;
                #pragma omp for collapse(2) schedule(static)
                for (int i = 0; i <= n; i++) for (int j = 0; j <= n; j++) if (i != j) tau_mat->rows[i][j] *= (1.0f - rho_f);
                float dep = (float)(0.3 * Q / fmax(*best_cost, 1e-9));
                #pragma omp for schedule(static)
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
                    #pragma omp for schedule(static)
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
#ifdef USE_MPI
                #pragma omp master
                if (mpi_size > 1) sync_tau_v3(tau_mat, mpi_size);
                #pragma omp barrier
#endif
            }
            pthread_mutex_lock(&tw->mutex);
            tw->active = 0; pthread_cond_broadcast(&tw->cond_work);
            pthread_mutex_unlock(&tw->mutex);
            solution_free(tw->sol); free(tw->visited); free(tw->route_loads);
        }
    }
    if (mpi_rank == 0) printf("V3 Final Wait-Signal Completion. Best: %.3f. Time: %.3fs\n", *best_cost, wall_time() - start_time);
    matrix_free_float(tau_mat); matrix_free_float(c_mat); free(score_mat); solution_free(iter_best); v3_shared_free(&shared);
}

void aco_vrp_v3(int n, int K, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    int cap = (K > 0) ? (int)(((long long)120 * n + 100 * K - 1) / (100 * K)) : n;
    aco_vrp_v3_with_capacity(n, K, cap, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}

void aco_vrp_v3_with_capacity(int n, int K, int vehicle_capacity_customers, int m, double **c, double alpha, double beta, double rho, double tau0, double Q, unsigned int seed, Solution *best_solution, double *best_cost) {
    aco_vrp_v3_run(n, K, vehicle_capacity_customers, m, c, alpha, beta, rho, tau0, Q, seed, best_solution, best_cost);
}
