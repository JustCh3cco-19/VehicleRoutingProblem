#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define ITERATIONS 5

typedef struct {
    float x, y;
} Point;

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);

    Point *coords = (Point*)malloc(n * sizeof(Point));
    float *results = (float*)malloc(n * sizeof(float));
    
    if (!coords || !results) {
        fprintf(stderr, "Failed to allocate basic arrays\n");
        return 1;
    }

    // Inizializzazione
    for(int i = 0; i < n; ++i) {
        coords[i].x = (float)i;
        coords[i].y = (float)(i * 2);
    }

    // Benchmark Compute
    double start_comp = get_time_ms();
    for(int it = 0; it < ITERATIONS; ++it) {
        #pragma omp parallel for
        for(int i = 0; i < n; ++i) {
            int j = (i + 1) % n;
            float dx = coords[i].x - coords[j].x;
            float dy = coords[i].y - coords[j].y;
            results[i] = sqrtf(dx*dx + dy*dy);
        }
    }
    double end_comp = get_time_ms();
    double ms_comp = (end_comp - start_comp) / ITERATIONS;

    // Benchmark Memory
    double ms_mem_val = -1.0;
    size_t matrix_size = (size_t)n * n * sizeof(float);
    
    // Limite di 8GB per evitare swap
    if (matrix_size < 8ULL * 1024 * 1024 * 1024) {
        float *costs = (float*)malloc(matrix_size);
        if (costs) {
            double start_mem = get_time_ms();
            for(int it = 0; it < ITERATIONS; ++it) {
                #pragma omp parallel for
                for(int i = 0; i < n; ++i) {
                    int j = (i + 1) % n;
                    results[i] = costs[(size_t)i * n + j];
                }
            }
            double end_mem = get_time_ms();
            ms_mem_val = (end_mem - start_mem) / ITERATIONS;
            free(costs);
        }
    }

    // Output: N, Memory_ms, Compute_ms
    printf("%d,%.6f,%.6f\n", n, ms_mem_val, ms_comp);

    free(coords);
    free(results);

    return 0;
}
