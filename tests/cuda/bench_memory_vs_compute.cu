#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 10000
#define ITERATIONS 100
#define THREADS_PER_BLOCK 256

// Kernel A: Accesso alla Global Memory (Distanza Pre-calcolata)
__global__ void kernel_global_memory(const float *costs, float *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Leggiamo la distanza i -> j (j è fisso per semplicità, ma simula un accesso casuale)
        // In un vero ACO j cambierebbe, rendendo l'accesso non-coalesced.
        int i = idx;
        int j = (idx + 500) % n; 
        results[idx] = costs[i * n + j];
    }
}

// Kernel B: Calcolo On-the-fly (Distanza dalle Coordinate)
__global__ void kernel_compute_live(const float2 *coords, float *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int i = idx;
        int j = (idx + 500) % n;
        float2 p1 = coords[i];
        float2 p2 = coords[j];
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        results[idx] = sqrtf(dx * dx + dy * dy);
    }
}

int main() {
    size_t matrix_size = (size_t)N * N * sizeof(float);
    size_t coords_size = N * sizeof(float2);
    size_t results_size = N * sizeof(float);

    printf("Benchmarking for N = %d, Iterations = %d\n", N, ITERATIONS);
    printf("Memory: t_matrix = %.2f MB, Coordinates = %.2f MB\n", 
           (float)matrix_size / (1024*1024), (float)coords_size / (1024*1024));

    float *h_costs = (float*)malloc(matrix_size);
    float2 *h_coords = (float2*)malloc(coords_size);
    float *h_results = (float*)malloc(results_size);

    for (int i = 0; i < N * N; ++i) h_costs[i] = (float)i;
    for (int i = 0; i < N; ++i) {
        h_coords[i].x = (float)i;
        h_coords[i].y = (float)(i * 2);
    }

    float *d_costs, *d_results;
    float2 *d_coords;
    cudaMalloc(&d_costs, matrix_size);
    cudaMalloc(&d_coords, coords_size);
    cudaMalloc(&d_results, results_size);

    cudaMemcpy(d_costs, h_costs, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coords, h_coords, coords_size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // --- Benchmark Kernel A (Memory) ---
    cudaEventRecord(start);
    for (int it = 0; it < ITERATIONS; ++it) {
        kernel_global_memory<<<blocks, THREADS_PER_BLOCK>>>(d_costs, d_results, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_mem = 0;
    cudaEventElapsedTime(&ms_mem, start, stop);
    printf("Kernel A (Global Memory): %.4f ms total, %.4f ms per iter\n", ms_mem, ms_mem / ITERATIONS);

    // --- Benchmark Kernel B (Compute) ---
    cudaEventRecord(start);
    for (int it = 0; it < ITERATIONS; ++it) {
        kernel_compute_live<<<blocks, THREADS_PER_BLOCK>>>(d_coords, d_results, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_comp = 0;
    cudaEventElapsedTime(&ms_comp, start, stop);
    printf("Kernel B (Live Compute):  %.4f ms total, %.4f ms per iter\n", ms_comp, ms_comp / ITERATIONS);

    printf("Speedup factor: %.2fx\n", ms_mem / ms_comp);

    cudaFree(d_costs);
    cudaFree(d_coords);
    cudaFree(d_results);
    free(h_costs);
    free(h_coords);
    free(h_results);
    
    return 0;
}
