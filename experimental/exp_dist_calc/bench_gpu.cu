#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define ITERATIONS 20
#define THREADS_PER_BLOCK 256

__global__ void kernel_global_memory(const float *costs, float *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int i = idx;
        int j = (idx + 1) % n; 
        results[idx] = costs[i * n + j];
    }
}

__global__ void kernel_compute_live(const float2 *coords, float *results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int i = idx;
        int j = (idx + 1) % n;
        float2 p1 = coords[i];
        float2 p2 = coords[j];
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        results[idx] = sqrtf(dx * dx + dy * dy);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);

    size_t matrix_size = (size_t)n * n * sizeof(float);
    size_t coords_size = n * sizeof(float2);
    size_t results_size = n * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    float *d_costs = NULL;
    float2 *d_coords;
    float *d_results;
    cudaMalloc(&d_coords, coords_size);
    cudaMalloc(&d_results, results_size);

    float ms_mem = -1.0f;
    // Tenta di allocare la matrice solo se c'è abbastanza memoria
    if (matrix_size < free_mem * 0.9) {
        if (cudaMalloc(&d_costs, matrix_size) == cudaSuccess) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

            cudaEventRecord(start);
            for (int it = 0; it < ITERATIONS; ++it) {
                kernel_global_memory<<<blocks, THREADS_PER_BLOCK>>>(d_costs, d_results, n);
            }
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&ms_mem, start, stop);
            ms_mem /= ITERATIONS;
            cudaFree(d_costs);
        }
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start);
    for (int it = 0; it < ITERATIONS; ++it) {
        kernel_compute_live<<<blocks, THREADS_PER_BLOCK>>>(d_coords, d_results, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_comp = 0;
    cudaEventElapsedTime(&ms_comp, start, stop);
    ms_comp /= ITERATIONS;

    // Output: N, Memory_ms, Compute_ms, Matrix_MB, Coords_MB
    printf("%d,%.6f,%.6f,%.2f,%.4f\n", 
           n, ms_mem, ms_comp, (float)matrix_size/(1024*1024), (float)coords_size/(1024*1024));

    cudaFree(d_coords);
    cudaFree(d_results);
    return 0;
}
