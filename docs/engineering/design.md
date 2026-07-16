# Architectural Design: Hybrid OpenMP-MPI ACO VRP Solver (v2 - Candidate List Optimized)

## Overview
This document outlines the architectural principles for the high-performance version of the Ant Colony Optimization (ACO) solver for the Vehicle Routing Problem (VRP). The focus is on **Hardware-Aware** design, maximizing cache hit rates, and enabling SIMD vectorization.

## 1. Core Parallelism Strategy: "Globally Sequential, Locally Parallel"
*   **MPI Layer**: Coarse-grained distribution of ants across nodes. Global synchronization of the pheromone matrix ($\tau$) at the end of each epoch using non-blocking primitives (`MPI_Iallreduce`) where possible to hide latency.
*   **OpenMP Layer**: Each thread executes multiple ants sequentially from a shared pool. Work is distributed using dynamic scheduling to eliminate load imbalance.
*   **NUMA Awareness**: Use `OMP_PROC_BIND=close` to pin threads to cores, ensuring data in L1/L2 caches remains local to the executing ant.

## 2. Shared Data Structures (Per-Rank)
To reduce memory footprint and redundant calculations, the following matrices are shared among all threads in a rank:

| Structure | Dimension | Type | Update Frequency |
| :--- | :--- | :--- | :--- |
| **Pheromone ($\tau$)** | $N \times N$ | `double` | Per Epoch (MPI Sync) |
| **Candidate Indices** | $N \times K$ | `int` | Once (Startup) |
| **Heuristic Power ($\eta^\beta$)** | $N \times K$ | `float` | Once (Startup) |
| **Score Matrix ($S$)** | $N \times K$ | `float` | Per Epoch (Local Update) |

*   **Mixed Precision**: Using `float` (32-bit) for the $N \times K$ Score Matrix raddoppia il throughput SIMD (16 valori per ciclo AVX-512) e dimezza il bandwidth di memoria rispetto a `double`.
*   **Cache-Line Alignment**: Ogni riga delle matrici $N \times K$ deve essere allineata a 64 byte (padding) per evitare letture parziali e massimizzare il prefetching.

## 3. Persistent Thread Workspace (Per-Thread)
To minimize allocation overhead and maximize cache reuse, each OpenMP thread manages a **Persistent Workspace** allocated once per epoch:

*   **Lifecycle**:
    1.  **Allocation**: At the start of the `#pragma omp parallel` region, each thread allocates its own workspace structures.
    2.  **Execution Loop**: The thread repeatedly "pulls" an ant ID from the pool and builds a solution using its private workspace.
    3.  **Lightweight Reset**: Between ant executions, only the bitmask and solution buffers are reset (zero-cost `memset`), while the memory blocks remain resident in L1 cache.
    4.  **Deallocation**: Memory is freed only at the end of the epoch.

*   **Workspace Components**:
    *   **Visited Bitmask**: Array of `uint64_t` for $O(1)$ availability checks and SIMD-friendly masking.
    *   **Solution Buffer**: Flat contiguous array for the current ant's route set.
    *   **RNG State**: Private xorshift state initialized with a unique seed derived from the thread ID and epoch.

## 4. Execution Model & Load Balancing
The solver employs a **Dynamic Task Pool** to handle the stochastic variability of VRP construction:

*   **Dynamic Scheduling**: The ant loop uses `#pragma omp for schedule(guided, 1)`. Threads that finish an ant execution early (e.g., due to vehicle capacity limits) immediately start the next available ant, ensuring 100% core utilization.
*   **Population Scaling**: The total number of ants $m$ is scaled to $m \approx 4C \dots 8C$ (where $C$ is the total core count). This "oversampling" provides enough work units for the `guided` scheduler to effectively balance the load across heterogeneous ant construction times.

## 5. Advanced Selection & Search Logic
*   **Dense SIMD Scan**: Selection performs a sequential scan of row $S[current]$, masked by the private bitmask. This leverages the hardware prefetcher and enables branchless SIMD reduction.
*   **Candidate-First Local Search**: 2-opt and relocate moves are restricted to the $K$-nearest neighbors, reducing complexity from $O(N^2)$ to $O(N \cdot K)$.
*   **Max-Min Ant System (MMAS)**: Enforces $[\tau_{min}, \tau_{max}]$ limits on pheromones to prevent premature convergence and encourage exploration.

## 6. Implementation Roadmap
1.  **Memory Layout**: 64-byte aligned row-major structures.
2.  **Thread Kernels**: Implementation of the persistent workspace and the `guided` execution loop.
3.  **SIMD Kernels**: AVX2/AVX-512 routines for score updates and roulette wheel selection.
