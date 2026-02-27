# Scaling Report

- Generated: 2026-02-24 08:53:15
- Host: FrancescoZompanti
- System: Linux 6.8.0-100-generic
- Python: 3.10.12

## Benchmark Configuration

- Repeats (measured): 3
- Warmups (discarded): 1
- Strong scaling workload: n=100, K=8, m=256, T=120
- Weak scaling base ants/worker: 64, n=100, K=8, T=120
- OpenMP threads tested: 1,2,4,8
- MPI ranks tested: 1,2,4
- MPI threads per rank: 2
- MPI sync_every: 1
- CUDA benchmark included: yes

## Aggregated Results

### cuda_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 256 | 0.283988 | 0.001980 | 1.000 | 1.000 |

### mpi_omp_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | 256 | 6.067358 | 0.006624 | 1.000 | 1.000 |
| 4 | 2 | 2 | 256 | 2.905125 | 0.032818 | 2.089 | 1.044 |
| 8 | 4 | 2 | 256 | 1.557871 | 0.068749 | 3.895 | 0.974 |

### mpi_omp_weak

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | 128 | 2.570577 | 0.023672 | 1.000 | 1.000 |
| 4 | 2 | 2 | 256 | 2.897283 | 0.012221 | 0.887 | 0.444 |
| 8 | 4 | 2 | 512 | 2.746805 | 0.010510 | 0.936 | 0.234 |

### omp_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 256 | 7.795832 | 0.019250 | 1.000 | 1.000 |
| 2 | 1 | 2 | 256 | 3.999373 | 0.011371 | 1.949 | 0.975 |
| 4 | 1 | 4 | 256 | 2.104208 | 0.164250 | 3.705 | 0.926 |
| 8 | 1 | 8 | 256 | 1.739070 | 0.024444 | 4.483 | 0.560 |

### seq_strong_baseline

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 256 | 7.854864 | 0.109246 | 1.000 | 1.000 |

## Failures

No benchmark failures.

Raw samples are available in `scaling_results_raw.csv`; aggregated values are in `scaling_summary.csv`.
