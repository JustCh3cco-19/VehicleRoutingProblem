# Scaling Report

Detailed narrative and graph-by-graph explanation: `reports/seq_vs_parallel_report.md`

- Generated: 2026-02-23 00:08:12
- Host: FrancescoZompanti
- System: Linux 6.8.0-100-generic
- Python: 3.10.12

## Benchmark Configuration

- Repeats (measured): 3
- Warmups (discarded): 1
- Strong scaling workload: n=90, K=8, m=240, T=110
- Weak scaling base ants/worker: 48, n=90, K=8, T=110
- OpenMP threads tested: 1,2,4,8
- MPI ranks tested: 1,2,4
- MPI threads per rank: 2
- MPI sync_every: 1
- CUDA benchmark included: yes

## Aggregated Results

### cuda_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 240 | 0.235407 | 0.001843 | 1.000 | 1.000 |

### mpi_omp_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | 240 | 4.327658 | 0.009631 | 1.000 | 1.000 |
| 4 | 2 | 2 | 240 | 2.165260 | 0.006428 | 1.999 | 0.999 |
| 8 | 4 | 2 | 240 | 1.367016 | 0.037714 | 3.166 | 0.791 |

### mpi_omp_weak

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 2 | 96 | 1.784288 | 0.001856 | 1.000 | 1.000 |
| 4 | 2 | 2 | 192 | 1.503849 | 0.008193 | 1.186 | 0.593 |
| 8 | 4 | 2 | 384 | 2.073381 | 0.061804 | 0.861 | 0.215 |

### omp_strong

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 240 | 5.469804 | 0.029876 | 1.000 | 1.000 |
| 2 | 1 | 2 | 240 | 2.781035 | 0.017064 | 1.967 | 0.983 |
| 4 | 1 | 4 | 240 | 1.547654 | 0.114238 | 3.534 | 0.884 |
| 8 | 1 | 8 | 240 | 1.320340 | 0.049455 | 4.143 | 0.518 |

### seq_strong_baseline

| Scale | Ranks | Threads/Rank | m | Mean Time (s) | Std Dev (s) | Speedup | Efficiency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 240 | 5.443026 | 0.019064 | 1.000 | 1.000 |

## Performance Plots

### Best backend comparison

![Best backend comparison](plots/backend_comparison.png)

### Seq vs parallel speedup

![Seq vs parallel speedup](plots/seq_vs_parallel_speedup.png)

### Strong scaling runtime

![Strong scaling runtime](plots/strong_runtime.png)

### Strong scaling speedup

![Strong scaling speedup](plots/strong_speedup.png)

### Strong scaling efficiency

![Strong scaling efficiency](plots/strong_efficiency.png)

### Weak scaling runtime

![Weak scaling runtime](plots/weak_runtime.png)

### Weak scaling time-per-ant

![Weak scaling time per ant](plots/weak_time_per_ant.png)

## Failures

No benchmark failures.

Raw samples are available in `scaling_results_raw.csv`; aggregated values are in `scaling_summary.csv`.
