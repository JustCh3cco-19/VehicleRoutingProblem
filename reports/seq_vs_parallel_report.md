# Seq vs Parallel Report (Detailed)

## 1) Scope and dataset

This report compares the sequential baseline against all parallel variants using the latest benchmark outputs:

- Source files: `reports/scaling_summary.csv`, `reports/scaling_results_raw.csv`
- Workload (strong scaling): `n=90`, `K=8`, `m=240`, `T=110`
- Repetitions: 3 measured runs + 1 warmup (discarded)

## 2) Metrics used

- `Mean Time (s)`: average runtime over measured repetitions
- `Std Dev (s)`: runtime variability
- `Speedup vs Seq`: `T_seq / T_cfg`
- `Gain vs Seq (%)`: `(1 - T_cfg / T_seq) * 100`

With current data, the sequential baseline is:

- `T_seq = 5.443026 s`

## 3) Direct Seq vs Parallel comparison

| Configuration | Mean Time (s) | Std Dev (s) | Speedup vs Seq | Gain vs Seq (%) |
| --- | ---: | ---: | ---: | ---: |
| SEQ (1x1) | 5.443026 | 0.019064 | 1.000x | 0.00 |
| OMP (1 thread) | 5.469804 | 0.029876 | 0.995x | -0.49 |
| OMP (2 threads) | 2.781035 | 0.017064 | 1.957x | 48.91 |
| OMP (4 threads) | 1.547654 | 0.114238 | 3.517x | 71.57 |
| OMP (8 threads) | 1.320340 | 0.049455 | 4.122x | 75.74 |
| MPI+OMP (1 rank x 2 threads) | 4.327658 | 0.009631 | 1.258x | 20.49 |
| MPI+OMP (2 rank x 2 threads) | 2.165260 | 0.006428 | 2.514x | 60.22 |
| MPI+OMP (4 rank x 2 threads) | 1.367016 | 0.037714 | 3.982x | 74.89 |
| CUDA (1 GPU) | 0.235407 | 0.001843 | 23.122x | 95.68 |

## 4) Main takeaways

- Best OMP runtime: **1.320340 s** at `8 threads`.
- Best MPI+OMP runtime: **1.367016 s** at `4x2` workers.
- CUDA runtime: **0.235407 s** (single GPU).
- Best overall runtime in this campaign: **CUDA** (`0.235407 s`).
- Note: `Speedup` values in `reports/scaling_summary.csv` are experiment-relative; this report uses **Seq-relative speedup** for direct comparison.

## 5) Graph-by-graph explanation

## 5.0 `reports/plots/seq_vs_parallel_speedup.png`

What this graph shows:

- Direct speedup against the sequential baseline for each strong-scaling configuration.
- Y-axis is `Speedup vs SEQ = T_seq / T_cfg`.

How to read it:

- `1.0x` means equal to SEQ.
- Above `1.0x` means faster than SEQ.
- Below `1.0x` means slower than SEQ.

What your data says:

- Best speedup is OMP (`4.12x`).
- Best MPI+OMP speedup is `3.98x`.
- CUDA speedup vs SEQ is `23.12x`.

Why this happens:

- CPU parallel versions reduce ant-construction wall time effectively.
- CUDA now benefits from reduced CPU↔GPU transfers and GPU-side best/deposit logic.

## 5.1 `reports/plots/backend_comparison.png`

What this graph shows:

- One bar per backend family (SEQ, best OMP, best MPI+OMP, CUDA).
- Y-axis is mean runtime in seconds.

How to read it:

- Lower bar means better performance.
- It is a direct visual summary for final “which backend wins”.

What your data says:

- SEQ: `5.443 s`
- Best OMP: `1.320 s`
- Best MPI+OMP: `1.367 s`
- CUDA: `0.235 s`

Why this happens:

- This implementation currently benefits from optimized CUDA orchestration and reduced transfer overhead.
- Backend ranking is workload- and implementation-dependent, not universal.

## 5.2 `reports/plots/strong_runtime.png`

What this graph shows:

- Runtime versus parallel scale for strong scaling.
- Includes OMP and MPI+OMP with error bars (std dev).
- Includes a horizontal SEQ reference line.

How to read it:

- For strong scaling, ideal trend is decreasing runtime as workers increase.
- Error bars indicate run-to-run stability.

What your data says:

- OMP: `5.470 -> 1.320 s` from scale 1 to 8.
- MPI+OMP: `4.328 -> 1.367 s` from scale 2 to 8.

Why this happens:

- Parallel work is reduced with more workers, but synchronization/communication eventually limits ideal linear behavior.

## 5.3 `reports/plots/strong_speedup.png`

What this graph shows:

- Measured speedup curves against ideal speedup lines.
- Speedup in this plot is relative to each experiment baseline used by the benchmark script.

How to read it:

- Ideal line = perfect linear scaling.
- Distance below ideal quantifies overhead.

What your data says:

- OMP reaches `4.143x` (experiment-relative) at max tested scale.
- MPI+OMP reaches `3.166x` (experiment-relative) at max tested scale.

Why this happens:

- Synchronization, memory bandwidth, and MPI collectives prevent perfect linear scaling at high concurrency.

## 5.4 `reports/plots/strong_efficiency.png`

What this graph shows:

- Efficiency versus scale (normalized speedup per worker).

How to read it:

- `1.0` means ideal scaling efficiency.
- Lower values mean rising overhead per added worker.

What your data says:

- OMP efficiency at max scale: `0.518`.
- MPI+OMP efficiency at max scale: `0.791`.

Why this happens:

- Communication/synchronization and non-parallel sections grow in relative weight as scale increases.

## 5.5 `reports/plots/weak_runtime.png`

What this graph shows:

- Weak scaling runtime for MPI+OMP, with problem growth (`m`) on the secondary axis.

How to read it:

- In ideal weak scaling, runtime remains approximately constant while total problem size grows.

What your data says:

- Runtime trend: `1.784 -> 2.073 s` as scale increases.

Why this happens:

- Weak scaling degrades when global synchronization cost grows faster than per-worker compute savings.

## 5.6 `reports/plots/weak_time_per_ant.png`

What this graph shows:

- Runtime normalized by number of ants (`seconds/ant`) in weak scaling.

How to read it:

- Lower values indicate better per-unit processing efficiency.

What your data says:

- Time/ant changes from `0.018586` to `0.005399` s/ant across tested scales.

Why this happens:

- Local ant construction scales reasonably, but global MPI coordination still impacts total weak-scaling runtime.

## 6) Practical conclusion

For this implementation and this machine:

- For fastest CPU-only execution, use the best tested OMP configuration.
- MPI+OMP remains a strong option when distributed execution is required.
- CUDA is now functionally stable and significantly improved; further gains require deeper kernel-level optimization and algorithm/device co-design.
