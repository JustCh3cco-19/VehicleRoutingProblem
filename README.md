# VehicleRoutingProblem

ACO (Ant Colony Optimization) solver for the **Capacitated Vehicle Routing Problem (CVRP)** with three C/CUDA backends:
- `seq`: sequential CPU baseline
- `openmp-mpi`: hybrid shared/distributed-memory backend
- `cuda`: single-device GPU backend

The project is designed for reproducible HPC experiments on clusters, with a `make` + `sbatch` pipeline and CSV/manifest result collection.

## 1. Problem

Given a depot and `n` customers with unit demand, CVRP requires building `K` routes that:
- start/end at the depot,
- satisfy vehicle capacity constraints,
- minimize total travel cost.

Instances are provided in `.vrp` (TSPLIB-like) format and generated/filtered in `instances/test_aligned`.

## 2. Algorithmic Approach

The core method is ACO: each ant builds a solution using pheromone information (`tau`) + heuristic information (`eta`), then pheromones are updated via evaporation/deposition.

Main architectural choices (aligned with internal technical docs):
- **V2 OpenMP+MPI** as the stable parallel baseline (persistent threading, parallel updates, optimized MPI sync).
- **V3 excluded** from main runs: collaborative intra-ant parallelism introduces synchronization overhead that is not beneficial on general-purpose CPUs.

Internal references:
- [RoadmapOpenMP_MPI.md](docs/RoadmapOpenMP_MPI.md)
- [v3_failure_analysis.md](docs/v3_failure_analysis.md)

## 3. Repository Structure

Solver code:
- `src/seq/`
- `src/openmp-mpi/`
- `src/cuda/`
- `src/common/`

Public headers:
- `include/`

Operational tooling:
- `tools/makefile/` make modules
- `tools/bash/` shell launchers (`solve_*`)
- `tools/python/` generators/analysis/report tools
- `tools/batch/` Slurm job scripts

Technical documentation:
- `docs/`

## 4. Requirements

Minimum:
- `make`, `gcc`
- `python3`

For MPI:
- `mpicc`, `mpirun` or `srun`

For CUDA:
- `nvcc` + NVIDIA GPU

## 5. Build

```bash
make seq
make openmp_mpi
make cuda
```

Full build:

```bash
make all
```

## 6. Instance Generation

```bash
make generate_problems
```

Main outputs:
- `instances/test_aligned/*.vrp`
- `instances/test_aligned/manifest.csv`
- `instances/test_aligned/manifest_openmp_mpi.csv`
- `instances/test_aligned/manifest_cuda.csv`

## 7. Solver Execution

Standard manifest-based runs:

```bash
make solve_seq
make solve_mpi
make solve_cuda
```

Useful variables (passed via `--make-args` in batch jobs):
- `SOLVE_CLIENTS`
- `SOLVE_*_REPEATS`
- `SOLVE_*_RUNTIME_S`
- `SOLVE_*_STAGNATION_EPOCHS`
- `SOLVE_*_MIN_REL_IMPROVEMENT`

## 8. Experiments (Strong/Weak/Quality)

Main targets:

```bash
make exp_strong_openmp
make exp_strong_mpi
make exp_strong_hybrid
make exp_weak_openmp
make exp_weak_mpi
make exp_weak_hybrid
```

Practical campaign pipeline:
- details: [practical_experiment_campaign.md](docs/practical_experiment_campaign.md)
- aggregated data: `merged_by_run_backend/*.csv`
- summary report: [REPORT.md](merged_by_run_backend/REPORT.md)

## 9. Results Summary (Current Campaign)

Source: [merged_by_run_backend/REPORT.md](merged_by_run_backend/REPORT.md)

Coverage (`status=ok` rows):
- `seq_performance`: 7
- `cuda_performance`: 10
- `openmp_strong`: 14
- `mpi_strong`: 12
- `hybrid_strong`: 13
- `openmp_weak`: 6
- `mpi_weak`: 6
- `hybrid_weak`: 3
- `seq_quality`: 11
- `mpi_quality`: 16
- `cuda_quality`: 30

Key findings:
- **OpenMP strong:** best average tradeoff at `4` threads (with variation at largest size).
- **MPI strong:** best average configuration at `2` ranks.
- **Hybrid strong:** best average configuration `4x4` (ranks x threads) on available data.
- **CUDA vs SEQ:** CUDA is faster on all overlapping sizes, observed speedup ~`2.87x`–`111.21x`.

Generated plots:
- `merged_by_run_backend/plots/strong_*`
- `merged_by_run_backend/plots/weak_*`
- `merged_by_run_backend/plots/seq_vs_cuda_elapsed.png`
- `merged_by_run_backend/plots/quality_*`s

Output:
- `merged_by_run_backend/plots/*.png`
- `merged_by_run_backend/plots/README.md`

## 11. Methodological Notes

- Large-size results include single-run points; variance may be underestimated there.
- For final academic tables, use medians and report standard deviation when `repeats > 1`.
- The `main` branch objective is keeping solver code stable; tuning/campaign work should be done through orchestration (`make`, batch, scripts), not continuous core-solver rewrites.

## 12. License and Usage

Use this repository for educational/research purposes and comparative CVRP benchmarking across CPU/MPI/CUDA backends.
