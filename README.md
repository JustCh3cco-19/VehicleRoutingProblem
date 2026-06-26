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

Instances are provided in `.vrp` (TSPLIB-like) format and generated/filtered in `instances/generated_benchmark`.

## 2. Algorithmic Approach

The core method is ACO: each ant builds a solution using pheromone information (`tau`) + heuristic information (`eta`), then pheromones are updated via evaporation/deposition.

Main architectural choices (aligned with internal technical docs):
- **V2 OpenMP+MPI** as the stable parallel baseline (persistent threading, parallel updates, optimized MPI sync).
- **V3 excluded** from main runs: collaborative intra-ant parallelism introduces synchronization overhead that is not beneficial on general-purpose CPUs.

Internal references:
- [RoadmapOpenMP_MPI.md](docs/engineering/RoadmapOpenMP_MPI.md)
- [v3_failure_analysis.md](docs/engineering/v3_failure_analysis.md)

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
- [architecture.md](docs/engineering/architecture.md)
- [instance_generation.md](docs/usage/instance_generation.md)
- [practical_experiment_campaign.md](docs/usage/practical_experiment_campaign.md)
- [implementation_improvements.md](docs/engineering/implementation_improvements.md)
- [structural_improvements.md](docs/engineering/structural_improvements.md)

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
- `instances/generated_benchmark/*.vrp`
- `instances/generated_benchmark/manifest.csv`
- `instances/generated_benchmark/manifest_openmp_mpi.csv`
- `instances/generated_benchmark/manifest_cuda.csv`

Detailed generation guide:
- [instance_generation.md](docs/usage/instance_generation.md)

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
- `SOLVE_*_MIN_REL_IMPROVEMENT` (percentuale: `0.1` = 0.1%, `10` = 10%)

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
- details: [practical_experiment_campaign.md](docs/usage/practical_experiment_campaign.md)
- generated data: `results/practical_campaign/<tag>/...`
- generated summaries: produced by `tools/python/summarize_practical_experiments.py`

## 9. Results And Reports

Generated benchmark outputs should go under `results/`, which is ignored by git
except for [results/README.md](results/README.md).

Versioned reports live under:
- [docs/reports/](docs/reports/)

## 10. Plot Reproduction

Regenerate plots from aggregated CSV files:

```bash
python3 tools/python/plot_merged_by_run_backend.py
```

Typical generated output should be stored under `results/`.

## 11. Methodological Notes

- Large-size results include single-run points; variance may be underestimated there.
- For final academic tables, use medians and report standard deviation when `repeats > 1`.
- The `main` branch objective is keeping solver code stable; tuning/campaign work should be done through orchestration (`make`, batch, scripts), not continuous core-solver rewrites.

## 12. License and Usage

Use this repository for educational/research purposes and comparative CVRP benchmarking across CPU/MPI/CUDA backends.
