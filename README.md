# VehicleRoutingProblem

Ant Colony Optimization (ACO) implementations for the Capacitated Vehicle
Routing Problem (CVRP), with sequential CPU, OpenMP, MPI, hybrid OpenMP+MPI,
and CUDA execution modes. The repository also contains instance generators,
manifest-driven experiment runners, Slurm submission scripts, result-analysis
tools, and an academic paper.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/`, `include/` | Solver implementations and shared C/CUDA interfaces |
| `instances/` | CVRP instances and solver manifests |
| `tools/` | Make modules, shell runners, Python analysis, and Slurm scripts |
| `results/` | Tracked campaign data and default location for new runs |
| `docs/paper/` | LaTeX paper, generated summaries, and figures |
| `docs/` | Project documentation |
| `experimental/` | Historical prototypes and standalone experiments |

See [Repository structure](docs/repository_structure.md) for the detailed map.

## Build

The CPU build requires GCC; the parallel build also requires an MPI compiler;
the CUDA build requires `nvcc` and an NVIDIA CUDA toolchain.

```bash
make seq          # aco_vrp_seq.out
make openmp_mpi   # aco_vrp_openmp_mpi.out
make cuda         # aco_vrp_cuda.out
```

`make all` builds all three binaries and therefore requires every toolchain.
Python dependencies are listed in `requirements.txt` and `pyproject.toml`.
See [Build guide](docs/build.md) for setup and configurable build flags.

## Implementations

| Mode | Executable | Configuration |
| --- | --- | --- |
| Sequential C | `aco_vrp_seq.out` | Single process |
| OpenMP | `aco_vrp_openmp_mpi.out` | One MPI rank, multiple OpenMP threads |
| MPI | `aco_vrp_openmp_mpi.out` | Multiple ranks, one OpenMP thread per rank |
| OpenMP+MPI | `aco_vrp_openmp_mpi.out` | Multiple ranks and threads per rank |
| CUDA | `aco_vrp_cuda.out` | One CUDA device |

The parallel CPU modes are configurations of one hybrid implementation; there
are no separate OpenMP-only or MPI-only executables. Technical details are in
[Implementations](docs/implementations.md).

## Run

Generate or refresh the manifest-driven instance set:

```bash
make generate_problems
```

Run one instance directly:

```bash
ACO_SOLVER_STAGNATION_EPOCHS=80 \
  ./aco_vrp_seq.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234
ACO_SOLVER_STAGNATION_EPOCHS=80 OMP_NUM_THREADS=8 \
  mpirun -np 2 ./aco_vrp_openmp_mpi.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234
ACO_SOLVER_STAGNATION_EPOCHS=80 \
  ./aco_vrp_cuda.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 256 1234
```

Run the manifests and save CSV files plus route files under `results/`:

```bash
make solve_seq
make solve_mpi SOLVE_MPI_RANKS=2 SOLVE_MPI_OMP_THREADS=8
make solve_cuda
```

See [Instances](docs/instances.md) and [Build and run](docs/build.md).

## Experiments and results

The Makefile provides strong- and weak-scaling targets for OpenMP, MPI, and
hybrid configurations, plus a fixed-GPU CUDA problem-size sweep:

```bash
make exp_strong_openmp
make exp_strong_mpi
make exp_strong_hybrid
make exp_weak_openmp
make exp_weak_mpi
make exp_weak_hybrid
make exp_cuda_all
```

Generated runs default to `results/solve_manifest/`; practical campaigns use
`results/practical_campaign/`. The repository also tracks the data in
`results/manual_campaign/` used by the paper.

- [Experiment workflows](docs/experiments.md)
- [Slurm usage](docs/cluster_usage.md)
- [Benchmarking and plotting](docs/benchmarking.md)
- [Results and validation](docs/results.md)

## Documentation and paper

The [documentation index](docs/README.md) is the entry point for all maintained
project documentation.

| Document | Description |
| --- | --- |
| [Repository structure](docs/repository_structure.md) | Map of production sources, tools, datasets, generated artifacts, and historical code. |
| [Build and run](docs/build.md) | Dependencies, compiler targets, direct solver execution, and manifest-driven runs. |
| [Implementations](docs/implementations.md) | Relationship between the sequential, OpenMP, MPI, hybrid, and CUDA execution modes. |
| [Instances](docs/instances.md) | CVRP instance generation, manifest formats, and dataset selection. |
| [Experiments](docs/experiments.md) | Strong/weak scaling targets, CUDA sweeps, and practical experiment campaigns. |
| [Cluster usage](docs/cluster_usage.md) | Slurm submission wrappers, resource limits, launch configuration, and monitoring. |
| [Benchmarking](docs/benchmarking.md) | Supported result pipeline, validation, aggregation, and plotting tools. |
| [Tooling audit](docs/tooling_audit.md) | KEEP/UPDATE/REMOVE classification for all automation under `tools/`. |
| [Results](docs/results.md) | Output directory layout, tracked campaign data, and interpretation constraints. |
| [Paper workflow](docs/paper.md) | Paper sources, generated summaries and figures, compilation, and known data limitations. |
| [Historical experiments](docs/historical_experiments.md) | Purpose and compatibility status of prototypes under `experimental/`. |
| [Troubleshooting](docs/troubleshooting.md) | Common build, MPI, CUDA, manifest, plotting, and paper-generation failures. |

Paper sources are in `docs/paper/`; build the checked-in `report.pdf` with:

```bash
make -C docs/paper
```

See the paper workflow above for figure regeneration, generated inputs, and the
current data-path limitation.
