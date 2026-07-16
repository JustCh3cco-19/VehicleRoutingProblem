# Build and run

Run commands from the repository root unless stated otherwise.

## Requirements

- `make` and a C11 compiler (`gcc` by default) for the sequential backend;
- `mpicc`, an MPI runtime, and OpenMP support for the parallel CPU backend;
- `nvcc` and a compatible NVIDIA CUDA environment for the CUDA backend;
- Python 3.10 or newer for generation, validation, benchmarks, and plots;
- `pdflatex` plus the LaTeX packages imported by `docs/paper/main.tex` for the
  paper PDF.

Install the declared Python packages in an isolated environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The Makefile automatically uses `.venv/bin/python` when it exists.

## Build targets

```bash
make seq
make openmp_mpi
make cuda
```

Outputs are `aco_vrp_seq.out`, `aco_vrp_openmp_mpi.out`, and
`aco_vrp_cuda.out`. `make all` builds all of them. Useful overrides include:

```bash
make seq CC=clang
make openmp_mpi MPICC=mpicc PERF_FLAGS="-march=native"
make cuda CUDA_ARCH=sm_80
```

The default CPU flags include `-O3 -Wall -Wextra -std=c11`; the CUDA build uses
`-O3 -std=c++17` and defaults to `sm_75`. Use `make clean` before rebuilding
with incompatible flags.

## Direct execution

All backends accept a VRP file, vehicle count, ant count, and optional seed:

```text
<executable> <instance.vrp> <K> <m> [seed]
```

Examples:

```bash
ACO_SOLVER_STAGNATION_EPOCHS=80 \
  ./aco_vrp_seq.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234

ACO_SOLVER_STAGNATION_EPOCHS=80 OMP_NUM_THREADS=8 mpirun -np 1 \
  ./aco_vrp_openmp_mpi.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234

ACO_SOLVER_STAGNATION_EPOCHS=80 OMP_NUM_THREADS=1 mpirun -np 2 \
  ./aco_vrp_openmp_mpi.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234

ACO_SOLVER_STAGNATION_EPOCHS=80 \
  ./aco_vrp_cuda.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 256 1234
```

The first parallel example is OpenMP-only operation; the second is MPI-only
operation. Set both counts above one for hybrid execution.

Solver stopping controls are environment variables:

```bash
ACO_SOLVER_TIMEOUT_SECONDS=60 \
ACO_SOLVER_STAGNATION_EPOCHS=80 \
ACO_SOLVER_MIN_REL_IMPROVEMENT=0.1 \
./aco_vrp_seq.out instances/generated_benchmark/n1000_k8_s19001.vrp 8 128 1234
```

`ACO_SOLVER_MIN_REL_IMPROVEMENT` is a percentage: `0.1` means 0.1%. Fixed-work
experiments on the OpenMP+MPI backend can set `ACO_SOLVER_FIXED_EPOCHS`.
Direct sequential and CUDA runs have no positive stopping limit by default, so
set a timeout or stagnation limit as shown above.

## Manifest-driven runs

The maintained runners write one CSV per backend and one route file per run:

```bash
make solve_pyvrp
make solve_seq
make solve_mpi SOLVE_MPI_RANKS=2 SOLVE_MPI_OMP_THREADS=8
make solve_cuda
```

Common controls are `SOLVE_CLIENTS`, `SOLVE_LIMIT`, `SOLVE_*_REPEATS`,
`SOLVE_*_RUNTIME_S`, `SOLVE_*_STAGNATION_EPOCHS`, and
`SOLVE_*_MIN_REL_IMPROVEMENT`. Use `make help` for defaults and all variables.

Use `make smoke`, or the backend-specific `smoke_seq`, `smoke_mpi`, and
`smoke_cuda` targets, for the small checks implemented by
`tools/bash/smoke_test.sh`.
