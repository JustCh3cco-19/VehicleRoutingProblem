# Implementations

All production implementations solve generated unit-demand CVRP instances with
the shared ACO interfaces in `include/aco.h`. Parsing, solution storage and
validation, runtime configuration, and common random helpers live in
`src/common/`.

## Supported modes

| Mode | Source | Build target | Runtime configuration |
| --- | --- | --- | --- |
| Sequential | `src/seq/aco_sequential.c` | `make seq` | `aco_vrp_seq.out` |
| OpenMP | `src/openmp-mpi/aco_parallel.c` | `make openmp_mpi` | one MPI rank, multiple threads |
| MPI | `src/openmp-mpi/aco_parallel.c` | `make openmp_mpi` | multiple ranks, one thread each |
| Hybrid | `src/openmp-mpi/aco_parallel.c` | `make openmp_mpi` | multiple ranks and threads |
| CUDA | `src/cuda/aco_cuda.cu`, `src/cuda/aco_cuda_kernels.cu` | `make cuda` | one CUDA device |

OpenMP and MPI are not separate source implementations or executables. Their
independent experiment labels select resource configurations of the hybrid
binary.

## Sequential CPU

`src/main.c` loads either a TSPLIB-like EUC_2D instance or a small built-in
demo matrix. The sequential solver uses dense distance and pheromone storage,
constructs ants serially, and returns route text plus the best cost.

## OpenMP and MPI

The hybrid backend is compiled with `mpicc`, `-fopenmp`, and `-DUSE_MPI`.
OpenMP parallelizes work inside each rank; MPI distributes work across ranks.
The program requests `MPI_THREAD_FUNNELED`, so MPI calls are made through the
funneled thread model. Runtime process and thread counts are supplied by the
launcher and `OMP_NUM_THREADS`.

The Make workflow maps `SOLVE_MPI_RANKS` and `SOLVE_MPI_OMP_THREADS` to those
settings. `SOLVE_MPI_LAUNCHER` accepts `auto`, `mpirun`, or `srun`.

## CUDA

`src/cuda/main_vrp.cu` loads instance coordinates and invokes the CUDA ACO
implementation. CUDA compilation is controlled by `NVCC`, `NVCC_FLAGS`, and
`CUDA_ARCH`; the default architecture is `sm_75`. The production build creates
one CUDA executable. Experimental variants under `experimental/` are not part
of the production targets.
