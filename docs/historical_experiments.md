# Historical experiments

The `experimental/` tree records development prototypes. It is not compiled by
the root Makefile and is not part of the supported solver workflow.

## Contents

`experimental/exp_dist_calc/` contains CPU and CUDA distance-calculation
benchmarks plus a Python launcher.

`experimental/experiments/openmp_v2/` contains numbered studies of:

1. OpenMP scheduling;
2. fixed-epoch strong scaling;
3. scaling to 8,000 customers;
4. scheduling versus problem size;
5. ants-per-core granularity;
6. granularity versus problem size;
7. double-versus-float performance breakdown;
8. post-float strong scaling;
9. prefetching;
10. AVX2 SIMD;
11. two-level candidate lists;
12. hierarchical visited masks;
13. adaptive tuning;
14. MPI stale-synchronous-parallel overlap;
15. sparse synchronization diagnostics;
16. sparse MPI synchronization;
17. sparse asynchronous synchronization;
18. a failed collaborative intra-ant prototype.

Each directory retains its source and, where present, `run.sh`. These scripts
are historical snapshots: several refer to the removed
`aco_vrp_v2_openmp_mpi.out` binary or the removed `src/openmp-mpi/aco_v2.c`
path, and some compile against an older shared-source list. They should not be
presented as reproducible current commands.

The former per-directory README files were consolidated here because they
duplicated script constants and included unverified expected results. To study
or port an experiment, treat its `run.sh` and copied source as the primary
record and update it explicitly for the current production interfaces.
