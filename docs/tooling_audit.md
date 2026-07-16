# Tooling audit

This page records the disposition of every file audited under `tools/` after
the repository refactor. `KEEP` means the file already matched the current
workflow. `UPDATE` means it remains supported and was corrected during this
audit. `REMOVE` lists obsolete files deleted by the audit.

No retained file qualified as `KEEP`: every current tool required at least one
path, validation, default, dependency, or documentation correction.

## Shell runners

| Classification | File | Responsibility or change |
| --- | --- | --- |
| UPDATE | `bash/merge_results_csv_by_n.sh` | Replaces rows by customer count; now validates CSV shape and matching headers. |
| UPDATE | `bash/run_scaling_tests.sh` | CUDA run/NCU sweep; now targets `aco_vrp_cuda.out`, the current kernel name, checked-in instances, and `results/scaling_tests/`. |
| UPDATE | `bash/smoke_test.sh` | Sequential, MPI, and CUDA smoke checks with current binaries and CLI. |
| UPDATE | `bash/solve_seq.sh` | Sequential manifest runner; validates inputs and current stopping controls. |
| UPDATE | `bash/solve_mpi.sh` | OpenMP/MPI manifest runner; validates launcher, ranks, threads, and current environment variables. |
| UPDATE | `bash/solve_cuda.sh` | CUDA manifest runner; removes dead variant/profile controls and honors `SOLVE_CUDA_M`. |
| UPDATE | `bash/solve_pyvrp.sh` | PyVRP manifest runner; removes the obsolete `VRP/` environment fallback and validates its inputs. |

The `solve_*` scripts are implementation helpers for the corresponding Make
targets. Invoke `make solve_seq`, `make solve_mpi`, `make solve_cuda`, or
`make solve_pyvrp` from the repository root so their required directories and
variables are prepared consistently.

## Slurm tools

| Classification | File | Responsibility or change |
| --- | --- | --- |
| UPDATE | `batch/run_solve.sbatch` | Generic job entry point; safely decodes Make arguments, checks module support, and defaults to `solve_seq`. |
| UPDATE | `batch/submit_solve.sh` | Validates targets, resources, option values, and `KEY=VALUE` Make arguments; assigns resources by backend. |
| UPDATE | `batch/submit_practical_campaign.sh` | Submits separate CPU and GPU campaigns; validates tags and required option values. |

`solve_all` is intentionally rejected by the generic submission wrapper: it
combines serial, MPI, and CUDA work that requires different Slurm allocations.

## Make modules

| Classification | File | Responsibility or change |
| --- | --- | --- |
| UPDATE | `makefile/build.mk` | Builds the three current executables and removes stale cleanup paths. |
| UPDATE | `makefile/generate.mk` | Generates the three manifests with defaults aligned to the checked-in size series. |
| UPDATE | `makefile/experiments.mk` | Uses experiment repeat counts and passes the CUDA ant-count override correctly. |
| UPDATE | `makefile/help.mk` | Documents current instance sizes, CUDA controls, targets, and defaults. |
| UPDATE | `makefile/pdfs.mk` | Builds only the maintained paper under `docs/paper/`. |
| UPDATE | `makefile/phony.mk` | Includes the maintained CUDA profiling target. |
| UPDATE | `makefile/solve.mk` | Checks only the manifest needed by each backend and passes backend-specific controls. |
| UPDATE | `makefile/vars.mk` | Removes unused CUDA-variant and MPI variables and aligns campaign defaults with checked-in manifests. |

## Python utilities

| Classification | File | Responsibility or change |
| --- | --- | --- |
| UPDATE | `python/generate_vrp_problem.py` | Generates current unit-demand CVRP files; validates generation parameters and optional reference solves. |
| UPDATE | `python/plot_memory_structures.py` | Estimates current double/float matrix layouts and mirrors the OpenMP L3-based candidate tuning. |
| UPDATE | `python/plot_scaling_generic.py` | Aggregates current per-run CSVs; validates filters and ignores invalid timings. |
| UPDATE | `python/plot_solution.py` | Plots current route files in single or batch mode with clearer path validation. |
| UPDATE | `python/run_practical_experiments.py` | Orchestrates current Make targets, manifests, paths, resource limits, and CUDA defaults. |
| UPDATE | `python/solve_pyvrp_runner.py` | Runs one PyVRP solve and reports infeasible results as failures. |
| UPDATE | `python/summarize_practical_experiments.py` | Reads the current fixed CUDA CSV name and emits campaign summaries under `<campaign>/reports/`. |
| UPDATE | `python/validate_pyvrp.py` | Validates any backend route file with current generic labels and input checks. |

## Removed obsolete tools

| Classification | Removed file | Reason |
| --- | --- | --- |
| REMOVE | `bash/run_and_validate.sh` | Used the absent `cuda-vrp` target and `aco_vrp_cuda_vrp` executable. |
| REMOVE | `python/benchmark_cuda_vs_pyvrp.py` | Used the old CUDA executable and command-line interface. |
| REMOVE | `python/benchmark_pipeline.py` | Duplicated the manifest workflow around obsolete build targets and output directories. |
| REMOVE | `python/benchmark_scaling.py` | Called absent `omp`/`mpi` targets and obsolete executable names. |
| REMOVE | `python/plot_scaling_results.py` | Consumed only the removed legacy scaling report layout. |
| REMOVE | `python/profile_cuda.py` | Used the absent `cuda-vrp` target and old report paths. |
| REMOVE | `python/generate_comparison_plot.py` | Required removed `docs/cuda/performance/v2` and `v3` CSV layouts. |
| REMOVE | `python/plot_merged_by_run_backend.py` | Required the no-longer-produced `merged_by_run_backend/` directory. |

For supported commands, see [Build and run](build.md),
[Experiments](experiments.md), [Cluster usage](cluster_usage.md), and
[Benchmarking](benchmarking.md).
