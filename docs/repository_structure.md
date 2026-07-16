# Repository structure

## Production solver

```text
include/                 Shared public headers
src/main.c               Sequential and OpenMP+MPI command-line entry point
src/common/              Configuration, parsing, matrices, routes, and RNG helpers
src/seq/                 Sequential ACO implementation
src/openmp-mpi/          Hybrid OpenMP+MPI ACO implementation
src/cuda/                CUDA entry point, host code, and kernels
tests/cuda/               CUDA memory-versus-compute benchmark source
```

The root `Makefile` includes the modules in `tools/makefile/`. Build products
are written to the repository root as `aco_vrp_seq.out`,
`aco_vrp_openmp_mpi.out`, and `aco_vrp_cuda.out`.

## Data and automation

```text
instances/generated_benchmark/   Generated `.vrp` files and three manifests
tools/bash/                       Manifest runners, smoke tests, and CSV utilities
tools/batch/                      Generic and campaign-specific Slurm submission
tools/python/                     Generation, benchmarking, validation, and plotting
results/                          Solver outputs and tracked manual campaign data
```

The supported data flow is:

```text
generate instances -> read manifests -> run backends -> save CSV/routes -> summarize/plot
```

`tools/makefile/vars.mk` defines defaults, `build.mk` compiles the backends,
`generate.mk` creates instances, `solve.mk` runs manifests, and
`experiments.mk` composes experiment campaigns.

### Tool inventory

| Group | Files |
| --- | --- |
| Manifest execution | `solve_seq.sh`, `solve_mpi.sh`, `solve_cuda.sh`, `solve_pyvrp.sh` |
| Checks and CSV handling | `smoke_test.sh`, `merge_results_csv_by_n.sh` |
| Slurm | `run_solve.sbatch`, `submit_solve.sh`, `submit_practical_campaign.sh` |
| Instance/PyVRP helpers | `generate_vrp_problem.py`, `solve_pyvrp_runner.py`, `validate_pyvrp.py` |
| Maintained campaigns | `run_practical_experiments.py`, `summarize_practical_experiments.py` |
| Maintained plotting | `plot_scaling_generic.py`, `plot_solution.py`, `plot_memory_structures.py` |
| Paper analysis | `docs/paper/scripts/analyze_results.py` |

The disposition of every audited tool is recorded in
[Tooling audit](tooling_audit.md).

## Documentation and report

```text
docs/                    Maintained project documentation
docs/paper/               Academic paper source
docs/paper/sections/      Paper sections included by `docs/paper/main.tex`
docs/paper/generated/     Analysis Markdown and generated CSV summaries
docs/paper/figures/       PDFs included by the paper
report.pdf                Paper output built from `docs/paper/`
```

`docs/paper/generated/analysis.md` is generated output, not a maintained guide.

## Historical code

`experimental/exp_dist_calc/` compares CPU and GPU distance calculation.
`experimental/experiments/openmp_v2/` contains numbered development
experiments with copied solver variants. These files are outside the production
build and several launchers retain obsolete paths; see
[Historical experiments](historical_experiments.md).
