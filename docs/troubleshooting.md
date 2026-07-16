# Troubleshooting

## `make all` fails on a machine without CUDA or MPI

`make all` requires every toolchain. Build only the available backend with
`make seq`, `make openmp_mpi`, or `make cuda`.

## `nvcc` rejects the target architecture

The default is `CUDA_ARCH=sm_75`. Select an architecture supported by the
installed toolkit and target GPU, for example `make cuda CUDA_ARCH=sm_80`.

## MPI ranks or OpenMP threads do not match the request

For direct runs, set both the launcher rank count and `OMP_NUM_THREADS`. For
manifest runs, use `SOLVE_MPI_RANKS`, `SOLVE_MPI_OMP_THREADS`, and a compatible
`SOLVE_MPI_LAUNCHER`. Inside Slurm, `srun` may be required by the site MPI
configuration.

## A manifest run reports a missing manifest or no selected rows

Regenerate the default dataset with `make generate_problems`, or pass the
correct `SOLVE_MANIFEST`, `SOLVE_MANIFEST_MPI`, or `SOLVE_MANIFEST_CUDA` path.
Check `SOLVE_CLIENTS` and `SOLVE_LIMIT` filters.

## A direct sequential or CUDA run does not stop

Those backends default to no timeout and no stagnation limit. Set
`ACO_SOLVER_TIMEOUT_SECONDS` or `ACO_SOLVER_STAGNATION_EPOCHS`; the maintained
`make solve_*` runners already pass bounded defaults.

## A Python plotter cannot find its input CSV

Use a per-run CSV written by `make solve_*` or `make exp_*`, and pass its exact
path to `plot_scaling_generic.py`. The removed legacy plotters used directory
layouts that are no longer produced. See [Benchmarking](benchmarking.md).

## Paper figures or summaries are stale

When the exact campaign instances are available, run
`make -C docs/paper plots` before `make -C docs/paper`. The analysis expects
the tracked manual campaign layout and writes its validation details to
`docs/paper/generated/analysis.md`.

In the current checkout, the campaign CSVs reference the missing
`instances/test_aligned/` directory, so plot regeneration cannot complete.
The paper itself can still be compiled from the checked-in figures with
`make -C docs/paper`.

If `pdflatex` reports a missing `.sty` file, install the corresponding LaTeX
package collection. A minimal `pdflatex` installation is insufficient for the
packages imported by `docs/paper/main.tex`.

## Where are all Make variables documented?

Run `make help`. The output is generated from current Makefile defaults and is
the detailed reference; the Markdown guides intentionally document only the
main controls.
