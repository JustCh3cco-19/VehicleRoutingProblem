# Experiments

The maintained experiment entry points reuse the manifest runners in
`tools/bash/` and write under `RESULTS_ROOT` (`results` by default).

## Make campaigns

CPU strong scaling keeps the selected instance size fixed and changes the
worker configuration:

```bash
make exp_strong_openmp
make exp_strong_mpi
make exp_strong_hybrid
```

CPU weak scaling changes the configured workload with the resources:

```bash
make exp_weak_openmp
make exp_weak_mpi
make exp_weak_hybrid
```

`make exp_all` runs those six CPU targets. Defaults and configuration series
are defined in `tools/makefile/vars.mk`; use `make help` before a campaign.
Important overrides include `EXP_RUNTIME_S`, `EXP_STAGNATION_EPOCHS`,
`EXP_MIN_REL_IMPROVEMENT`, `EXP_MPI_LAUNCHER`, and the target-specific thread,
rank, pair, or triplet lists.

CUDA has no process- or node-scaling target. The maintained aggregate target is
a problem-size sweep on one fixed GPU:

```bash
make exp_cuda_all
```

It currently delegates only to `exp_cuda_scaling_input`. The separate
`exp_cuda_scaling_ants` target varies the ant count and is not included in
`exp_cuda_all`; each ant count is stored in its own `m<value>/` CSV and solution
subdirectory so later points do not replace earlier results.

## Practical campaign

`tools/python/run_practical_experiments.py` composes sequential, OpenMP, MPI,
hybrid, CUDA, and quality runs through the supported `make solve_*` targets.
Always inspect the planned commands first:

```bash
python3 tools/python/run_practical_experiments.py --dry-run
```

For a local or allocated-node run:

```bash
python3 tools/python/run_practical_experiments.py \
  --results-root results/practical_campaign/my_run \
  --launcher mpirun \
  --skip-cuda

python3 tools/python/summarize_practical_experiments.py \
  --root results/practical_campaign/my_run
```

The summarizer writes CSV summaries and
`reports/practical_experiments_report.md` below the campaign root. CPU-only and
GPU-only Make wrappers are `exp_practical_cpu` and `exp_practical_gpu`.

The script validates requested sizes and referenced instance paths against the
current manifests before it runs. Its built-in CPU and CUDA series match the
checked-in manifests; custom `--cuda-clients` and
`--cuda-cpu-comparison-clients` values must also be present.

## Reproducibility

- Keep manifests, seeds, ant counts, stopping criteria, ranks, and thread counts
  with every result set.
- Use fixed epochs when comparing execution time at equal work.
- Do not interpret the CUDA problem-size sweep as GPU scaling.
- Do not compare backend cost or speedup unless work and stopping conditions
  are aligned.

For Slurm submission, see [Cluster usage](cluster_usage.md). For aggregation and
plots, see [Benchmarking](benchmarking.md).
