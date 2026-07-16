# Benchmarking and plotting

## Maintained result pipeline

The supported benchmark path is manifest-driven:

1. `make generate_problems` creates instances and backend manifests.
2. `make solve_*` or `make exp_*` writes per-run CSV rows and route files.
3. `tools/python/summarize_practical_experiments.py` aggregates practical campaigns.
4. Plotters consume a selected CSV or the paper summaries.

The generic scaling aggregator groups a per-run CSV and can write a speedup
plot:

```bash
python3 tools/python/plot_scaling_generic.py \
  --csv results/solve_manifest/csv/exp_strong_openmp/manifest_openmp_mpi_per_instance_results.csv \
  --x omp_threads \
  --out-csv results/solve_manifest/csv/exp_strong_openmp/summary.csv \
  --out-plot results/solve_manifest/csv/exp_strong_openmp/speedup.png
```

`tools/python/plot_solution.py` renders one solution or scans a solutions glob;
use `--help` for its single-run and batch modes. `plot_memory_structures.py`
estimates and plots sequential/OpenMP memory structures from the manifests.

## Paper analysis

The paper has a separate analysis pipeline tied to the tracked
`results/manual_campaign/` layout:

```bash
make -C docs/paper plots
```

This runs `docs/paper/scripts/analyze_results.py`, validates saved routes, writes
CSV summaries to `docs/paper/generated/summaries/`, updates
`docs/paper/generated/analysis.md`, and regenerates the PDFs in `docs/paper/figures/`.
See [Paper](paper.md) and [Results](results.md).

The checked-in CSVs reference the absent `instances/test_aligned/` directory,
so this regeneration command currently fails validation. The checked-in
summaries and figures are retained as the last generated artifacts.

`make run_scaling_tests` performs the fixed CUDA size sweep and profiles the
production `kernel_construct_solutions` kernel with NVIDIA Nsight Compute. It
requires `ncu` and writes logs below `results/scaling_tests/` by default.

The obsolete benchmark pipeline and old CUDA-variant utilities were removed
after the tooling audit. See [Tooling audit](tooling_audit.md) for the complete
classification and removal rationale.
