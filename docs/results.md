# Results and validation

## Output locations

| Path | Contents |
| --- | --- |
| `results/solve_manifest/csv/` | Default per-backend CSV output from `make solve_*` |
| `results/solve_manifest/solutions/` | Default route text files |
| `results/practical_campaign/<tag>/` | Practical campaign runs and summaries |
| `results/slurm/` | Slurm standard output and error logs |
| `results/analysis/` | Default memory-analysis outputs |
| `results/manual_campaign/` | Tracked measurements and solutions used by the paper |

Most new outputs are created on demand. `results/manual_campaign/` is already
tracked and currently contains sequential/CUDA size sweeps plus OpenMP and MPI
strong/weak configurations.

The Make runners name CSV files after the selected manifest, for example
`manifest_seq_per_instance_results.csv` and
`manifest_openmp_mpi_per_instance_results.csv`. Rows include run configuration,
elapsed time, status, and cost; sequential and parallel CPU runners also record
maximum RSS. Route files are stored by backend below the matching solutions
directory.

## Paper-derived outputs

`docs/paper/scripts/analyze_results.py` discovers the tracked manual campaign,
reconstructs routes against the instance files, and produces:

- `docs/paper/generated/analysis.md`, the current validation narrative;
- `docs/paper/generated/summaries/run_validation.csv`;
- backend-size, strong-scaling, combined-scaling, and weak-scaling summaries;
- the PDF plots under `docs/paper/figures/scaling/` and
  `docs/paper/figures/comparison/`.

The last checked-in validation report records 93 timing rows and flags seven
multi-rank routes whose recomputed route cost differs from the printed scalar
cost. Those timings are retained by the paper analysis, while the affected
costs are excluded. However, all 93 CSV rows reference the absent
`instances/test_aligned/` directory, so the report cannot currently be
regenerated from this checkout. Treat `docs/paper/generated/analysis.md` as a
checked-in generated artifact, not as proof of a fresh validation run.

## Interpreting measurements

- Compare speedup only when the problem, ant population, work limit, and
  stopping rule are controlled.
- Record ranks and OpenMP threads separately; total workers are their product.
- CUDA size sweeps on one GPU measure growth with input size, not GPU scaling.
- Preserve raw CSV and route files so validation and summaries can be rebuilt.
