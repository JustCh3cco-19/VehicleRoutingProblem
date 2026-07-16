# Documentation

This directory is the authoritative location for project documentation.

## Start here

- [Repository structure](repository_structure.md): source, tooling, data, and generated artifacts.
- [Build and run](build.md): dependencies, build targets, direct execution, and manifest runs.
- [Implementations](implementations.md): supported sequential, OpenMP, MPI, hybrid, and CUDA modes.
- [Instances](instances.md): generated CVRP instances and manifest formats.

## Experiments and outputs

- [Experiments](experiments.md): maintained Make targets and practical campaigns.
- [Cluster usage](cluster_usage.md): Slurm submission, resource limits, and monitoring.
- [Benchmarking](benchmarking.md): result collection, summaries, validation, and plots.
- [Tooling audit](tooling_audit.md): status and responsibility of every tool under `tools/`.
- [Results](results.md): output locations and the tracked paper dataset.
- [Troubleshooting](troubleshooting.md): common build and workflow failures.

## Reports and historical material

- [Paper](paper.md): LaTeX sources, generated tables, figures, and build commands.
- [Historical experiments](historical_experiments.md): prototypes under `experimental/` and their compatibility status.

Run `make help` for the complete Make target and variable reference. Run
`make pdfs` to rebuild the paper PDF.
