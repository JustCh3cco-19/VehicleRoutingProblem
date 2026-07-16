# Documentation

The project documentation is grouped by topic:

All LaTeX documents can be compiled from the repository root with:

```bash
make pdfs
```

Generated PDFs and LaTeX intermediates can be removed with `make clean_pdfs`.

## Backend overview

- [Sequential, OpenMP--MPI and CUDA explanation](backends/spiegazione_backend.pdf)
- [LaTeX source](backends/spiegazione_backend.tex)

## OpenMP and MPI

- [OpenMP--MPI roadmap](openmp-mpi/RoadmapOpenMP_MPI.md)
- [V3 collaborative approach failure analysis](openmp-mpi/v3_failure_analysis.md)

## CUDA

- [Academic implementation notes](cuda/cuda_v6_academic_paper.md)
- [Quality assurance](cuda/cuda_v6_qa.md)
- [Scaling report](cuda/cuda_v6_scaling_report.md)

## Experiments

- [Practical experiment campaign](experiments/practical_experiment_campaign.md)

## Operations

- [Slurm cluster commands](operations/slurm_cluster_commands.md)
