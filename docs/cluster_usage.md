# Cluster usage

The repository provides one generic Slurm job script,
`tools/batch/run_solve.sbatch`, and two submission wrappers. Run submission
commands from the repository root on a system with `sbatch`.

## Submit one target

Preview the generated submission first:

```bash
tools/batch/submit_solve.sh --dry-run --target solve_seq \
  --make-args "SOLVE_CLIENTS=1000,2000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=60"
```

Examples:

```bash
tools/batch/submit_solve.sh --target solve_mpi \
  --nodes 2 --ntasks 2 --cpus 16 \
  --make-args "SOLVE_MPI_RANKS=2 SOLVE_MPI_OMP_THREADS=16 SOLVE_MPI_LAUNCHER=mpirun"

tools/batch/submit_solve.sh --target solve_cuda \
  --partition gpu --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=1000,2000 SOLVE_CUDA_REPEATS=3 CUDA_ARCH=sm_75"
```

The wrapper supports `--time`, `--nodes`, `--ntasks`, `--cpus`, `--mem`,
`--partition`, `--account`, `--qos`, `--gres`, and `--module-loads`.
Its default target is `solve_seq`. Submit `solve_all` as separate backend jobs,
because its sequential, MPI, and CUDA prerequisites require different Slurm
resources.

## Repository-enforced limits

`submit_solve.sh` rejects requests above:

- 30 minutes wall time;
- 4 nodes and 4 tasks;
- 32 CPUs per task;
- 32 GiB per node;
- one GPU (`gpu:1`);
- 300 seconds for solver runtime variables passed through `--make-args`.

These checks describe this repository's submission wrapper, not a general
Slurm policy. The job file defaults to the `students_limit` QoS and writes logs
to `results/slurm/`.

## Submit the practical campaign

The campaign wrapper submits separate CPU and GPU jobs with one shared tag:

```bash
tools/batch/submit_practical_campaign.sh --dry-run --tag trial
tools/batch/submit_practical_campaign.sh --tag trial
```

Its CPU job requests four nodes, four tasks, and 32 CPUs per task. Its GPU job
requests one node, one task, 32 CPUs, and one GPU. Both default to 30 minutes.
Override module names with `--module-loads` when the cluster does not provide
the required compilers by default.

## Inside an allocation

Use `SOLVE_MPI_LAUNCHER=srun` if the cluster requires Slurm to launch MPI ranks:

```bash
make solve_mpi SOLVE_MPI_LAUNCHER=srun \
  SOLVE_MPI_RANKS="${SLURM_NTASKS}" \
  SOLVE_MPI_OMP_THREADS="${SLURM_CPUS_PER_TASK}"
```

Check `results/slurm/<job-name>_<job-id>.out` and `.err` for job output. Use
standard site commands such as `squeue -j <job-id>` and `sacct -j <job-id>` for
monitoring; available fields and MPI plugins depend on the cluster.
