#!/usr/bin/env bash
#SBATCH --job-name=vrp_openmp_mpi_tests
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --qos=students_limit
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p results/slurm

# Build only the needed test binary.
make tests/openmp_mpi_tests.out

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec_args="${EXEC_ARGS:-}"

mpirun -n "${SLURM_NTASKS}" ./tests/openmp_mpi_tests.out ${exec_args}
