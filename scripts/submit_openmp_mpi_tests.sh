#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/slurm

exec_args="${1:-}"

if [ -n "${exec_args}" ]; then
  echo "[SUBMIT] EXEC_ARGS=${exec_args}"
  sbatch --export=ALL,EXEC_ARGS="${exec_args}" scripts/run_openmp_mpi_tests.sbatch
else
  sbatch scripts/run_openmp_mpi_tests.sbatch
fi
