#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/slurm

hpc_init_script="${HPC_INIT_SCRIPT:-/home/guest/init-hpc.sh}"
if [ -f "${hpc_init_script}" ]; then
  # shellcheck disable=SC1090
  source "${hpc_init_script}"
  echo "[SUBMIT] Sourced HPC init: ${hpc_init_script}"
else
  echo "[SUBMIT][WARN] HPC init script not found: ${hpc_init_script}"
fi

exec_args="${1:-}"
test_profile="${2:-light}"
checkpoint_mode="${3:-fresh}"
checkpoint_path="${4:-}"

if [[ "${test_profile}" != "light" && "${test_profile}" != "heavy" ]]; then
  echo "[ERROR] test_profile must be one of: light, heavy"
  exit 1
fi

if [[ "${checkpoint_mode}" != "fresh" && "${checkpoint_mode}" != "resume" && "${checkpoint_mode}" != "reset" ]]; then
  echo "[ERROR] checkpoint_mode must be one of: fresh, resume, reset"
  exit 1
fi

export_args="ALL,TEST_PROFILE=${test_profile},CHECKPOINT_MODE=${checkpoint_mode}"
if [ -n "${checkpoint_path}" ]; then
  export_args="${export_args},CHECKPOINT_PATH=${checkpoint_path}"
fi

if [ -n "${exec_args}" ]; then
  echo "[SUBMIT] TEST_PROFILE=${test_profile}"
  echo "[SUBMIT] CHECKPOINT_MODE=${checkpoint_mode}"
  [ -n "${checkpoint_path}" ] && echo "[SUBMIT] CHECKPOINT_PATH=${checkpoint_path}"
  echo "[SUBMIT] EXEC_ARGS=${exec_args}"
  sbatch --export="${export_args},EXEC_ARGS=${exec_args}" scripts/run_openmp_mpi_tests.sbatch
else
  echo "[SUBMIT] TEST_PROFILE=${test_profile}"
  echo "[SUBMIT] CHECKPOINT_MODE=${checkpoint_mode}"
  [ -n "${checkpoint_path}" ] && echo "[SUBMIT] CHECKPOINT_PATH=${checkpoint_path}"
  sbatch --export="${export_args}" scripts/run_openmp_mpi_tests.sbatch
fi
