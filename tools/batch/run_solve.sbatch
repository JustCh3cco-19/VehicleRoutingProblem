#!/usr/bin/env bash
#SBATCH --job-name=vrp_solve
#SBATCH --output=results/slurm/%x_%j.out
#SBATCH --error=results/slurm/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --qos=students_limit

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$SLURM_SUBMIT_DIR}"
TARGET="${TARGET:-solve_all}"
MAKE_ARGS_B64="${MAKE_ARGS_B64:-}"
MODULE_LOADS_B64="${MODULE_LOADS_B64:-}"
MAKE_ARGS=""
MODULE_LOADS=""

if [[ -n "${MAKE_ARGS_B64}" ]]; then
  MAKE_ARGS="$(printf '%s' "${MAKE_ARGS_B64}" | base64 -d)"
fi
if [[ -n "${MODULE_LOADS_B64}" ]]; then
  MODULE_LOADS="$(printf '%s' "${MODULE_LOADS_B64}" | base64 -d)"
fi

if [[ -z "${ROOT_DIR}" ]]; then
  echo "[ERROR] ROOT_DIR non impostata e SLURM_SUBMIT_DIR vuota."
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p results/slurm

if [[ -f "/home/guest/init-hpc.sh" ]]; then
  # Cluster environment bootstrap (MPI/CUDA/OpenMP toolchains and modules).
  # shellcheck disable=SC1091
  set +u
  source /home/guest/init-hpc.sh
  set -u
fi

if [[ -n "${MODULE_LOADS}" ]]; then
  # Esempio: MODULE_LOADS="gcc/13.2 openmpi/4.1 cuda/12.2"
  for mod in ${MODULE_LOADS}; do
    module load "${mod}"
  done
fi

echo "[INFO] job_id=${SLURM_JOB_ID:-n/a}"
echo "[INFO] target=${TARGET}"
echo "[INFO] make_args=${MAKE_ARGS}"
echo "[INFO] cwd=$(pwd)"

make "${TARGET}" ${MAKE_ARGS}
