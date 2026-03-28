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

hpc_init_script="${HPC_INIT_SCRIPT:-/home/guest/init-hpc.sh}"
if [ -f "${hpc_init_script}" ]; then
  # Some cluster init scripts read undefined vars (e.g., LD_LIBRARY_PATH).
  # Temporarily disable nounset while sourcing to avoid hard failure.
  set +u
  # shellcheck disable=SC1090
  source "${hpc_init_script}"
  set -u
  echo "[INFO] Sourced HPC init: ${hpc_init_script}"
else
  echo "[WARN] HPC init script not found: ${hpc_init_script}"
fi

test_profile="${TEST_PROFILE:-light}"
if [ "${test_profile}" = "heavy" ]; then
  test_bin="./tests/openmp_mpi_tests_heavy.out"
  checkpoint_path="${CHECKPOINT_PATH:-results/openmp_mpi_tests_heavy.checkpoint}"
  make tests/openmp_mpi_tests_heavy.out
else
  test_bin="./tests/openmp_mpi_tests.out"
  checkpoint_path="${CHECKPOINT_PATH:-results/openmp_mpi_tests_light.checkpoint}"
  make tests/openmp_mpi_tests.out
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
exec_args="${EXEC_ARGS:-}"
checkpoint_mode="${CHECKPOINT_MODE:-fresh}"
checkpoint_flags="--checkpoint ${checkpoint_path}"

if [ "${checkpoint_mode}" = "reset" ]; then
  rm -f "${checkpoint_path}"
  checkpoint_flags="${checkpoint_flags} --reset-checkpoint"
elif [ "${checkpoint_mode}" = "resume" ]; then
  checkpoint_flags="${checkpoint_flags} --resume"
fi

echo "[INFO] test_profile=${test_profile}"
echo "[INFO] test_bin=${test_bin}"
echo "[INFO] checkpoint_mode=${checkpoint_mode}"
echo "[INFO] checkpoint_path=${checkpoint_path}"
mpirun -n "${SLURM_NTASKS}" "${test_bin}" ${checkpoint_flags} ${exec_args}
