#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p results/slurm

usage() {
  cat <<'EOF'
usage: scripts/submit_openmp_mpi_tests.sh [--test light|heavy] [--customers N] [--checkpoint-mode fresh|resume|reset] [--exec-args "ARGS"] [--checkpoint-path PATH]

Backward-compatible positional form:
  scripts/submit_openmp_mpi_tests.sh [exec_args] [test_profile] [checkpoint_mode] [checkpoint_path] [customers]
EOF
}

hpc_init_script="${HPC_INIT_SCRIPT:-/home/guest/init-hpc.sh}"
if [ -f "${hpc_init_script}" ]; then
  # Some cluster init scripts read undefined vars (e.g., LD_LIBRARY_PATH).
  # Temporarily disable nounset while sourcing to avoid hard failure.
  set +u
  # shellcheck disable=SC1090
  source "${hpc_init_script}"
  set -u
  echo "[SUBMIT] Sourced HPC init: ${hpc_init_script}"
else
  echo "[SUBMIT][WARN] HPC init script not found: ${hpc_init_script}"
fi

exec_args=""
test_profile="light"
checkpoint_mode="fresh"
checkpoint_path=""
customers=""
positionals=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --test)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] missing value for --test"
        usage
        exit 1
      fi
      test_profile="$2"
      shift 2
      ;;
    --checkpoint-mode)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] missing value for --checkpoint-mode"
        usage
        exit 1
      fi
      checkpoint_mode="$2"
      shift 2
      ;;
    --customers)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] missing value for --customers"
        usage
        exit 1
      fi
      customers="$2"
      shift 2
      ;;
    --exec-args)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] missing value for --exec-args"
        usage
        exit 1
      fi
      exec_args="$2"
      shift 2
      ;;
    --checkpoint-path)
      if [ "$#" -lt 2 ]; then
        echo "[ERROR] missing value for --checkpoint-path"
        usage
        exit 1
      fi
      checkpoint_path="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "[ERROR] unknown option: $1"
      usage
      exit 1
      ;;
    *)
      positionals+=("$1")
      shift
      ;;
  esac
done

if [ "${#positionals[@]}" -gt 0 ]; then
  exec_args="${positionals[0]}"
fi
if [ "${#positionals[@]}" -gt 1 ]; then
  test_profile="${positionals[1]}"
fi
if [ "${#positionals[@]}" -gt 2 ]; then
  checkpoint_mode="${positionals[2]}"
fi
if [ "${#positionals[@]}" -gt 3 ]; then
  checkpoint_path="${positionals[3]}"
fi
if [ "${#positionals[@]}" -gt 4 ]; then
  customers="${positionals[4]}"
fi

if [[ "${test_profile}" != "light" && "${test_profile}" != "heavy" ]]; then
  echo "[ERROR] test_profile must be one of: light, heavy"
  exit 1
fi

if [ -n "${customers}" ] && ! [[ "${customers}" =~ ^[0-9]+$ ]] ; then
  echo "[ERROR] customers must be a positive integer"
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
if [ -n "${customers}" ]; then
  if [ -n "${exec_args}" ]; then
    exec_args="${exec_args} --only-n ${customers}"
  else
    exec_args="--only-n ${customers}"
  fi
fi

if [ -n "${exec_args}" ]; then
  echo "[SUBMIT] TEST_PROFILE=${test_profile}"
  echo "[SUBMIT] CHECKPOINT_MODE=${checkpoint_mode}"
  [ -n "${checkpoint_path}" ] && echo "[SUBMIT] CHECKPOINT_PATH=${checkpoint_path}"
  [ -n "${customers}" ] && echo "[SUBMIT] CUSTOMERS=${customers}"
  echo "[SUBMIT] EXEC_ARGS=${exec_args}"
  sbatch --export="${export_args},EXEC_ARGS=${exec_args}" scripts/run_openmp_mpi_tests.sbatch
else
  echo "[SUBMIT] TEST_PROFILE=${test_profile}"
  echo "[SUBMIT] CHECKPOINT_MODE=${checkpoint_mode}"
  [ -n "${checkpoint_path}" ] && echo "[SUBMIT] CHECKPOINT_PATH=${checkpoint_path}"
  [ -n "${customers}" ] && echo "[SUBMIT] CUSTOMERS=${customers}"
  sbatch --export="${export_args}" scripts/run_openmp_mpi_tests.sbatch
fi
