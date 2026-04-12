#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/batch/submit_solve.sh [options]

Opzioni:
  --target NAME           Target make da eseguire (default: solve_all)
                          Esempi: solve_seq, solve_mpi, solve_cuda, solve_pyvrp, solve_all, solve_memory_growth_non_cuda, exp_all
  --make-args "ARGS"      Argomenti extra passati a make (default: "")
  --time HH:MM:SS         Override tempo job (default: 00:30:00, QoS students_limit)
  --nodes N               Override numero nodi
  --ntasks N              Override numero task Slurm (MPI ranks allocabili)
  --cpus N                Override cpus-per-task
  --partition NAME        Override partizione
  --account NAME          Override account
  --qos NAME              Override qos (default nel job: students_limit)
  --gres SPEC             Override risorse GRES (es. gpu:1)
  --module-loads "LIST"   Moduli da caricare nel job (es. "gcc/13.2 openmpi/4.1")
  --dry-run               Stampa comando sbatch senza inviarlo
  -h, --help              Mostra help

Esempi:
  tools/batch/submit_solve.sh --target solve_seq \
    --make-args "SOLVE_CLIENTS=500,1000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=60"

  tools/batch/submit_solve.sh --target solve_mpi --cpus 32 \
    --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun SOLVE_MPI_REPEATS=3"

  tools/batch/submit_solve.sh --target solve_cuda --partition gpu --gres gpu:1 \
    --make-args "SOLVE_CLIENTS=500,1000 SOLVE_CUDA_REPEATS=3 CUDA_ARCH=sm_75"
EOF
}

target="solve_all"
make_args=""
module_loads=""
dry_run=0

sbatch_args=()
has_nodes=0
has_ntasks=0
has_cpus=0
has_gres=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:-}"
      shift 2
      ;;
    --make-args)
      make_args="${2:-}"
      shift 2
      ;;
    --module-loads)
      module_loads="${2:-}"
      shift 2
      ;;
    --time)
      sbatch_args+=(--time "${2:-}")
      shift 2
      ;;
    --nodes)
      sbatch_args+=(--nodes "${2:-}")
      has_nodes=1
      shift 2
      ;;
    --ntasks)
      sbatch_args+=(--ntasks "${2:-}")
      has_ntasks=1
      shift 2
      ;;
    --cpus)
      sbatch_args+=(--cpus-per-task "${2:-}")
      has_cpus=1
      shift 2
      ;;
    --partition)
      sbatch_args+=(--partition "${2:-}")
      shift 2
      ;;
    --account)
      sbatch_args+=(--account "${2:-}")
      shift 2
      ;;
    --qos)
      sbatch_args+=(--qos "${2:-}")
      shift 2
      ;;
    --gres)
      sbatch_args+=(--gres "${2:-}")
      has_gres=1
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] opzione non riconosciuta: $1"
      usage
      exit 2
      ;;
  esac
done

# Target-aware defaults when not explicitly overridden:
# - solve_mpi and exp_* (except exp_cuda_*): spread on 4 nodes with 4 MPI ranks total, 32 OMP threads each
# - solve_cuda and exp_cuda_*: single node/task with one GPU
# - others (including solve_seq): single node/task
if [[ "${target}" == "solve_cuda" || "${target}" == exp_cuda_* ]]; then
  if [[ "${has_nodes}" -eq 0 ]]; then
    sbatch_args+=(--nodes 1)
  fi
  if [[ "${has_ntasks}" -eq 0 ]]; then
    sbatch_args+=(--ntasks 1)
  fi
  if [[ "${has_cpus}" -eq 0 ]]; then
    sbatch_args+=(--cpus-per-task 32)
  fi
  if [[ "${has_gres}" -eq 0 ]]; then
    sbatch_args+=(--gres gpu:1)
  fi
elif [[ "${target}" == "solve_mpi" || "${target}" == exp_* ]]; then
  if [[ "${has_nodes}" -eq 0 ]]; then
    sbatch_args+=(--nodes 4)
  fi
  if [[ "${has_ntasks}" -eq 0 ]]; then
    sbatch_args+=(--ntasks 4)
  fi
  if [[ "${has_cpus}" -eq 0 ]]; then
    sbatch_args+=(--cpus-per-task 32)
  fi
else
  if [[ "${has_nodes}" -eq 0 ]]; then
    sbatch_args+=(--nodes 1)
  fi
  if [[ "${has_ntasks}" -eq 0 ]]; then
    sbatch_args+=(--ntasks 1)
  fi
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/../.." && pwd)"
job_script="${script_dir}/run_solve.sbatch"

if [[ ! -f "${job_script}" ]]; then
  echo "[ERROR] file non trovato: ${job_script}"
  exit 1
fi

if command -v base64 >/dev/null 2>&1; then
  make_args_b64="$(printf '%s' "${make_args}" | base64 | tr -d '\n')"
else
  echo "[ERROR] comando 'base64' non disponibile: impossibile serializzare --make-args in modo sicuro."
  exit 1
fi
module_loads_b64="$(printf '%s' "${module_loads}" | base64 | tr -d '\n')"
export_vars="ALL,ROOT_DIR=${root_dir},TARGET=${target},MAKE_ARGS_B64=${make_args_b64},MODULE_LOADS_B64=${module_loads_b64}"

cmd=(sbatch "${sbatch_args[@]}" --export "${export_vars}" "${job_script}")

echo "[SUBMIT] ROOT_DIR=${root_dir}"
echo "[SUBMIT] TARGET=${target}"
echo "[SUBMIT] MAKE_ARGS=${make_args}"
if [[ -n "${module_loads}" ]]; then
  echo "[SUBMIT] MODULE_LOADS=${module_loads}"
fi
if [[ ${#sbatch_args[@]} -gt 0 ]]; then
  echo "[SUBMIT] SBATCH overrides: ${sbatch_args[*]}"
fi

if [[ "${dry_run}" -eq 1 ]]; then
  printf '[DRY-RUN] %q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
