#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/batch/submit_solve.sh [options]

Opzioni:
  --target NAME           Target make da eseguire (default: solve_seq)
                          Esempi: solve_seq, solve_mpi, solve_cuda, solve_pyvrp, exp_all
  --make-args "ARGS"      Argomenti extra passati a make (default: "")
  --time HH:MM:SS         Override tempo job (default: 00:30:00, QoS students_limit)
  --nodes N               Override numero nodi
  --ntasks N              Override numero task Slurm (MPI ranks allocabili)
  --cpus N                Override cpus-per-task
  --mem SIZE              Memoria richiesta per nodo (es. 32G)
  --partition NAME        Override partizione
  --account NAME          Override account
  --qos NAME              Override qos (default nel job: students_limit)
  --gres SPEC             Override risorse GRES (es. gpu:1)
  --module-loads "LIST"   Moduli da caricare nel job (es. "gcc/13.2 openmpi/4.1")
  --dry-run               Stampa comando sbatch senza inviarlo
  -h, --help              Mostra help

Esempi:
  tools/batch/submit_solve.sh --target solve_seq \
    --make-args "SOLVE_CLIENTS=1000,2000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=60"

  tools/batch/submit_solve.sh --target solve_mpi --cpus 32 \
    --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun SOLVE_MPI_REPEATS=3"

  tools/batch/submit_solve.sh --target solve_cuda --partition gpu --gres gpu:1 \
    --make-args "SOLVE_CLIENTS=1000,2000 SOLVE_CUDA_REPEATS=3 CUDA_ARCH=sm_75"
EOF
}

target="solve_seq"
make_args=""
module_loads=""
dry_run=0

sbatch_args=()
has_nodes=0
has_ntasks=0
has_cpus=0
has_gres=0
time_value="00:30:00"
nodes_value=""
ntasks_value=""
cpus_value=""
gres_value=""
mem_value=""

require_value() {
  if [ "$#" -lt 2 ] || [ -z "$2" ]; then
    echo "[ERROR] missing value for $1" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      require_value "$@"
      target="$2"
      shift 2
      ;;
    --make-args)
      require_value "$@"
      make_args="$2"
      shift 2
      ;;
    --module-loads)
      require_value "$@"
      module_loads="$2"
      shift 2
      ;;
    --time)
      require_value "$@"
      time_value="$2"
      sbatch_args+=(--time "${time_value}")
      shift 2
      ;;
    --nodes)
      require_value "$@"
      nodes_value="$2"
      sbatch_args+=(--nodes "${nodes_value}")
      has_nodes=1
      shift 2
      ;;
    --ntasks)
      require_value "$@"
      ntasks_value="$2"
      sbatch_args+=(--ntasks "${ntasks_value}")
      has_ntasks=1
      shift 2
      ;;
    --cpus)
      require_value "$@"
      cpus_value="$2"
      sbatch_args+=(--cpus-per-task "${cpus_value}")
      has_cpus=1
      shift 2
      ;;
    --mem)
      require_value "$@"
      mem_value="$2"
      sbatch_args+=(--mem "${mem_value}")
      shift 2
      ;;
    --partition)
      require_value "$@"
      sbatch_args+=(--partition "$2")
      shift 2
      ;;
    --account)
      require_value "$@"
      sbatch_args+=(--account "$2")
      shift 2
      ;;
    --qos)
      require_value "$@"
      sbatch_args+=(--qos "$2")
      shift 2
      ;;
    --gres)
      require_value "$@"
      gres_value="$2"
      sbatch_args+=(--gres "${gres_value}")
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

case "$target" in
  solve_seq|solve_mpi|solve_cuda|solve_pyvrp|solve_memory_growth_non_cuda|\
  exp_strong_openmp|exp_strong_openmp_n8000_m256|exp_weak_openmp|\
  exp_strong_mpi|exp_strong_hybrid|exp_weak_mpi|exp_weak_hybrid|exp_all|\
  exp_cuda_scaling_input|exp_cuda_scaling_ants|exp_seq_for_cuda|exp_cuda_all|\
  exp_practical_cpu|exp_practical_gpu|run_scaling_tests)
    ;;
  solve_all)
    echo "[ERROR] solve_all mixes CPU, MPI, and CUDA resources; submit its solve targets separately" >&2
    exit 2
    ;;
  *)
    echo "[ERROR] unsupported batch target: $target" >&2
    exit 2
    ;;
esac

if [[ ! "${time_value}" =~ ^([0-9]+):([0-5][0-9]):([0-5][0-9])$ ]]; then
  echo "[ERROR] formato --time non valido: ${time_value} (atteso HH:MM:SS)"
  exit 2
fi
time_seconds=$((10#${BASH_REMATCH[1]} * 3600 + 10#${BASH_REMATCH[2]} * 60 + 10#${BASH_REMATCH[3]}))
if (( time_seconds > 1800 )); then
  echo "[ERROR] la policy consente al massimo --time 00:30:00"
  exit 2
fi
if [[ -n "${nodes_value}" ]] && { [[ ! "${nodes_value}" =~ ^[1-9][0-9]*$ ]] || (( nodes_value > 4 )); }; then
  echo "[ERROR] --nodes deve essere compreso tra 1 e 4"
  exit 2
fi
if [[ -n "${ntasks_value}" ]] && { [[ ! "${ntasks_value}" =~ ^[1-9][0-9]*$ ]] || (( ntasks_value > 4 )); }; then
  echo "[ERROR] --ntasks deve essere compreso tra 1 e 4"
  exit 2
fi
if [[ -n "${cpus_value}" ]] && { [[ ! "${cpus_value}" =~ ^[1-9][0-9]*$ ]] || (( cpus_value > 32 )); }; then
  echo "[ERROR] --cpus deve essere compreso tra 1 e 32"
  exit 2
fi
if [[ -n "${mem_value}" ]]; then
  if [[ ! "${mem_value}" =~ ^([1-9][0-9]*)([GgMm])$ ]]; then
    echo "[ERROR] formato --mem non valido: usare G o M (es. 32G)"
    exit 2
  fi
  mem_amount="${BASH_REMATCH[1]}"
  mem_unit="${BASH_REMATCH[2]}"
  if { [[ "${mem_unit^^}" == "G" ]] && (( mem_amount > 32 )); } || \
     { [[ "${mem_unit^^}" == "M" ]] && (( mem_amount > 32768 )); }; then
    echo "[ERROR] la memoria massima richiedibile è 32G per nodo"
    exit 2
  fi
fi
if [[ -n "${gres_value}" && "${gres_value}" != "gpu:1" ]]; then
  echo "[ERROR] è consentita una sola GPU: usare --gres gpu:1"
  exit 2
fi

read -r -a make_tokens <<< "$make_args"
for make_token in "${make_tokens[@]}"; do
  if [[ "$make_token" != *=* ]]; then
    echo "[ERROR] invalid --make-args token (expected KEY=VALUE): $make_token" >&2
    exit 2
  fi
  make_key="${make_token%%=*}"
  make_value="${make_token#*=}"
  case "${make_key}" in
    SOLVE_MPI_RANKS)
      if [[ ! "${make_value}" =~ ^[1-9][0-9]*$ ]] || (( make_value > 4 )); then
        echo "[ERROR] SOLVE_MPI_RANKS deve essere compreso tra 1 e 4"
        exit 2
      fi
      ;;
    SOLVE_MPI_OMP_THREADS)
      if [[ ! "${make_value}" =~ ^[1-9][0-9]*$ ]] || (( make_value > 32 )); then
        echo "[ERROR] SOLVE_MPI_OMP_THREADS deve essere compreso tra 1 e 32"
        exit 2
      fi
      ;;
    SOLVE_SEQ_RUNTIME_S|SOLVE_MPI_RUNTIME_S|SOLVE_CUDA_RUNTIME_S)
      if [[ ! "${make_value}" =~ ^[1-9][0-9]*$ ]] || (( make_value > 300 )); then
        echo "[ERROR] ${make_key} deve essere compreso tra 1 e 300 secondi"
        exit 2
      fi
      ;;
  esac
done

# Target-aware defaults when not explicitly overridden:
# - CUDA targets: single node/task with one GPU
# - MPI/hybrid targets: four nodes and four ranks, with 32 threads per rank
# - OpenMP and serial targets: one node
if [[ "$target" == solve_cuda || "$target" == exp_cuda_* || "$target" == exp_practical_gpu || "$target" == run_scaling_tests ]]; then
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
elif [[ "$target" == solve_mpi || "$target" == solve_memory_growth_non_cuda || \
        "$target" == exp_strong_mpi || "$target" == exp_strong_hybrid || \
        "$target" == exp_weak_mpi || "$target" == exp_weak_hybrid || \
        "$target" == exp_all || "$target" == exp_practical_cpu ]]; then
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

mkdir -p "${root_dir}/results/slurm"
cd "$root_dir"

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
  printf '[DRY-RUN]'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
