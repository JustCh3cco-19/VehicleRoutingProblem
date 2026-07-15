#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: tools/batch/submit_practical_campaign.sh [options]

Sottomette automaticamente due job Slurm:
  1) CPU campaign  -> target make: exp_practical_cpu
  2) GPU campaign  -> target make: exp_practical_gpu

Opzioni:
  --tag TAG                Tag risultati (default: YYYYMMDD_HHMMSS)
  --cpu-time HH:MM:SS      Tempo job CPU (default: 00:30:00)
  --gpu-time HH:MM:SS      Tempo job GPU (default: 00:30:00)
  --module-loads "LIST"    Moduli da caricare (es: "gcc/13.2 openmpi/4.1 cuda/12.2")
  --dry-run                Mostra i comandi senza inviare job
  -h, --help               Mostra help
EOF
}

tag="$(date +%Y%m%d_%H%M%S)"
cpu_time="00:30:00"
gpu_time="00:30:00"
module_loads=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      tag="${2:-}"
      shift 2
      ;;
    --cpu-time)
      cpu_time="${2:-}"
      shift 2
      ;;
    --gpu-time)
      gpu_time="${2:-}"
      shift 2
      ;;
    --module-loads)
      module_loads="${2:-}"
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submit_solve="${script_dir}/submit_solve.sh"

if [[ ! -x "${submit_solve}" ]]; then
  echo "[ERROR] file non eseguibile o non trovato: ${submit_solve}"
  exit 1
fi

cpu_cmd=(
  "${submit_solve}"
  --target exp_practical_cpu
  --time "${cpu_time}"
  --nodes 4
  --ntasks 4
  --cpus 32
  --mem 32G
  --make-args "PRACTICAL_TAG=${tag}"
)
gpu_cmd=(
  "${submit_solve}"
  --target exp_practical_gpu
  --time "${gpu_time}"
  --nodes 1
  --ntasks 1
  --cpus 32
  --mem 32G
  --gres gpu:1
  --make-args "PRACTICAL_TAG=${tag}"
)

if [[ -n "${module_loads}" ]]; then
  cpu_cmd+=(--module-loads "${module_loads}")
  gpu_cmd+=(--module-loads "${module_loads}")
fi
if [[ "${dry_run}" -eq 1 ]]; then
  cpu_cmd+=(--dry-run)
  gpu_cmd+=(--dry-run)
fi

echo "[INFO] campaign tag=${tag}"
echo "[INFO] results root prefix: results/practical_campaign/${tag}/"
echo "[INFO] submitting CPU job..."
"${cpu_cmd[@]}"

echo "[INFO] submitting GPU job..."
"${gpu_cmd[@]}"

if [[ "${dry_run}" -eq 1 ]]; then
  echo "[INFO] dry-run completed."
  exit 0
fi

cat <<EOF
[INFO] submission completed.
[INFO] when jobs finish, run:
  python3 tools/python/summarize_practical_experiments.py --root results/practical_campaign/${tag}/cpu
  python3 tools/python/summarize_practical_experiments.py --root results/practical_campaign/${tag}/gpu
EOF
