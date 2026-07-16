#!/usr/bin/env bash
set -euo pipefail

if ! command -v timeout >/dev/null 2>&1; then
  echo "[ERROR] required command not found: timeout" >&2
  exit 2
fi

backend="${1:-mpi}"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/vrp_smoke.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

instance="$tmp_dir/smoke.vrp"
log="$tmp_dir/${backend}.log"

cat > "$instance" <<'VRP'
NAME : smoke
TYPE : CVRP
DIMENSION : 9
VEHICLES : 2
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 5
NODE_COORD_SECTION
1 50 50
2 10 10
3 20 10
4 10 20
5 20 20
6 80 80
7 90 80
8 80 90
9 90 90
DEMAND_SECTION
1 0
2 1
3 1
4 1
5 1
6 1
7 1
8 1
9 1
DEPOT_SECTION
1
-1
EOF
VRP

check_output() {
  local expected_routes="$1"

  if ! grep -Eq '^Cost: [0-9]+([.][0-9]+)?$' "$log"; then
    echo "FAILURE: missing numeric Cost line for $backend"
    cat "$log"
    exit 1
  fi

  if ! grep -Eq '^best cost: [0-9]+([.][0-9]+)?$' "$log"; then
    echo "FAILURE: missing numeric best cost line for $backend"
    cat "$log"
    exit 1
  fi

  local route_count
  route_count="$(grep -Ec '^Route [0-9]+:' "$log")"
  if [ "$route_count" -ne "$expected_routes" ]; then
    echo "FAILURE: expected $expected_routes routes for $backend, got $route_count"
    cat "$log"
    exit 1
  fi
}

run_seq() {
  echo "[smoke_seq] building seq"
  make -C "$root_dir" seq

  echo "[smoke_seq] running solver"
  ACO_SOLVER_TIMEOUT_SECONDS=10 \
  ACO_SOLVER_STAGNATION_EPOCHS=10 \
  timeout 20s "$root_dir/aco_vrp_seq.out" "$instance" 2 8 1234 > "$log" 2>&1
  check_output 2
  grep '^best cost:' "$log"
}

run_mpi() {
  if ! command -v mpirun >/dev/null 2>&1; then
    echo "[smoke_mpi] skipped: mpirun not found"
    return 0
  fi

  if ! timeout 10s mpirun -np 1 true > "$tmp_dir/mpirun_preflight.log" 2>&1; then
    echo "[smoke_mpi] skipped: mpirun cannot start in this environment"
    cat "$tmp_dir/mpirun_preflight.log"
    return 0
  fi

  echo "[smoke_mpi] building openmp_mpi"
  make -C "$root_dir" openmp_mpi

  echo "[smoke_mpi] running solver"
  OMP_NUM_THREADS="${SMOKE_MPI_OMP_THREADS:-2}" \
  ACO_SOLVER_TIMEOUT_SECONDS=10 \
  ACO_SOLVER_STAGNATION_EPOCHS=10 \
  timeout 20s mpirun -np "${SMOKE_MPI_RANKS:-1}" "$root_dir/aco_vrp_openmp_mpi.out" "$instance" 2 8 1234 > "$log" 2>&1
  check_output 2
  grep '^best cost:' "$log"
}

run_cuda() {
  if ! command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    echo "[smoke_cuda] skipped: nvcc not found"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[smoke_cuda] skipped: no CUDA GPU visible"
    return 0
  fi

  echo "[smoke_cuda] building cuda"
  make -C "$root_dir" cuda

  echo "[smoke_cuda] running solver"
  ACO_SOLVER_TIMEOUT_SECONDS=10 \
  ACO_SOLVER_STAGNATION_EPOCHS=10 \
  timeout 20s "$root_dir/aco_vrp_cuda.out" "$instance" 2 8 1234 > "$log" 2>&1
  check_output 2
  grep '^best cost:' "$log"
}

case "$backend" in
  seq) run_seq ;;
  mpi) run_mpi ;;
  cuda) run_cuda ;;
  all)
    run_seq
    run_mpi
    run_cuda
    ;;
  *)
    echo "usage: $0 {seq|mpi|cuda|all}" >&2
    exit 2
    ;;
esac
