#!/usr/bin/env bash
# Automated Guided Profiling script for VRP ACO Solver
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORTS_DIR="${ROOT_DIR}/reports/profile"
mkdir -p "$REPORTS_DIR"

# Check available tools
HAS_GPROF=0
if command -v gprof >/dev/null 2>&1; then
  HAS_GPROF=1
fi

HAS_NSYS=0
if command -v nsys >/dev/null 2>&1; then
  HAS_NSYS=1
fi

HAS_NCU=0
if command -v ncu >/dev/null 2>&1; then
  HAS_NCU=1
fi

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
fi

echo "============================================================"
echo "Automating Guided Profiling (Point 1)"
echo "============================================================"
echo "Available tools:"
echo "  gprof:       $([ $HAS_GPROF -eq 1 ] && echo 'FOUND' || echo 'NOT FOUND')"
echo "  nsys:        $([ $HAS_NSYS -eq 1 ] && echo 'FOUND' || echo 'NOT FOUND')"
echo "  ncu:         $([ $HAS_NCU -eq 1 ] && echo 'FOUND' || echo 'NOT FOUND')"
echo "  NVIDIA GPU:  $([ $HAS_GPU -eq 1 ] && echo 'FOUND' || echo 'NOT FOUND')"
echo "============================================================"

# 1. CPU Profiling with gprof
if [ $HAS_GPROF -eq 1 ]; then
  echo "--> Running CPU profiling with gprof..."
  # Rebuild sequential with profiling flags
  make -C "$ROOT_DIR" clean >/dev/null 2>&1
  make -C "$ROOT_DIR" seq EXTRA_FLAGS="-pg" FORCE_OPT="-O3" >/dev/null 2>&1

  # Run on a small/medium demo problem to generate gmon.out
  echo "  Running sequential solver..."
  ACO_SOLVER_TIMEOUT_SECONDS=5 ACO_SOLVER_STAGNATION_EPOCHS=100 \
    "$ROOT_DIR/seq.out" 200 5 16 1234 > /dev/null 2>&1 || true

  # Run gprof and write output
  if [ -f "${ROOT_DIR}/gmon.out" ]; then
    gprof "$ROOT_DIR/seq.out" "${ROOT_DIR}/gmon.out" > "${REPORTS_DIR}/cpu_profile_gprof.txt"
    echo "  SUCCESS: CPU profile saved to reports/profile/cpu_profile_gprof.txt"
    rm -f "${ROOT_DIR}/gmon.out"
  else
    echo "  WARNING: gmon.out not generated."
  fi
  # Clean up profiling build
  make -C "$ROOT_DIR" clean >/dev/null 2>&1
else
  echo "--> Skipping CPU profiling: gprof not found."
fi

# 2. GPU Profiling with Nsight Systems & Compute
if [ $HAS_GPU -eq 1 ] && { [ $HAS_NSYS -eq 1 ] || [ $HAS_NCU -eq 1 ]; }; then
  # Rebuild CUDA
  echo "--> Building CUDA binary..."
  make -C "$ROOT_DIR" cuda CUDA_ARCH=sm_75 >/dev/null 2>&1

  if [ $HAS_NSYS -eq 1 ]; then
    echo "--> Running GPU profiling with Nsight Systems (nsys)..."
    nsys profile \
      -o "${REPORTS_DIR}/nsys_cvrp_profile" \
      --force-overwrite true \
      --stats=true \
      "$ROOT_DIR/cuda.out" 100 3 16 1234 >/dev/null 2>&1 || true
    echo "  SUCCESS: Nsys report saved to reports/profile/nsys_cvrp_profile.nsys-rep"
  fi

  if [ $HAS_NCU -eq 1 ]; then
    echo "--> Running GPU profiling with Nsight Compute (ncu)..."
    # Run NCU
    ncu \
      -o "${REPORTS_DIR}/ncu_cvrp_profile" \
      --force-overwrite \
      "$ROOT_DIR/cuda.out" 100 3 16 1234 >/dev/null 2>&1 || true
    echo "  SUCCESS: Ncu report saved to reports/profile/ncu_cvrp_profile.ncu-rep"
  fi
else
  echo "--> Skipping GPU profiling: No visible GPU or missing nsys/ncu tools."
fi

echo "============================================================"
echo "Profiling complete. Reports stored in: reports/profile/"
echo "============================================================"
