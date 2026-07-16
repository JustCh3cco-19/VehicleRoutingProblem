#!/bin/bash
set -e

export ACO_SOLVER_TIMEOUT_SECONDS=300
export ACO_SOLVER_STAGNATION_EPOCHS=100
export ACO_SOLVER_MIN_REL_IMPROVEMENT=5

instances=(
  "instances/generated_benchmark/n2000_k8_s19002.vrp 8 256"
  "instances/generated_benchmark/n8000_k8_s19004.vrp 8 512"
  "instances/generated_benchmark/n32000_k32_s19006.vrp 32 512"
  "instances/generated_benchmark/n64000_k63_s19007.vrp 63 512"
  "instances/generated_benchmark/n100000_k98_s19008.vrp 98 512"
)

results_root="${RESULTS_ROOT:-results}"
out_dir="${SCALING_TESTS_OUT_DIR:-${results_root}/scaling_tests}"
mkdir -p "$out_dir"

echo "Starting gradual scaling tests..."

for inst in "${instances[@]}"; do
  read -r file k m <<< "$inst"
  name=$(basename "$file" .vrp)
  echo "====================================="
  echo "Testing $name (K=$k, M=$m)"
  echo "====================================="
  
  # Run normally to measure total runtime and iterations
  echo "Running solver..."
  time ./aco_vrp_cuda.out "$file" "$k" "$m" 1234 > "${out_dir}/run_${name}.txt" 2>&1
  
  # Run ncu to measure specific metrics
  echo "Profiling with ncu..."
  # Just run 1 launch to get metrics without taking hours
  ncu --launch-count 1 --set full -k kernel_construct_solutions_v6 ./aco_vrp_cuda.out "$file" "$k" "$m" 1234 > "${out_dir}/ncu_${name}.txt" 2>&1
  
  echo "Done testing $name."
done

echo "All tests completed."
