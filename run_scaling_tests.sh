#!/bin/bash
set -e

export ACO_SOLVER_TIMEOUT_SECONDS=300
export ACO_SOLVER_STAGNATION_EPOCHS=100
export ACO_SOLVER_MIN_REL_IMPROVEMENT=0.05

instances=(
  "instances/test_aligned/n2000_k8_s19002.vrp 8 256"
  "instances/test_aligned/n8000_k8_s19004.vrp 8 512"
  "instances/test_aligned/n32000_k32_s19006.vrp 32 512"
  "instances/test_aligned/n64000_k63_s19007.vrp 63 512"
  "instances/test_aligned/n128000_k125_s19008.vrp 125 512"
)

mkdir -p results_scaling

echo "Starting gradual scaling tests..."

for inst in "${instances[@]}"; do
  read -r file k m <<< "$inst"
  name=$(basename "$file" .vrp)
  echo "====================================="
  echo "Testing $name (K=$k, M=$m)"
  echo "====================================="
  
  # Run normally to measure total runtime and iterations
  echo "Running solver..."
  time ./aco_vrp_cuda.out "$file" "$k" "$m" 1234 > "results_scaling/run_${name}.txt" 2>&1
  
  # Run ncu to measure specific metrics
  echo "Profiling with ncu..."
  # Just run 1 launch to get metrics without taking hours
  ncu --launch-count 1 --set full -k kernel_construct_solutions_v6 ./aco_vrp_cuda.out "$file" "$k" "$m" 1234 > "results_scaling/ncu_${name}.txt" 2>&1
  
  echo "Done testing $name."
done

echo "All tests completed."
