#!/bin/bash
set -e

# Configuration
NODES=${1:-20}
VEHICLES=${2:-5}
ANTS=${3:-64}
ITERS=${4:-100}
INSTANCE="problem.vrp"
SOLUTION="solution.txt"

# 1. Generate problem and reference solution
echo "--- Generating problem and reference solution ---"
python3 scripts/generate_vrp_problem.py --nodes $NODES --vehicles $VEHICLES --output $INSTANCE --seed 42 --solve --runtime 5.0
REF_SOLUTION="problem_solution.txt"

# 2. Build solver
echo "--- Building solver ---"
make cuda-vrp

# 3. Run solver
echo "--- Running CUDA solver ---"
./aco_vrp_cuda_vrp $INSTANCE $VEHICLES $ANTS $ITERS 42 > $SOLUTION

# 4. Show CUDA Summary
echo "--- CUDA Solution Summary ---"
NUM_ROUTES=$(grep -c "Route" $SOLUTION)
USED_ROUTES=$(grep "Route" $SOLUTION | grep -vE ":[[:space:]]*$" | wc -l)
echo "Total Vehicles: $VEHICLES, Vehicles Used: $USED_ROUTES"
echo "Routes Found:"
grep "Route" $SOLUTION | grep -vE ":[[:space:]]*$"

# 5. Clean up C output for validator
sed -i '/Route/,$!d' $SOLUTION

# 6. Validate with pyvrp and compare with reference
echo "--- Validating and Comparing ---"
python3 scripts/validate_pyvrp.py $INSTANCE $SOLUTION --round-func none --reference $REF_SOLUTION

echo "--- SUCCESS ---"
