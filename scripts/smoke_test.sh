#!/bin/bash
set -e

echo "--- SMOKE TEST ---"

# 1. Compile
echo "Step 1: Compiling openmp_mpi..."
make openmp_mpi

if [ ! -f aco_vrp_openmp_mpi.out ]; then
    echo "Error: Binary not found."
    exit 1
fi

# 2. Run small problem
echo "Step 2: Running small problem (N=100, K=5, M=20, T=5) with 4 threads..."
export OMP_NUM_THREADS=4
# Set termination conditions to avoid infinite loops if logic is broken
export ACO_SOLVER_TIMEOUT_SECONDS=10
export ACO_SOLVER_STAGNATION_EPOCHS=10

# We'll use timeout command as a safety net
timeout 15s mpirun -np 1 ./aco_vrp_openmp_mpi.out 100 5 20 1234 > /tmp/smoke_test.log 2>&1 || {
    echo "FAILURE: Test timed out or crashed."
    cat /tmp/smoke_test.log
    exit 1
}

if grep -q "best cost:" /tmp/smoke_test.log; then
    echo "SUCCESS: Solver produced output."
    grep "best cost:" /tmp/smoke_test.log
else
    echo "FAILURE: No output found in log."
    cat /tmp/smoke_test.log
    exit 1
fi

echo "--- SMOKE TEST PASSED ---"
