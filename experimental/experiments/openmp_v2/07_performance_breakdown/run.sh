#!/bin/bash

# Performance Breakdown Experiment (N=16000)
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536 # 64 ants/core
SEED=1234
EPOCHS=10

# 1. Compile and Run DOUBLE version
echo "Compiling profiled V2 (DOUBLE)..."
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c \
    experimental/experiments/openmp_v2/07_performance_breakdown/aco_v2_profiled.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o experimental/experiments/openmp_v2/07_performance_breakdown/aco_double.out

# 2. Compile and Run FLOAT version
echo "Compiling profiled V2 (FLOAT)..."
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c \
    experimental/experiments/openmp_v2/07_performance_breakdown/aco_v2_float.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o experimental/experiments/openmp_v2/07_performance_breakdown/aco_float.out

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

echo "===================================================="
echo "RUNNING DOUBLE VERSION"
echo "===================================================="
./experimental/experiments/openmp_v2/07_performance_breakdown/aco_double.out "$INSTANCE" "$K" "$M" "$SEED" | grep -A 10 "V2 PERFORMANCE"

echo ""
echo "===================================================="
echo "RUNNING FLOAT VERSION"
echo "===================================================="
./experimental/experiments/openmp_v2/07_performance_breakdown/aco_float.out "$INSTANCE" "$K" "$M" "$SEED" | grep -A 10 "V2 FLOAT PERFORMANCE"
