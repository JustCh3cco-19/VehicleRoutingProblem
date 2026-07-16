#!/bin/bash

# Hierarchical Bitmask Experiment
BIN_BASELINE="./prefetch.out"
BIN_H_MASK="./hierarchical.out"
INSTANCE="instances/generated_benchmark/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Baseline (V2 Prefetch)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/09_prefetching_test/aco_v2_prefetch.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_BASELINE

# Compile Hierarchical
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/12_hierarchical_bitmask/aco_v2_hierarchical.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_H_MASK

echo "----------------------------------------------------"
echo "ACO V2: Hierarchical Bitmask Test (N=16000, 24 Threads)"
echo "----------------------------------------------------"

# Run Baseline
echo "Running Baseline (Prefetch Only)..."
START=$(date +%s.%N)
$BIN_BASELINE "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_BASE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Baseline Time/Epoch: %.2f ms\n" "$TIME_BASE"

# Run Hierarchical
echo "Running Hierarchical Bitmask..."
START=$(date +%s.%N)
$BIN_H_MASK "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_H=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Hierarchical Time/Epoch: %.2f ms\n" "$TIME_H"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_BASE / $TIME_H" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Meta-Masking): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_BASELINE $BIN_H_MASK
