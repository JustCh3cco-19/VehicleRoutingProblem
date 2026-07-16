#!/bin/bash

# Prefetching Experiment
BIN_BASELINE="./baseline.out"
BIN_PREFETCH="./prefetch.out"
INSTANCE="instances/generated_benchmark/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Baseline
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/09_prefetching_test/aco_v2_baseline.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_BASELINE

# Compile Prefetch
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/09_prefetching_test/aco_v2_prefetch.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_PREFETCH

echo "----------------------------------------------------"
echo "ACO V2: Prefetching Test (N=16000, 24 Threads)"
echo "----------------------------------------------------"

# Run Baseline
echo "Running Baseline (No Prefetch)..."
START=$(date +%s.%N)
$BIN_BASELINE "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_BASE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Baseline Time/Epoch: %.2f ms\n" "$TIME_BASE"

# Run Prefetch
echo "Running Prefetch Optimized..."
START=$(date +%s.%N)
$BIN_PREFETCH "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_PRE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Prefetch Time/Epoch: %.2f ms\n" "$TIME_PRE"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_BASE / $TIME_PRE" | bc -l)
echo "----------------------------------------------------"
echo "Speedup: ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_BASELINE $BIN_PREFETCH
