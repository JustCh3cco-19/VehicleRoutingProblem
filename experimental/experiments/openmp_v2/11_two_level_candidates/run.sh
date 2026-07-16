#!/bin/bash

# Two-Level Candidates Experiment
BIN_PREFETCH="./prefetch.out"
BIN_TWO_LEVEL="./two_level.out"
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Prefetch (Baseline attuale)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/09_prefetching_test/aco_v2_prefetch.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_PREFETCH

# Compile Two-Level (Sparsificazione)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/11_two_level_candidates/aco_v2_two_level.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_TWO_LEVEL

echo "----------------------------------------------------"
echo "ACO V2: Two-Level Candidates Test (N=16000, 24 Threads)"
echo "----------------------------------------------------"

# Run Prefetch
echo "Running Baseline (Prefetch Only)..."
START=$(date +%s.%N)
$BIN_PREFETCH "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_PRE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Baseline Time/Epoch: %.2f ms\n" "$TIME_PRE"

# Run Two-Level
echo "Running Two-Level Candidates (512 + 2048)..."
START=$(date +%s.%N)
$BIN_TWO_LEVEL "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_TL=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Two-Level Time/Epoch: %.2f ms\n" "$TIME_TL"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_PRE / $TIME_TL" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Sparsification): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_PREFETCH $BIN_TWO_LEVEL
