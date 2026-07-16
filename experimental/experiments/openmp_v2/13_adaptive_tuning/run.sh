#!/bin/bash

# Adaptive Tuning Experiment
BIN_BASELINE="./baseline.out"
BIN_ADAPTIVE="./adaptive.out"
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Baseline (Old Tuning)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c src/openmp-mpi/aco_v2.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_BASELINE

# Compile Adaptive (Reasoned Tuning)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/13_adaptive_tuning/aco_v2_adaptive.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_ADAPTIVE

echo "----------------------------------------------------"
echo "ACO V2: Adaptive Tuning Test (N=16000, 24 Threads)"
echo "----------------------------------------------------"

# Run Baseline
echo "Running Baseline (Old Tuning)..."
$BIN_BASELINE "$INSTANCE" "$K" "$M" "$SEED" 2>&1 | grep -E "Epoch|Time" | head -n 5
START=$(date +%s.%N)
$BIN_BASELINE "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_BASE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Baseline Time/Epoch: %.2f ms\n" "$TIME_BASE"

# Run Adaptive
echo "Running Adaptive (Reasoned Tuning)..."
# We want to see what K it picked
# Let's add a temporary print in aco_v2_adaptive.c if we could, 
# but let's just measure performance for now.
START=$(date +%s.%N)
$BIN_ADAPTIVE "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_ADAP=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Adaptive Time/Epoch: %.2f ms\n" "$TIME_ADAP"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_BASE / $TIME_ADAP" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Adaptive): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_BASELINE $BIN_ADAPTIVE
