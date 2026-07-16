#!/bin/bash

# SIMD AVX2 Experiment
BIN_PREFETCH="./prefetch.out"
BIN_SIMD="./simd.out"
INSTANCE="instances/generated_benchmark/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Prefetch (Current Best)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/09_prefetching_test/aco_v2_prefetch.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_PREFETCH

# Compile SIMD (New Challenger)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -mavx2 -mfma -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/10_simd_avx2_test/aco_v2_simd.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SIMD

echo "----------------------------------------------------"
echo "ACO V2: SIMD AVX2 Test (N=16000, 24 Threads)"
echo "----------------------------------------------------"

# Run Prefetch
echo "Running Prefetch Optimized..."
START=$(date +%s.%N)
$BIN_PREFETCH "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_PRE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "Prefetch Time/Epoch: %.2f ms\n" "$TIME_PRE"

# Run SIMD
echo "Running SIMD Optimized..."
START=$(date +%s.%N)
$BIN_SIMD "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
END=$(date +%s.%N)
TIME_SIMD=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
printf "SIMD Time/Epoch: %.2f ms\n" "$TIME_SIMD"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_PRE / $TIME_SIMD" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (vs Prefetch): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_PREFETCH $BIN_SIMD
