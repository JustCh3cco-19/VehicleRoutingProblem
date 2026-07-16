#!/bin/bash

# Configuration
INSTANCE="instances/test_aligned/n8000_k8_s19004.vrp"
BIN="./aco_vrp_v2_openmp_mpi.out"
K=8
M=768
SEED=1234
EPOCHS=20
REPEATS=2

# Solver settings
export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24

echo "----------------------------------------------------"
echo "ACO V2 SCALING UP (N=8000)"
echo "Problem: N=8000, Ants (M)=768, Epochs=$EPOCHS"
echo "----------------------------------------------------"

STRATEGIES=("guided,1" "guided,4" "guided,8" "guided,16")

printf "%-15s | %-10s | %-15s\n" "Strategy" "Run" "Time per Epoch (ms)"
echo "----------------------------------------------------"

for sched in "${STRATEGIES[@]}"; do
    export OMP_SCHEDULE="$sched"
    for (( r=1; r<=$REPEATS; r++ )); do
        START=$(date +%s.%N)
        $BIN "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        
        TOTAL_MS=$(echo "($END - $START) * 1000" | bc)
        TIME_PER_EPOCH=$(echo "scale=2; $TOTAL_MS / $EPOCHS" | bc)
        
        printf "%-15s | %-10s | %-15s\n" "$sched" "$r" "$TIME_PER_EPOCH"
    done
    echo "----------------------------------------------------"
done
