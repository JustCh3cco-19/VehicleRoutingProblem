#!/bin/bash

# Configuration
INSTANCE="instances/generated_benchmark/n2000_k8_s19002.vrp"
K=8
M=768
SEED=1234
REPEATS=3

# Solver settings
export ACO_SOLVER_TIMEOUT_SECONDS=30
export ACO_SOLVER_STAGNATION_EPOCHS=100
export ACO_SOLVER_MIN_REL_IMPROVEMENT=0.001
export OMP_NUM_THREADS=24

echo "----------------------------------------------------"
echo "ACO V2 Scheduling Experiment"
echo "Problem: N=2000, M=768, Threads=24"
echo "----------------------------------------------------"

STRATEGIES=("dynamic,1" "dynamic,4" "guided,1")

printf "%-15s | %-10s | %-10s\n" "Strategy" "Run" "Time (s)"
echo "----------------------------------------------------"

# Paths relative to project root
BIN="./aco_vrp_v2_openmp_mpi.out"
INSTANCE="instances/generated_benchmark/n2000_k8_s19002.vrp"

for sched in "${STRATEGIES[@]}"; do
    export OMP_SCHEDULE="$sched"
    for (( r=1; r<=$REPEATS; r++ )); do
        # Misura il tempo
        START=$(date +%s.%N)
        $BIN "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        ELAPSED=$(echo "$END - $START" | bc)
        
        printf "%-15s | %-10s | %-10s\n" "$sched" "$r" "$ELAPSED"
    done
    echo "----------------------------------------------------"
done
