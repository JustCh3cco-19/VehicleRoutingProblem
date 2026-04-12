#!/bin/bash

# Post-Float Strong Scaling Experiment
BIN="./aco_vrp_v2_openmp_mpi.out"
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536 # 64 ants/core
SEED=1234
EPOCHS=20

THREAD_VALS=("1" "4" "12" "24")

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_SCHEDULE="guided,4"

echo "-----------------------------------------------------------------------"
echo "ACO V2: POST-FLOAT STRONG SCALING (N=16000, M=1536)"
echo "-----------------------------------------------------------------------"
printf "%-10s | %-15s | %-10s\n" "Threads" "Time/Epoch (ms)" "Efficiency"
echo "-----------------------------------------------------------------------"

T1=0

for t in "${THREAD_VALS[@]}"; do
    export OMP_NUM_THREADS=$t
    START=$(date +%s.%N)
    $BIN "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
    END=$(date +%s.%N)
    
    TIME=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
    
    if [ "$t" == "1" ]; then
        T1=$TIME
    fi
    
    EFF=$(echo "scale=2; $T1 / ($TIME * $t)" | bc -l)
    
    printf "%-10s | %-15.2f | %-10s\n" "$t" "$TIME" "$EFF"
done
echo "-----------------------------------------------------------------------"
