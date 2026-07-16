#!/bin/bash

# Ants Granularity Experiment
BIN="./aco_vrp_v2_openmp_mpi.out"
INSTANCE="instances/test_aligned/n2000_k8_s19002.vrp"
K=8
SEED=1234
EPOCHS=20

ANTS_VALS=("24" "96" "384" "1536")
THREAD_VALS=("1" "4" "12" "24")

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_SCHEDULE="guided,1"

echo "-----------------------------------------------------------------------"
echo "ACO V2: ANTS GRANULARITY SWEEP (M vs Threads)"
echo "-----------------------------------------------------------------------"
printf "%-10s | %-10s | %-15s | %-10s\n" "Ants (M)" "Threads" "Time/Epoch (ms)" "Efficiency"
echo "-----------------------------------------------------------------------"

for m in "${ANTS_VALS[@]}"; do
    # Baseline for this M (1 thread)
    export OMP_NUM_THREADS=1
    START=$(date +%s.%N)
    $BIN "$INSTANCE" "$K" "$m" "$SEED" > /dev/null 2>&1
    END=$(date +%s.%N)
    T1=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
    
    printf "%-10s | %-10s | %-15.2f | %-10s\n" "$m" "1" "$T1" "1.00"

    for t in "${THREAD_VALS[@]}"; do
        if [ "$t" == "1" ]; then continue; fi
        
        export OMP_NUM_THREADS=$t
        START=$(date +%s.%N)
        $BIN "$INSTANCE" "$K" "$m" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        
        TIME=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
        EFF=$(echo "scale=2; $T1 / ($TIME * $t)" | bc -l)
        
        printf "%-10s | %-10s | %-15.2f | %-10s\n" "$m" "$t" "$TIME" "$EFF"
    done
    echo "-----------------------------------------------------------------------"
done
