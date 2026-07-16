#!/bin/bash

# Multi-Size Granularity Sweep
BIN="./aco_vrp_v2_openmp_mpi.out"
EPOCHS=10
SEED=1234

# Problems
SIZES=("2000" "8000" "16000")
INSTANCES=("instances/generated_benchmark/n2000_k8_s19002.vrp" "instances/generated_benchmark/n8000_k8_s19004.vrp" "instances/generated_benchmark/n16000_k16_s19005.vrp")
K_VALS=("8" "8" "16")

# Ants per core ratios to test (for 24 cores)
# 4 ants/core  => M = 96
# 16 ants/core => M = 384
# 64 ants/core => M = 1536
DENSITIES=("4" "16" "64")

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_SCHEDULE="guided,1"

echo "-----------------------------------------------------------------------"
echo "ACO V2: GRANULARITY SWEEP (N vs Ants/Core)"
echo "-----------------------------------------------------------------------"
printf "%-10s | %-12s | %-10s | %-15s | %-10s\n" "Size (N)" "Ants/Core" "Ants (M)" "Time/Epoch(ms)" "Efficiency"
echo "-----------------------------------------------------------------------"

for i in "${!SIZES[@]}"; do
    N=${SIZES[$i]}
    INST=${INSTANCES[$i]}
    K=${K_VALS[$i]}
    
    for d in "${DENSITIES[@]}"; do
        M=$((d * 24))
        
        # 1. Baseline (1 Thread)
        export OMP_NUM_THREADS=1
        START=$(date +%s.%N)
        $BIN "$INST" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        T1=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
        
        # 2. 24 Threads
        export OMP_NUM_THREADS=24
        START=$(date +%s.%N)
        $BIN "$INST" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        T24=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
        
        EFF=$(echo "scale=2; $T1 / ($T24 * 24)" | bc -l)
        
        printf "%-10s | %-12s | %-10s | %-15.2f | %-10s\n" "$N" "$d" "$M" "$T24" "$EFF"
    done
    echo "-----------------------------------------------------------------------"
done
