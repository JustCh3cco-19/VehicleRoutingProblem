#!/bin/bash

# Multi-Size Scheduling Experiment
BIN="./aco_vrp_v2_openmp_mpi.out"
M=768
SEED=1234
EPOCHS=10
REPEATS=1 # Singola run per velocità, ma epoche fisse per stabilità

# Instances and their K
SIZES=("2000" "4000" "8000" "16000")
INSTANCES=("instances/generated_benchmark/n2000_k8_s19002.vrp" "instances/generated_benchmark/n4000_k8_s19003.vrp" "instances/generated_benchmark/n8000_k8_s19004.vrp" "instances/generated_benchmark/n16000_k16_s19005.vrp")
K_VALS=("8" "8" "8" "16")

# Strategies
SCHED_LIST=("dynamic,1" "dynamic,4" "guided,1" "guided,4")

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24

echo "-----------------------------------------------------------------------"
echo "ACO V2: SCHEDULING STRATEGY VS PROBLEM SIZE (N=2k to 16k)"
echo "-----------------------------------------------------------------------"
printf "%-10s | %-12s | %-20s\n" "Size (N)" "Strategy" "Time/Epoch (ms)"
echo "-----------------------------------------------------------------------"

for i in "${!SIZES[@]}"; do
    N=${SIZES[$i]}
    INST=${INSTANCES[$i]}
    K=${K_VALS[$i]}
    
    for sched in "${SCHED_LIST[@]}"; do
        export OMP_SCHEDULE="$sched"
        
        START=$(date +%s.%N)
        $BIN "$INST" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        
        TOTAL_MS=$(echo "($END - $START) * 1000" | bc)
        TIME_PER_EPOCH=$(echo "scale=2; $TOTAL_MS / $EPOCHS" | bc)
        
        printf "%-10s | %-12s | %-20s\n" "$N" "$sched" "$TIME_PER_EPOCH"
    done
    echo "-----------------------------------------------------------------------"
done
