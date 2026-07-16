#!/bin/bash

# Configuration
INSTANCE="instances/generated_benchmark/n2000_k8_s19002.vrp"
K=8
M=768
SEED=1234
EPOCHS=100
REPEATS=2

# Scientific Solver settings
export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_SCHEDULE="guided,1"

echo "----------------------------------------------------"
echo "ACO V2 SCIENTIFIC STRONG SCALING (FIXED EPOCHS)"
echo "Problem: N=2000, Ants (M)=768, Epochs=$EPOCHS"
echo "----------------------------------------------------"

printf "%-10s | %-10s | %-15s | %-10s\n" "Threads" "Run" "Time per Epoch (ms)" "Efficiency"
echo "----------------------------------------------------"

# Paths relative to project root
BIN="./aco_vrp_v2_openmp_mpi.out"
INSTANCE="instances/generated_benchmark/n2000_k8_s19002.vrp"

T1_MS=0
THREAD_COUNTS=(1 2 4 8 12 16 20 24)

for t in "${THREAD_COUNTS[@]}"; do
    export OMP_NUM_THREADS=$t
    for (( r=1; r<=$REPEATS; r++ )); do
        START=$(date +%s.%N)
        $BIN "$INSTANCE" "$K" "$M" "$SEED" > /dev/null 2>&1
        END=$(date +%s.%N)
        
        # Tempo totale in millisecondi
        TOTAL_MS=$(echo "($END - $START) * 1000" | bc)
        TIME_PER_EPOCH=$(echo "scale=2; $TOTAL_MS / $EPOCHS" | bc)
        
        if [ $t -eq 1 ] && [ $r -eq 1 ]; then
            T1_MS=$TIME_PER_EPOCH
        fi
        
        EFF=$(echo "scale=2; ($T1_MS / ($TIME_PER_EPOCH * $t))" | bc)
        
        printf "%-10s | %-10s | %-19s | %-10s\n" "$t" "$r" "$TIME_PER_EPOCH" "$EFF"
    done
    echo "----------------------------------------------------"
done
