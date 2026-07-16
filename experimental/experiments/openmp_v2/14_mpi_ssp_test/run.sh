#!/bin/bash

# MPI SSP Experiment (LOCAL SATURATION VERSION)
BIN_SYNC="./sync.out"
BIN_SSP="./ssp.out"
INSTANCE="instances/generated_benchmark/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=50

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
# 2 ranks x 12 threads = 24 cores
export OMP_NUM_THREADS=12
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE="guided,4"

# Compile Synchronous (Baseline)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c src/openmp-mpi/aco_v2.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SYNC

# Compile SSP (Asynchronous)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/14_mpi_ssp_test/aco_v2_ssp.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SSP

echo "----------------------------------------------------"
echo "ACO V2: MPI SSP Overlap Test (N=16000, 2 Ranks x 12T)"
echo "Targeting 24 Cores total."
echo "----------------------------------------------------"

# Run Synchronous
echo "Running Synchronous MPI (Blocking)..."
START=$(date +%s.%N)
# --bind-to none is CRITICAL for multi-threaded MPI ranks on single node
mpirun -np 2 --bind-to none $BIN_SYNC "$INSTANCE" "$K" "$M" "$SEED" > sync_out.txt 2>&1
END=$(date +%s.%N)
TIME_SYNC=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_SYNC=$(grep "Final Best" sync_out.txt | tail -n 1 | awk '{print $4}')
printf "Sync Time/Epoch: %.2f ms | Cost: %s\n" "$TIME_SYNC" "$COST_SYNC"

# Run SSP
echo "Running Asynchronous SSP (Non-blocking)..."
START=$(date +%s.%N)
mpirun -np 2 --bind-to none $BIN_SSP "$INSTANCE" "$K" "$M" "$SEED" > ssp_out.txt 2>&1
END=$(date +%s.%N)
TIME_SSP=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_SSP=$(grep "Final Best" ssp_out.txt | tail -n 1 | awk '{print $4}')
printf "SSP Time/Epoch:  %.2f ms | Cost: %s\n" "$TIME_SSP" "$COST_SSP"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_SYNC / $TIME_SSP" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Overlap): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_SYNC $BIN_SSP sync_out.txt ssp_out.txt
