#!/bin/bash

# Sparse SSP Final Experiment
BIN_SYNC="./sparse_sync.out"
BIN_SSP="./sparse_ssp.out"
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=12
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE="guided,4"

# Compile Sparse Sync (Blocking challenger)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experiments/openmp_v2/16_mpi_sparse_sync_impl/aco_v2_sparse.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SYNC

# Compile Sparse SSP (Non-blocking challenger)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experiments/openmp_v2/17_mpi_sparse_ssp_final/aco_v2_sparse_ssp.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SSP

echo "----------------------------------------------------"
echo "ACO V2: Sparse vs Sparse-SSP Test (N=16000, 2 Ranks x 12T)"
echo "----------------------------------------------------"

# Run Sync
echo "Running Sparse Sync (Synchronous)..."
START=$(date +%s.%N)
mpirun -np 2 --bind-to none $BIN_SYNC "$INSTANCE" "$K" "$M" "$SEED" > sync_out.txt 2>&1
END=$(date +%s.%N)
TIME_SYNC=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_SYNC=$(grep "Best:" sync_out.txt | tail -n 1 | awk '{print $4}')
printf "Sync Time/Epoch: %.2f ms | Cost: %s\n" "$TIME_SYNC" "$COST_SYNC"

# Run SSP
echo "Running Sparse SSP (Asynchronous)..."
START=$(date +%s.%N)
mpirun -np 2 --bind-to none $BIN_SSP "$INSTANCE" "$K" "$M" "$SEED" > ssp_out.txt 2>&1
END=$(date +%s.%N)
TIME_SSP=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_SSP=$(grep "Best:" ssp_out.txt | tail -n 1 | awk '{print $4}')
printf "SSP Time/Epoch:  %.2f ms | Cost: %s\n" "$TIME_SSP" "$COST_SSP"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_SYNC / $TIME_SSP" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Overlap): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_SYNC $BIN_SSP sync_out.txt ssp_out.txt
