#!/bin/bash

# Sparse MPI Sync Experiment
BIN_FULL="./full_sync.out"
BIN_SPARSE="./sparse_sync.out"
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

# Compile Full Sync (Baseline Adaptive)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/13_adaptive_tuning/aco_v2_adaptive.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_FULL

# Compile Sparse Sync (Optimized)
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/16_mpi_sparse_sync_impl/aco_v2_sparse.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN_SPARSE

echo "----------------------------------------------------"
echo "ACO V2: Sparse MPI Sync Test (N=16000, 2 Ranks x 12T)"
echo "----------------------------------------------------"

# Run Full
echo "Running Full Matrix Sync (Allreduce 1GB)..."
START=$(date +%s.%N)
mpirun -np 2 --bind-to none $BIN_FULL "$INSTANCE" "$K" "$M" "$SEED" > full_out.txt 2>&1
END=$(date +%s.%N)
TIME_FULL=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_FULL=$(grep "Final Best" full_out.txt | tail -n 1 | awk '{print $4}')
printf "Full Sync Time/Epoch:   %.2f ms | Cost: %s\n" "$TIME_FULL" "$COST_FULL"

# Run Sparse
echo "Running Sparse Delta Sync (Allgatherv 2MB)..."
START=$(date +%s.%N)
mpirun -np 2 --bind-to none $BIN_SPARSE "$INSTANCE" "$K" "$M" "$SEED" > sparse_out.txt 2>&1
END=$(date +%s.%N)
TIME_SPARSE=$(echo "($END - $START) * 1000 / $EPOCHS" | bc -l)
COST_SPARSE=$(grep "Sparse MPI Completion" sparse_out.txt | tail -n 1 | awk '{print $4}') # Placeholder cost check
printf "Sparse Sync Time/Epoch: %.2f ms\n" "$TIME_SPARSE"

# Calc speedup
SPEEDUP=$(echo "scale=2; $TIME_FULL / $TIME_SPARSE" | bc -l)
echo "----------------------------------------------------"
echo "Speedup (Network Reduction): ${SPEEDUP}x"
echo "----------------------------------------------------"

rm $BIN_FULL $BIN_SPARSE full_out.txt sparse_out.txt
