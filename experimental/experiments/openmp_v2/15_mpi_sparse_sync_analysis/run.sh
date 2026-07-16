#!/bin/bash

# Sparse Sync Analysis
BIN="./sparsity_diag.out"
INSTANCE="instances/test_aligned/n16000_k16_s19005.vrp"
K=16
M=1536
SEED=1234
EPOCHS=20

export ACO_SOLVER_FIXED_EPOCHS=$EPOCHS
export OMP_NUM_THREADS=24
export OMP_SCHEDULE="guided,4"

# Compile Diagnostic Version
mpicc -Wall -Wextra -std=c11 -Iinclude -O3 -fopenmp -DUSE_MPI -DACO_VRP_V2 \
    src/main.c experimental/experiments/openmp_v2/15_mpi_sparse_sync_analysis/aco_v2_sparsity_diag.c \
    src/common/aco_shared.c src/common/solution.c src/common/matrix.c src/common/instance_parser.c \
    -lm -o $BIN

echo "----------------------------------------------------"
echo "ACO V2: Sparse Sync Analysis (N=16000, M=1536)"
echo "----------------------------------------------------"

# Run Diagnostic
$BIN "$INSTANCE" "$K" "$M" "$SEED" 2>&1 | grep -A 5 "Sparsity Analysis"

rm $BIN
echo "----------------------------------------------------"
