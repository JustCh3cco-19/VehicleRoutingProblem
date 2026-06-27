# Compilers
CC=gcc
MPICC=mpicc
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-Wall -Wextra -std=c11 -Iinclude
FORCE_OPT=-O3
PERF_FLAGS?=
LIBS=-lm
NVCC?=$(if $(wildcard /usr/local/cuda-12.8/bin/nvcc),/usr/local/cuda-12.8/bin/nvcc,nvcc)
CUDA_ARCH?=$(shell ./tools/bash/detect_gpu.sh)
CUDA_VARIANT?=core
NVCC_FLAGS=-Iinclude -O3 -std=c++17
CUDA_ARCH_FLAG=$(if $(strip $(CUDA_ARCH)),-arch=$(CUDA_ARCH),)
COVERAGE_FLAGS=-g --coverage
COVERAGE_LIBS=--coverage
EXTRA_FLAGS=
RESULTS_ROOT?=results
PYTHON_BIN?=$(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Targets
BIN=seq.out
SEQ_DIR=src/seq
PAR_DIR=src/openmp-mpi
COMMON_DIR=src/common

SEQ_SRC=$(SEQ_DIR)/sequential.c $(SEQ_DIR)/candidates.c $(SEQ_DIR)/candidates_avx.c $(SEQ_DIR)/tour.c $(SEQ_DIR)/tour_select.c $(SEQ_DIR)/tour_select_large.c $(SEQ_DIR)/workspace.c $(SEQ_DIR)/math_utils.c $(SEQ_DIR)/mem_utils.c $(SEQ_DIR)/seq_ctx.c $(SEQ_DIR)/seq_epoch.c $(SEQ_DIR)/seq_pheromone.c
PAR_SRC=$(PAR_DIR)/parallel.c $(PAR_DIR)/par_solver_ctx.c $(PAR_DIR)/par_solver_epoch.c $(PAR_DIR)/par_solver_pheromone.c
PAR_CANDIDATES_SRC=$(PAR_DIR)/candidates.c $(PAR_DIR)/candidates_init.c
PAR_SYNC_SRC=$(PAR_DIR)/sync.c $(PAR_DIR)/par_workspace.c $(PAR_DIR)/par_tour.c $(PAR_DIR)/par_utils_math.c $(PAR_DIR)/par_utils_mem.c
ACO_SHARED_SRC=$(COMMON_DIR)/shared.c $(COMMON_DIR)/shared_solver.c $(COMMON_DIR)/score_cache.c $(COMMON_DIR)/score_cache_lifecycle.c
ACO_CONFIG_SRC=$(COMMON_DIR)/config.c
SOLUTION_SRC=$(COMMON_DIR)/solution.c
MATRIX_SRC=$(COMMON_DIR)/matrix.c
INSTANCE_PARSER_SRC=$(COMMON_DIR)/instance_parser.c
CLI_COMMON_SRC=$(COMMON_DIR)/cli_common.c
SEQ_OBJ=$(SEQ_DIR)/sequential.o $(SEQ_DIR)/candidates.o $(SEQ_DIR)/candidates_avx.o $(SEQ_DIR)/tour.o $(SEQ_DIR)/tour_select.o $(SEQ_DIR)/tour_select_large.o $(SEQ_DIR)/workspace.o $(SEQ_DIR)/math_utils.o $(SEQ_DIR)/mem_utils.o $(SEQ_DIR)/seq_ctx.o $(SEQ_DIR)/seq_epoch.o $(SEQ_DIR)/seq_pheromone.o
ACO_SHARED_OBJ=$(COMMON_DIR)/shared.o $(COMMON_DIR)/shared_solver.o $(COMMON_DIR)/score_cache.o $(COMMON_DIR)/score_cache_lifecycle.o
ACO_CONFIG_OBJ=$(COMMON_DIR)/config.o
SOLUTION_OBJ=$(COMMON_DIR)/solution.o
MATRIX_OBJ=$(COMMON_DIR)/matrix.o
INSTANCE_PARSER_OBJ=$(COMMON_DIR)/instance_parser.o
CLI_COMMON_OBJ=$(COMMON_DIR)/cli_common.o

OBJ=src/main.o $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(ACO_CONFIG_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ) $(INSTANCE_PARSER_OBJ) $(CLI_COMMON_OBJ)

OPENMP_MPI_BIN=openmp_mpi.out
OPENMP_MPI_SRC=src/main.c $(PAR_SRC) $(PAR_CANDIDATES_SRC) $(PAR_SYNC_SRC) $(ACO_SHARED_SRC) $(ACO_CONFIG_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(INSTANCE_PARSER_SRC) $(CLI_COMMON_SRC)
CUDA_BIN=cuda.out
CUDA_MAIN_SRC=src/main_cuda.cu
CUDA_ACO_SRC=src/cuda/cuda.cu
CUDA_KERNELS_SRC=src/cuda/kernels.cu
CUDA_MAIN_OBJ=src/main_cuda.o
CUDA_ACO_OBJ=src/cuda/cuda.o
CUDA_KERNELS_OBJ=src/cuda/kernels.o
CUDA_COMMON_OBJ=$(SOLUTION_OBJ) $(MATRIX_OBJ) $(CLI_COMMON_OBJ) $(ACO_CONFIG_OBJ) $(ACO_SHARED_OBJ)
CUDA_OBJ=$(CUDA_MAIN_OBJ) $(CUDA_ACO_OBJ) $(CUDA_KERNELS_OBJ) $(CUDA_COMMON_OBJ) $(INSTANCE_PARSER_OBJ)
MPI_NP=2
MPI_OMP_THREADS=2

SOLVE_OUT_DIR?=$(RESULTS_ROOT)/solve_manifest
SOLVE_CSV_DIR?=$(SOLVE_OUT_DIR)/csv
SOLVE_SOLUTIONS_DIR?=$(SOLVE_OUT_DIR)/solutions
SOLVE_MANIFEST?=instances/generated_benchmark/manifest.csv
SOLVE_MANIFEST_MPI?=instances/generated_benchmark/manifest_openmp_mpi.csv
SOLVE_MANIFEST_CUDA?=instances/generated_benchmark/manifest_cuda.csv
SOLVE_PYVRP_RUNTIME_S?=10
SOLVE_SEQ_RUNTIME?=
SOLVE_SEQ_RUNTIME_S?=0
SOLVE_SEQ_STAGNATION_EPOCHS?=0
SOLVE_SEQ_MIN_REL_IMPROVEMENT?=0.1
SOLVE_SEQ_M?=0
SOLVE_SEQ_REPEATS?=1
SOLVE_MPI_RUNTIME_S?=0
SOLVE_MPI_STAGNATION_EPOCHS?=0
SOLVE_MPI_MIN_REL_IMPROVEMENT?=0.1
SOLVE_MPI_REPEATS?=1
SOLVE_PYVRP_SEED?=1234
SOLVE_LIMIT?=0
SOLVE_CLIENTS?=
SOLVE_MPI_RANKS?=2
SOLVE_MPI_OMP_THREADS?=2
SOLVE_MPI_LAUNCHER?=auto
SOLVE_CUDA_IMPROVEMENT?=0.1
SOLVE_CUDA_VARIANT?=$(CUDA_VARIANT)
SOLVE_CUDA_REPEATS?=1
SOLVE_CUDA_RUNTIME_S?=
SOLVE_CUDA_STAGNATION_EPOCHS?=
SOLVE_CUDA_MIN_REL_IMPROVEMENT?=
SOLVE_MEMORY_GROWTH_REPEATS?=3
SOLVE_SEQ_RUNTIME_EFFECTIVE=$(if $(strip $(SOLVE_SEQ_RUNTIME)),$(SOLVE_SEQ_RUNTIME),$(SOLVE_SEQ_RUNTIME_S))

# Scaling experiments defaults
EXP_REPEATS?=5
EXP_STAGNATION_EPOCHS?=500
EXP_MIN_REL_IMPROVEMENT?=0.1
EXP_MAX_CLUSTER_NODES?=4
EXP_WEAK_BASE_N_PER_WORKER?=2000
EXP_MPI_LAUNCHER?=mpirun

EXP_STRONG_OPENMP_N?=2000
EXP_STRONG_OPENMP_THREADS?=auto
EXP_STRONG_CLIENTS_SERIES?=2000,4000,8000,16000,32000
EXP_WEAK_OPENMP_PAIRS?=auto

EXP_STRONG_MPI_N?=16000
EXP_STRONG_MPI_RANKS?=auto
EXP_STRONG_MPI_OMP_THREADS?=8
EXP_STRONG_HYBRID_N?=16000
EXP_STRONG_HYBRID_PAIRS?=auto

EXP_WEAK_MPI_PAIRS?=auto
EXP_WEAK_MPI_OMP_THREADS?=8
EXP_WEAK_HYBRID_TRIPLETS?=auto

# CUDA experiments defaults
EXP_CUDA_CLIENTS_FUNCTIONAL?=500,1000,2000,4000,8000
EXP_CUDA_M_LIST?=64 128 256
EXP_CUDA_SCALING_M_FIXED?=256
EXP_CUDA_SCALING_CLIENTS?=500,1000,2000,4000,8000
EXP_CUDA_REPEATS?=1
EXP_CUDA_RUNTIME_S?=300
EXP_CUDA_STAGNATION_EPOCHS?=500
EXP_CUDA_MIN_REL_IMPROVEMENT?=10
EXP_CUDA_VARIANT?=$(CUDA_VARIANT)

GEN_INST_DIR?=instances/generated_benchmark
GEN_CLIENTS?=4000,8000,16000,32000,64000,128000,200000
GEN_SEED_BASE?=19000
GEN_GRID?=100
GEN_SOLVER_SEED?=1234
GEN_TARGET_CUSTOMERS_PER_VEHICLE?=1024
GEN_MIN_VEHICLES?=8
GEN_MAX_VEHICLES?=512
GEN_CAPACITY_SLACK_PERCENT?=20
GEN_CUDA_M?=256

COVERAGE_FILES=src/*.gcno src/*.gcda src/seq/*.gcno src/seq/*.gcda src/openmp-mpi/*.gcno src/openmp-mpi/*.gcda src/common/*.gcno src/common/*.gcda tests/*.gcno tests/*.gcda
LEGACY_BINARIES=aco_vrp_seq aco_vrp_hybrid aco_vrp_openmp_mpi tests/test tests/test_mpi tests/test_final tests/test_final.out tests/test_final_mpi.out tests/c_scaling_tests tests/c_scaling_tests.out
