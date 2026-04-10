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
CUDA_ARCH?=sm_120
CUDA_VARIANT?=v2
NVCC_FLAGS=-Iinclude -O3 -std=c++17
CUDA_ARCH_FLAG=$(if $(strip $(CUDA_ARCH)),-arch=$(CUDA_ARCH),)
COVERAGE_FLAGS=-g --coverage
COVERAGE_LIBS=--coverage
EXTRA_FLAGS=
RESULTS_ROOT?=results
PYTHON_BIN?=$(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

# Targets
BIN=aco_vrp_seq.out
SEQ_DIR=src/seq
PAR_DIR=src/openmp-mpi
COMMON_DIR=src/common

SEQ_SRC=$(SEQ_DIR)/aco_sequential.c
PAR_SRC=$(PAR_DIR)/aco_parallel.c
ACO_SHARED_SRC=$(COMMON_DIR)/aco_shared.c
SOLUTION_SRC=$(COMMON_DIR)/solution.c
MATRIX_SRC=$(COMMON_DIR)/matrix.c
INSTANCE_PARSER_SRC=$(COMMON_DIR)/instance_parser.c
SEQ_OBJ=$(SEQ_DIR)/aco_sequential.o
ACO_SHARED_OBJ=$(COMMON_DIR)/aco_shared.o
SOLUTION_OBJ=$(COMMON_DIR)/solution.o
MATRIX_OBJ=$(COMMON_DIR)/matrix.o
INSTANCE_PARSER_OBJ=$(COMMON_DIR)/instance_parser.o

OBJ=src/main.o $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ) $(INSTANCE_PARSER_OBJ)

OPENMP_MPI_BIN=aco_vrp_openmp_mpi.out
OPENMP_MPI_SRC=src/main.c $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(INSTANCE_PARSER_SRC)
CUDA_BIN=aco_vrp_cuda.out
CUDA_MAIN_SRC=src/cuda/main_vrp.cu
CUDA_ACO_SRC=src/cuda/aco_cuda_$(CUDA_VARIANT).cu
CUDA_KERNELS_SRC=src/cuda/aco_cuda_$(CUDA_VARIANT)_kernels.cu
CUDA_MAIN_OBJ=src/cuda/main_vrp.o
CUDA_ACO_OBJ=src/cuda/aco_cuda_$(CUDA_VARIANT).o
CUDA_KERNELS_OBJ=src/cuda/aco_cuda_$(CUDA_VARIANT)_kernels.o
CUDA_COMMON_OBJ=$(SOLUTION_OBJ) $(MATRIX_OBJ)
CUDA_OBJ=$(CUDA_MAIN_OBJ) $(CUDA_ACO_OBJ) $(CUDA_KERNELS_OBJ) $(CUDA_COMMON_OBJ) $(INSTANCE_PARSER_OBJ)
MPI_NP=2
MPI_OMP_THREADS=2

SOLVE_OUT_DIR?=$(RESULTS_ROOT)/solve_manifest
SOLVE_CSV_DIR?=$(SOLVE_OUT_DIR)/csv
SOLVE_SOLUTIONS_DIR?=$(SOLVE_OUT_DIR)/solutions
SOLVE_MANIFEST?=instances/test_aligned/manifest.csv
SOLVE_MANIFEST_MPI?=instances/test_aligned/manifest_openmp_mpi.csv
SOLVE_PYVRP_RUNTIME_S?=10
SOLVE_SEQ_RUNTIME?=
SOLVE_SEQ_RUNTIME_S?=0
SOLVE_SEQ_STAGNATION_EPOCHS?=0
SOLVE_SEQ_MIN_REL_IMPROVEMENT?=0.001
SOLVE_SEQ_M?=0
SOLVE_SEQ_REPEATS?=1
SOLVE_MPI_RUNTIME_S?=0
SOLVE_MPI_STAGNATION_EPOCHS?=0
SOLVE_MPI_MIN_REL_IMPROVEMENT?=0.001
SOLVE_MPI_REPEATS?=1
SOLVE_PYVRP_SEED?=1234
SOLVE_LIMIT?=0
SOLVE_CLIENTS?=
SOLVE_MPI_RANKS?=2
SOLVE_MPI_OMP_THREADS?=2
SOLVE_CUDA_IMPROVEMENT?=0.001
SOLVE_CUDA_VARIANT?=$(CUDA_VARIANT)
SOLVE_CUDA_REPEATS?=1
SOLVE_MEMORY_GROWTH_REPEATS?=3
SOLVE_SEQ_RUNTIME_EFFECTIVE=$(if $(strip $(SOLVE_SEQ_RUNTIME)),$(SOLVE_SEQ_RUNTIME),$(SOLVE_SEQ_RUNTIME_S))

GEN_INST_DIR?=instances/test_aligned
GEN_CLIENTS?=4000,8000,16000,32000,64000,128000,200000
GEN_SEED_BASE?=19000
GEN_GRID?=100
GEN_SOLVER_SEED?=1234
GEN_TARGET_CUSTOMERS_PER_VEHICLE?=1024
GEN_MIN_VEHICLES?=8
GEN_MAX_VEHICLES?=512

COVERAGE_FILES=src/*.gcno src/*.gcda src/seq/*.gcno src/seq/*.gcda src/openmp-mpi/*.gcno src/openmp-mpi/*.gcda src/common/*.gcno src/common/*.gcda tests/*.gcno tests/*.gcda
LEGACY_BINARIES=aco_vrp_seq aco_vrp_hybrid aco_vrp_openmp_mpi tests/test tests/test_mpi tests/test_final tests/test_final.out tests/test_final_mpi.out tests/c_scaling_tests tests/c_scaling_tests.out
