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
TEST_MODE?=background
TEST_LOGS_DIR?=$(RESULTS_ROOT)/detached
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

SEQUENTIAL_TESTS_SRC=tests/sequential_tests.c
SEQUENTIAL_TESTS_OBJ=tests/sequential_tests.o
SEQUENTIAL_TESTS_BIN=tests/sequential_tests.out

C_COMPARE_CASE_SRC=tests/c_compare_case.c
C_COMPARE_CASE_BIN=tests/c_compare_case.out
C_COMPARE_CASE_MPI_SRC=tests/c_compare_case_mpi.c
C_COMPARE_CASE_MPI_BIN=tests/c_compare_case_mpi.out

OPENMP_MPI_TESTS_SRC=tests/openmp_mpi_tests.c
OPENMP_MPI_TESTS_BIN=tests/openmp_mpi_tests.out
OPENMP_MPI_TESTS_BUILD_SRC=$(OPENMP_MPI_TESTS_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)
OPENMP_MPI_TESTS_HEAVY_SRC=tests/openmp_mpi_tests_heavy.c
OPENMP_MPI_TESTS_HEAVY_BIN=tests/openmp_mpi_tests_heavy.out
OPENMP_MPI_TESTS_HEAVY_BUILD_SRC=$(OPENMP_MPI_TESTS_HEAVY_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)

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
MPI_TEST_ARGS?=
SCALING_DIR?=$(RESULTS_ROOT)/scaling
CHECKPOINT_MODE?=fresh
LIGHT_CHECKPOINT_PATH?=$(SCALING_DIR)/openmp_mpi_tests_light.checkpoint
HEAVY_CHECKPOINT_PATH?=$(SCALING_DIR)/openmp_mpi_tests_heavy.checkpoint
LIGHT_CP_FLAGS=--checkpoint $(LIGHT_CHECKPOINT_PATH) $(if $(filter resume,$(CHECKPOINT_MODE)),--resume,$(if $(filter reset,$(CHECKPOINT_MODE)),--reset-checkpoint,))
HEAVY_CP_FLAGS=--checkpoint $(HEAVY_CHECKPOINT_PATH) $(if $(filter resume,$(CHECKPOINT_MODE)),--resume,$(if $(filter reset,$(CHECKPOINT_MODE)),--reset-checkpoint,))

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

define RUN_TEST_CMD
@if [ "$(TEST_MODE)" = "background" ]; then \
	log_dir="$(TEST_LOGS_DIR)"; \
	mkdir -p "$$log_dir"; \
	ts=$$(date +%Y%m%d_%H%M%S); \
	job="$(1)"; \
	log_file="$$log_dir/$${job}_$${ts}.log"; \
	pid_file="$$log_dir/$${job}_$${ts}.pid"; \
	cmd_file="$$log_dir/$${job}_$${ts}.cmd"; \
	run_cmd="cd '$(CURDIR)' && $(2)"; \
	printf '%s\n' "$$run_cmd" >"$$cmd_file"; \
	nohup bash -lc "$$run_cmd" >"$$log_file" 2>&1 </dev/null & \
	pid=$$!; \
	printf '%s\n' "$$pid" >"$$pid_file"; \
	echo "[DETACHED] job=$$job"; \
	echo "[DETACHED] pid=$$pid"; \
	echo "[DETACHED] log=$$log_file"; \
	echo "[DETACHED] pid_file=$$pid_file"; \
	echo "[DETACHED] cmd_file=$$cmd_file"; \
	echo "[DETACHED] tail -f $$log_file"; \
else \
	bash -lc "$(2)"; \
fi
endef
