# Compilers
CC=gcc
MPICC=mpicc
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-Wall -Wextra -std=c11 -Iinclude
FORCE_OPT=-O3
PERF_FLAGS?=
LIBS=-lm
NVCC?=nvcc
CUDA_ARCH?=
NVCC_FLAGS=-Iinclude -O3 -std=c++17
CUDA_ARCH_FLAG=$(if $(strip $(CUDA_ARCH)),-arch=$(CUDA_ARCH),)
COVERAGE_FLAGS=-g --coverage
COVERAGE_LIBS=--coverage
EXTRA_FLAGS=
TEST_MODE?=background
TEST_LOGS_DIR?=results/detached

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

SEQ_OBJ=$(SEQ_DIR)/aco_sequential.o
ACO_SHARED_OBJ=$(COMMON_DIR)/aco_shared.o
SOLUTION_OBJ=$(COMMON_DIR)/solution.o
MATRIX_OBJ=$(COMMON_DIR)/matrix.o

OBJ=src/main.o $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)

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
OPENMP_MPI_SRC=src/main.c $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)
CUDA_BIN=aco_vrp_cuda.out
CUDA_MAIN_SRC=src/cuda/main_vrp.cu
CUDA_ACO_SRC=src/cuda/aco_cuda.cu
CUDA_KERNELS_SRC=src/cuda/aco_cuda_kernels.cu
CUDA_MAIN_OBJ=src/cuda/main_vrp.o
CUDA_ACO_OBJ=src/cuda/aco_cuda.o
CUDA_KERNELS_OBJ=src/cuda/aco_cuda_kernels.o
CUDA_COMMON_OBJ=$(SOLUTION_OBJ) $(MATRIX_OBJ)
CUDA_PARSER_SRC=$(firstword $(wildcard src/common/instance_parser.c src/cuda/instance_parser.c src/instance_parser.c))
CUDA_PARSER_OBJ=$(patsubst %.c,%.o,$(CUDA_PARSER_SRC))
CUDA_OBJ=$(CUDA_MAIN_OBJ) $(CUDA_ACO_OBJ) $(CUDA_KERNELS_OBJ) $(CUDA_COMMON_OBJ) $(CUDA_PARSER_OBJ)
MPI_NP=2
MPI_OMP_THREADS=2
MPI_TEST_ARGS?=
CHECKPOINT_MODE?=fresh
LIGHT_CHECKPOINT_PATH?=results/openmp_mpi_tests_light.checkpoint
HEAVY_CHECKPOINT_PATH?=results/openmp_mpi_tests_heavy.checkpoint
LIGHT_CP_FLAGS=--checkpoint $(LIGHT_CHECKPOINT_PATH) $(if $(filter resume,$(CHECKPOINT_MODE)),--resume,$(if $(filter reset,$(CHECKPOINT_MODE)),--reset-checkpoint,))
HEAVY_CP_FLAGS=--checkpoint $(HEAVY_CHECKPOINT_PATH) $(if $(filter resume,$(CHECKPOINT_MODE)),--resume,$(if $(filter reset,$(CHECKPOINT_MODE)),--reset-checkpoint,))

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

all: $(BIN)

help:
	@printf "\n"
	@printf "Ant Colony Optimization for VRP\n"
	@printf "================================\n\n"
	@printf "Build Targets:\n"
	@printf "  %-18s %s\n" "all" "Build sequential binary"
	@printf "  %-18s %s\n" "openmp_mpi" "Build MPI+OpenMP binary"
	@printf "  %-18s %s\n" "cuda" "Build CUDA binary"
	@printf "  %-18s %s\n" "debug" "Build with -DDEBUG"
	@printf "  %-18s %s\n\n" "clean" "Remove binaries, objects, and coverage artifacts"
	@printf "Test Targets:\n"
	@printf "  %-18s %s\n" "sequential_tests" "Run progressive C scaling tests up to n=100000 (memory-aware)"
	@printf "  %-18s %s\n" "openmp_mpi_tests" "Run OpenMP+MPI light tests up to 16k customers"
	@printf "  %-18s %s\n" "openmp_mpi_tests_heavy" "Run OpenMP+MPI heavy tests from 24k to 100k customers"
	@printf "  %-18s %s\n" "openmp_mpi_tests_resume" "Resume OpenMP+MPI light tests from checkpoint"
	@printf "  %-18s %s\n" "openmp_mpi_tests_reset" "Reset checkpoint and run OpenMP+MPI light tests"
	@printf "  %-18s %s\n" "openmp_mpi_tests_heavy_resume" "Resume OpenMP+MPI heavy tests from checkpoint"
	@printf "  %-18s %s\n" "openmp_mpi_tests_heavy_reset" "Reset checkpoint and run OpenMP+MPI heavy tests"
	@printf "  %-18s %s\n" "c_compare_case" "Build single-scenario C runner for C-vs-PyVRP comparisons"
	@printf "  %-18s %s\n" "c_compare_case_mpi" "Build single-scenario MPI+OpenMP runner for C-vs-PyVRP comparisons"
	@printf "\n"
	@printf "Test Run Options:\n"
	@printf "  %-35s %s\n" "TEST_MODE=background (default)" "Run detached with nohup and return immediately"
	@printf "  %-35s %s\n" "TEST_MODE=foreground" "Run test in current terminal and stream output live"
	@printf "  %-35s %s\n" "TEST_LOGS_DIR=PATH" "Directory for detached .log/.pid/.cmd files"
	@printf "  %-35s %s\n\n" "Example" "make sequential_tests TEST_MODE=foreground"

src/main.o: src/main.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SEQ_OBJ): $(SEQ_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(ACO_SHARED_OBJ): $(ACO_SHARED_SRC) include/aco.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SOLUTION_OBJ): $(SOLUTION_SRC) include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(MATRIX_OBJ): $(MATRIX_SRC) include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $^ $(LIBS) -o $@

$(SEQUENTIAL_TESTS_OBJ): $(SEQUENTIAL_TESTS_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SEQUENTIAL_TESTS_BIN): $(SEQUENTIAL_TESTS_OBJ) $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $^ $(LIBS) -o $@

$(C_COMPARE_CASE_BIN): $(C_COMPARE_CASE_SRC) $(SEQ_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(C_COMPARE_CASE_SRC) $(SEQ_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(LIBS) -o $@

$(C_COMPARE_CASE_MPI_BIN): $(C_COMPARE_CASE_MPI_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(C_COMPARE_CASE_MPI_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(LIBS) -o $@

openmp_mpi: $(OPENMP_MPI_BIN)

$(OPENMP_MPI_BIN): $(OPENMP_MPI_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_SRC) $(LIBS) -o $@

cuda: $(CUDA_BIN)

$(CUDA_MAIN_OBJ): $(CUDA_MAIN_SRC) include/acocopy.h include/instance_parser.h include/matrixcopy.h include/solutioncopy.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_ACO_OBJ): $(CUDA_ACO_SRC) include/acocopy.h include/aco_cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_KERNELS_OBJ): $(CUDA_KERNELS_SRC) include/aco_cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_PARSER_OBJ): %.o: %.c
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(CUDA_BIN): $(CUDA_OBJ)
	$(NVCC) $(CUDA_ARCH_FLAG) $^ -o $@
	@if [ -z "$(CUDA_PARSER_SRC)" ]; then \
		echo "[WARN] instance parser source not found (looked for src/common/instance_parser.c, src/cuda/instance_parser.c, src/instance_parser.c)"; \
		echo "[WARN] Link may fail with undefined reference to vrp_load_tsplib_euc2d_matrix"; \
	fi

$(OPENMP_MPI_TESTS_BIN): $(OPENMP_MPI_TESTS_BUILD_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_TESTS_BUILD_SRC) $(LIBS) -o $@

$(OPENMP_MPI_TESTS_HEAVY_BIN): $(OPENMP_MPI_TESTS_HEAVY_BUILD_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_TESTS_HEAVY_BUILD_SRC) $(LIBS) -o $@

sequential_tests: $(SEQUENTIAL_TESTS_BIN)
	$(call RUN_TEST_CMD,sequential_tests,./$(SEQUENTIAL_TESTS_BIN))

openmp_mpi_tests: $(OPENMP_MPI_TESTS_BIN)
	$(call RUN_TEST_CMD,openmp_mpi_tests,OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(OPENMP_MPI_TESTS_BIN) $(LIGHT_CP_FLAGS) $(MPI_TEST_ARGS))

openmp_mpi_tests_heavy: $(OPENMP_MPI_TESTS_HEAVY_BIN)
	$(call RUN_TEST_CMD,openmp_mpi_tests_heavy,OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(OPENMP_MPI_TESTS_HEAVY_BIN) $(HEAVY_CP_FLAGS) $(MPI_TEST_ARGS))

openmp_mpi_tests_resume:
	$(MAKE) openmp_mpi_tests CHECKPOINT_MODE=resume TEST_MODE=$(TEST_MODE) MPI_NP=$(MPI_NP) MPI_OMP_THREADS=$(MPI_OMP_THREADS) MPI_TEST_ARGS="$(MPI_TEST_ARGS)" LIGHT_CHECKPOINT_PATH="$(LIGHT_CHECKPOINT_PATH)"

openmp_mpi_tests_reset:
	$(MAKE) openmp_mpi_tests CHECKPOINT_MODE=reset TEST_MODE=$(TEST_MODE) MPI_NP=$(MPI_NP) MPI_OMP_THREADS=$(MPI_OMP_THREADS) MPI_TEST_ARGS="$(MPI_TEST_ARGS)" LIGHT_CHECKPOINT_PATH="$(LIGHT_CHECKPOINT_PATH)"

openmp_mpi_tests_heavy_resume:
	$(MAKE) openmp_mpi_tests_heavy CHECKPOINT_MODE=resume TEST_MODE=$(TEST_MODE) MPI_NP=$(MPI_NP) MPI_OMP_THREADS=$(MPI_OMP_THREADS) MPI_TEST_ARGS="$(MPI_TEST_ARGS)" HEAVY_CHECKPOINT_PATH="$(HEAVY_CHECKPOINT_PATH)"

openmp_mpi_tests_heavy_reset:
	$(MAKE) openmp_mpi_tests_heavy CHECKPOINT_MODE=reset TEST_MODE=$(TEST_MODE) MPI_NP=$(MPI_NP) MPI_OMP_THREADS=$(MPI_OMP_THREADS) MPI_TEST_ARGS="$(MPI_TEST_ARGS)" HEAVY_CHECKPOINT_PATH="$(HEAVY_CHECKPOINT_PATH)"

c_compare_case: $(C_COMPARE_CASE_BIN)

c_compare_case_mpi: $(C_COMPARE_CASE_MPI_BIN)

clean:
	rm -f $(BIN) $(OBJ) $(SEQUENTIAL_TESTS_BIN) $(SEQUENTIAL_TESTS_OBJ) $(OPENMP_MPI_BIN) $(CUDA_BIN) $(LEGACY_BINARIES) \
		$(OPENMP_MPI_TESTS_BIN) $(OPENMP_MPI_TESTS_HEAVY_BIN) \
		src/*.o src/seq/*.o src/openmp-mpi/*.o src/common/*.o src/cuda/*.o \
		tests/*.o tests/*.out \
		$(COVERAGE_FILES) \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG

.PHONY: all help clean debug openmp_mpi cuda sequential_tests openmp_mpi_tests openmp_mpi_tests_heavy openmp_mpi_tests_resume openmp_mpi_tests_reset openmp_mpi_tests_heavy_resume openmp_mpi_tests_heavy_reset c_compare_case c_compare_case_mpi
