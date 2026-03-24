# Compilers
CC=gcc
MPICC=mpicc
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-Wall -Wextra -std=c11 -Iinclude
FORCE_OPT=-O3
LIBS=-lm
COVERAGE_FLAGS=-g --coverage
COVERAGE_LIBS=--coverage
EXTRA_FLAGS=

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

TEST_SRC=tests/tests.c
TEST_BIN=tests/test.out
TEST_OBJ=tests/test.o

SEQUENTIAL_TESTS_SRC=tests/sequential_tests.c
SEQUENTIAL_TESTS_OBJ=tests/sequential_tests.o
SEQUENTIAL_TESTS_BIN=tests/sequential_tests.out

OPENMP_MPI_SCALING_SRC=tests/openmp_mpi_scaling_tests.c
OPENMP_MPI_SCALING_BIN=tests/openmp_mpi_scaling_tests.out
OPENMP_MPI_SCALING_TEST_SRC=$(OPENMP_MPI_SCALING_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)

OPENMP_MPI_BIN=aco_vrp_openmp_mpi.out
MPI_TEST_BIN=tests/test_mpi.out
OPENMP_MPI_SRC=src/main.c $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)
MPI_TEST_SRC=$(TEST_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)

MPI_CASE_DIR=tests/files
MPI_CASE_FILE=test_03_a20_p4_w1
MPI_NP=2
MPI_OMP_THREADS=2
RACE_CASE_FILES=test_03_a20_p4_w1 test_04_a20_p4_w1 test_05_a20_p4_w1 test_06_a20_p4_w1

COVERAGE_FILES=src/*.gcno src/*.gcda src/seq/*.gcno src/seq/*.gcda src/openmp-mpi/*.gcno src/openmp-mpi/*.gcda src/common/*.gcno src/common/*.gcda tests/*.gcno tests/*.gcda
LEGACY_BINARIES=aco_vrp_seq aco_vrp_hybrid aco_vrp_openmp_mpi tests/test tests/test_mpi tests/test_final tests/test_final.out tests/test_final_mpi.out tests/c_scaling_tests tests/c_scaling_tests.out

all: $(BIN)

help:
	@printf "\n"
	@printf "Ant Colony Optimization for VRP\n"
	@printf "================================\n\n"
	@printf "Build Targets:\n"
	@printf "  %-18s %s\n" "all" "Build sequential binary"
	@printf "  %-18s %s\n" "openmp_mpi" "Build MPI+OpenMP binary"
	@printf "  %-18s %s\n" "debug" "Build with -DDEBUG"
	@printf "  %-18s %s\n\n" "clean" "Remove binaries, objects, and coverage artifacts"
	@printf "Test Targets:\n"
	@printf "  %-18s %s\n" "test" "Build and run final file-based tests"
	@printf "  %-18s %s\n" "test_mpi" "Run one MPI test (uses MPI_CASE_FILE, MPI_NP, MPI_OMP_THREADS)"
	@printf "  %-18s %s\n" "test_mpi_race" "Run race-oriented MPI cases (test_03..test_06)"
	@printf "  %-18s %s\n" "scaling_tests" "Run progressive PyVRP-only scaling tests up to n=100000"
	@printf "  %-18s %s\n" "sequential_tests" "Run progressive C scaling tests up to n=100000 (memory-aware)"
	@printf "  %-18s %s\n\n" "openmp_mpi_scaling_tests" "Run progressive OpenMP+MPI scaling tests up to n=100000 (uses MPI_NP, MPI_OMP_THREADS)"
	@printf "Analysis/Docs:\n"
	@printf "  %-18s %s\n" "experiments" "Run correctness + scaling experiments"
	@printf "  %-18s %s\n" "coverage" "Build with gcov and run sequential tests"
	@printf "  %-18s %s\n\n" "report" "Build report/report.pdf"

src/main.o: src/main.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(SEQ_OBJ): $(SEQ_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(ACO_SHARED_OBJ): $(ACO_SHARED_SRC) include/aco.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(SOLUTION_OBJ): $(SOLUTION_SRC) include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(MATRIX_OBJ): $(MATRIX_SRC) include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $^ $(LIBS) -o $@

$(TEST_OBJ): $(TEST_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(TEST_BIN): $(TEST_OBJ) $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $^ $(LIBS) -o $@

$(SEQUENTIAL_TESTS_OBJ): $(SEQUENTIAL_TESTS_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) -c $< -o $@

$(SEQUENTIAL_TESTS_BIN): $(SEQUENTIAL_TESTS_OBJ) $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $^ $(LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN) tests/files/golden_pyvrp.csv

openmp_mpi: $(OPENMP_MPI_BIN)

$(OPENMP_MPI_BIN): $(OPENMP_MPI_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_SRC) $(LIBS) -o $@

$(MPI_TEST_BIN): $(MPI_TEST_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(OMPFLAG) -DUSE_MPI $(MPI_TEST_SRC) $(LIBS) -o $@

$(OPENMP_MPI_SCALING_BIN): $(OPENMP_MPI_SCALING_TEST_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_SCALING_TEST_SRC) $(LIBS) -o $@

test_mpi: $(MPI_TEST_BIN)
	OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(MPI_TEST_BIN) $(MPI_CASE_DIR) $(MPI_CASE_FILE)

test_mpi_race: $(MPI_TEST_BIN)
	@for case_file in $(RACE_CASE_FILES); do \
		for np in 2 4; do \
			echo "Running $$case_file with np=$$np and OMP_NUM_THREADS=$$np"; \
			OMP_NUM_THREADS=$$np mpirun -np $$np ./$(MPI_TEST_BIN) $(MPI_CASE_DIR) $$case_file; \
		done; \
	done

experiments: $(BIN) $(OPENMP_MPI_BIN)
	MPLCONFIGDIR=/tmp/mpl python3 scripts/run_experiments.py

scaling_tests:
	python3 tests/pyvrp_tests.py

sequential_tests: $(SEQUENTIAL_TESTS_BIN)
	./$(SEQUENTIAL_TESTS_BIN)

openmp_mpi_scaling_tests: $(OPENMP_MPI_SCALING_BIN)
	OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(OPENMP_MPI_SCALING_BIN)

report: experiments
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null

coverage:
	$(MAKE) clean
	$(MAKE) test FLAGS="$(FLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_LIBS)"

clean:
	rm -f $(BIN) $(OBJ) $(TEST_BIN) $(TEST_OBJ) $(SEQUENTIAL_TESTS_BIN) $(SEQUENTIAL_TESTS_OBJ) $(OPENMP_MPI_BIN) $(MPI_TEST_BIN) $(LEGACY_BINARIES) \
		$(OPENMP_MPI_SCALING_BIN) \
		src/*.o src/seq/*.o src/openmp-mpi/*.o src/common/*.o \
		tests/*.o tests/*.out \
		$(COVERAGE_FILES) \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG

.PHONY: all help clean debug test test_mpi test_mpi_race coverage openmp_mpi experiments scaling_tests sequential_tests openmp_mpi_scaling_tests report
