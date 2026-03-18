# Compilers
CC=gcc
MPICC=mpicc
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-O3 -Wall -Wextra -std=c11 -Iinclude
LIBS=-lm
COVERAGE_FLAGS=-O0 -g --coverage
COVERAGE_LIBS=--coverage
EXTRA_FLAGS=

# Targets
BIN=aco_vrp_seq
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

TEST_BIN=tests/test_aco
TEST_OBJ=tests/test_aco.o

HYBRID_BIN=aco_vrp_hybrid
MPI_TEST_BIN=tests/test_parallel_mpi
HYBRID_SRC=src/main.c $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)
MPI_TEST_SRC=tests/test_parallel_mpi.c $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC)

COVERAGE_FILES=src/*.gcno src/*.gcda src/seq/*.gcno src/seq/*.gcda src/openmp-mpi/*.gcno src/openmp-mpi/*.gcda src/common/*.gcno src/common/*.gcda tests/*.gcno tests/*.gcda

all: $(BIN)

help:
	@echo
	@echo "Ant Colony Optimization for VRP"
	@echo
	@echo "make all         Build sequential binary"
	@echo "make hybrid      Build MPI+OpenMP binary"
	@echo "make test        Build and run sequential tests"
	@echo "make test_mpi    Build and run MPI+OpenMP test (2 ranks)"
	@echo "make experiments Run correctness + scaling experiments"
	@echo "make report      Build PDF report/report.pdf"
	@echo "make coverage    Build with gcov and run sequential tests"
	@echo "make debug       Build with -DDEBUG"
	@echo "make clean       Remove targets"
	@echo

src/main.o: src/main.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(SEQ_OBJ): $(SEQ_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(ACO_SHARED_OBJ): $(ACO_SHARED_SRC) include/aco.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(SOLUTION_OBJ): $(SOLUTION_SRC) include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(MATRIX_OBJ): $(MATRIX_SRC) include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $^ $(LIBS) -o $@

tests/test_aco.o: tests/test_aco.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(TEST_BIN): $(TEST_OBJ) $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $^ $(LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

hybrid: $(HYBRID_BIN)

$(HYBRID_BIN): $(HYBRID_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(OMPFLAG) -DUSE_MPI $(HYBRID_SRC) $(LIBS) -o $@

$(MPI_TEST_BIN): $(MPI_TEST_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(OMPFLAG) -DUSE_MPI $(MPI_TEST_SRC) $(LIBS) -o $@

test_mpi: $(MPI_TEST_BIN)
	OMP_NUM_THREADS=2 mpirun -np 2 ./$(MPI_TEST_BIN)

experiments: $(BIN) $(HYBRID_BIN)
	MPLCONFIGDIR=/tmp/mpl python3 scripts/run_experiments.py

report: experiments
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null
	cd report && pdflatex -interaction=nonstopmode -halt-on-error report.tex >/dev/null

coverage:
	$(MAKE) clean
	$(MAKE) test FLAGS="$(FLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_LIBS)"

clean:
	rm -f $(BIN) $(OBJ) $(TEST_BIN) $(TEST_OBJ) $(HYBRID_BIN) $(MPI_TEST_BIN) \
		$(COVERAGE_FILES) \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG

.PHONY: all help clean debug test coverage hybrid test_mpi experiments report
