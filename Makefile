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
OBJ=src/main.o src/aco.o src/solution.o src/matrix.o

TEST_BIN=tests/test_aco
TEST_OBJ=tests/test_aco.o

HYBRID_BIN=aco_vrp_hybrid
MPI_TEST_BIN=tests/test_parallel_mpi

all: $(BIN)

help:
	@echo
	@echo "Ant Colony Optimization for VRP"
	@echo
	@echo "make all        Build sequential binary"
	@echo "make hybrid     Build MPI+OpenMP binary"
	@echo "make test       Build and run sequential tests"
	@echo "make test_mpi   Build and run MPI+OpenMP test (2 ranks)"
	@echo "make experiments Run correctness + scaling experiments"
	@echo "make report     Build PDF report/report.pdf"
	@echo "make coverage   Build with gcov and run sequential tests"
	@echo "make debug      Build with -DDEBUG"
	@echo "make clean      Remove targets"
	@echo

src/main.o: src/main.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

src/aco.o: src/aco.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

src/solution.o: src/solution.c include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

src/matrix.o: src/matrix.c include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $^ $(LIBS) -o $@

tests/test_aco.o: tests/test_aco.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) -c $< -o $@

$(TEST_BIN): $(TEST_OBJ) src/aco.o src/solution.o src/matrix.o
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $^ $(LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

hybrid: $(HYBRID_BIN)

$(HYBRID_BIN): src/main.c src/aco.c src/solution.c src/matrix.c include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(OMPFLAG) -DUSE_MPI $^ $(LIBS) -o $@

$(MPI_TEST_BIN): tests/test_parallel_mpi.c src/aco.c src/solution.c src/matrix.c include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(OMPFLAG) -DUSE_MPI $^ $(LIBS) -o $@

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
		src/*.gcno src/*.gcda tests/*.gcno tests/*.gcda \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG

.PHONY: all help clean debug test coverage hybrid test_mpi experiments report
