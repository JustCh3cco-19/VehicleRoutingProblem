# Compilers
CC=gcc
MPICC=mpicc
CUDACC=nvcc -arch=sm_75
OMPFLAG=-fopenmp

# Flags for optimization and libs
FLAGS=-O3 -Wall -Wextra -std=c11 -Iinclude
LIBS=-lm
COVERAGE_FLAGS=-O0 -g --coverage
COVERAGE_LIBS=--coverage

# Targets to build
BIN=aco_vrp_seq
OBJ=src/main.o src/aco.o src/solution.o src/matrix.o
TEST_BIN=tests/test_aco
TEST_OBJ=tests/test_aco.o
OBJS=$(BIN) $(OBJ)

all: $(OBJS)

# Rules. By default show help
help:
		@echo
		@echo "Ant Colony Optimization for VRP (sequential)"
		@echo
		@echo "Parallel Computing - Project scaffold"
		@echo
		@echo "make all      Build the sequential version"
		@echo "make test     Build and run tests"
		@echo "make coverage Build with gcov and run tests"
		@echo "make debug    Build with -DDEBUG"
		@echo "make clean    Remove targets"
		@echo
		@echo "make $(BIN)   Build only the sequential version"

src/main.o: src/main.c include/aco.h include/matrix.h include/solution.h
		$(CC) $(DEBUG) $(FLAGS) -c $< -o $@

src/aco.o: src/aco.c include/aco.h include/matrix.h include/solution.h
		$(CC) $(DEBUG) $(FLAGS) -c $< -o $@

src/solution.o: src/solution.c include/solution.h
		$(CC) $(DEBUG) $(FLAGS) -c $< -o $@

src/matrix.o: src/matrix.c include/matrix.h
		$(CC) $(DEBUG) $(FLAGS) -c $< -o $@

$(BIN): $(OBJ)
		$(CC) $(DEBUG) $(FLAGS) $^ $(LIBS) -o $@

tests/test_aco.o: tests/test_aco.c include/aco.h include/matrix.h include/solution.h
		$(CC) $(DEBUG) $(FLAGS) -c $< -o $@

tests/test_aco: tests/test_aco.o src/aco.o src/solution.o src/matrix.o
		$(CC) $(DEBUG) $(FLAGS) $^ $(LIBS) -o $@

test: tests/test_aco
		./tests/test_aco

coverage:
		$(MAKE) clean
		$(MAKE) test FLAGS="$(FLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_LIBS)"

# Remove the target files
clean:
		rm -f $(OBJS) $(TEST_BIN) $(TEST_OBJ) src/*.gcno src/*.gcda tests/*.gcno tests/*.gcda

# Compile in debug mode
debug:
		make DEBUG=-DDEBUG

.PHONY: all help clean debug test coverage
