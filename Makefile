CC=gcc
MPICC=mpicc
CUDACC?=/usr/local/cuda/bin/nvcc

# Set GPU arch explicitly to avoid PTX JIT/toolchain mismatch at runtime.
# For GTX 1650 use CUDA_ARCH=75 (default here). Override as needed:
#   make cuda CUDA_ARCH=86
CUDA_ARCH?=75
CUDA_GENCODE=-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
CUDA_FAST_MATH?=1
CUDA_MATH_FLAGS=
ifeq ($(CUDA_FAST_MATH),1)
CUDA_MATH_FLAGS=-use_fast_math
endif

SRC_COMMON=src/common
SRC_SEQ=src/seq
SRC_OMP_MPI=src/omp_mpi
SRC_CUDA=src/cuda

CFLAGS=-O3 -Wall -Wextra -std=c11 -Iinclude -I$(SRC_COMMON) -I$(SRC_OMP_MPI) -I$(SRC_CUDA)
MPICFLAGS=$(CFLAGS)
OMPFLAGS=-fopenmp
CUDAFLAGS=-O3 -Iinclude -I$(SRC_COMMON) -I$(SRC_OMP_MPI) -I$(SRC_CUDA) $(CUDA_GENCODE) $(CUDA_MATH_FLAGS)
LIBS=-lm
CUDALIBS=-lm -lcudart
COVERAGE_FLAGS=-O0 -g --coverage
COVERAGE_LIBS=--coverage

SEQ_BIN=aco_vrp_seq
OMP_BIN=aco_vrp_omp
MPI_BIN=aco_vrp_mpi
CUDA_BIN=aco_vrp_cuda
TEST_BIN=tests/test_aco

CORE_OBJ=$(SRC_COMMON)/solution.o $(SRC_COMMON)/matrix.o $(SRC_COMMON)/aco_common.o
SEQ_OBJ=$(SRC_SEQ)/aco_seq.o
OMP_OBJ=$(SRC_OMP_MPI)/aco_openmp.o
MPI_OBJ=$(SRC_OMP_MPI)/aco_mpi.o $(SRC_OMP_MPI)/aco_mpi_utils.o
CUDA_OBJ=$(SRC_CUDA)/aco_cuda.o $(SRC_CUDA)/aco_cuda_kernels.o $(SRC_CUDA)/aco_cuda_host_utils.o
CUDA_CORE_OBJ=$(SRC_CUDA)/main_cuda.o $(SRC_COMMON)/solution_cuda.o $(SRC_COMMON)/matrix_cuda.o
MAIN_SEQ_OBJ=$(SRC_SEQ)/main_seq.o
MAIN_OMP_OBJ=$(SRC_OMP_MPI)/main_omp.o
MAIN_MPI_OBJ=$(SRC_OMP_MPI)/main_mpi.o
TEST_OBJ=tests/test_aco.o

all: $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN)

help:
	@echo
	@echo "Ant Colony Optimization for VRP"
	@echo
	@echo "make all      Build seq + OpenMP + MPI+OpenMP"
	@echo "make seq      Build sequential binary"
	@echo "make omp      Build OpenMP binary"
	@echo "make mpi      Build MPI+OpenMP binary"
	@echo "make cuda     Build CUDA binary (if nvcc is available)"
	@echo "make test     Build and run tests (seq + OpenMP)"
	@echo "make test-mpi Build and run MPI smoke test"
	@echo "make benchmark Run scaling benchmark script"
	@echo "make coverage Build with gcov and run tests"
	@echo "make debug    Build with -DDEBUG"
	@echo "make clean    Remove targets"
	@echo

$(SRC_SEQ)/main_seq.o: $(SRC_SEQ)/main_seq.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_OMP_MPI)/main_omp.o: $(SRC_OMP_MPI)/main_omp.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_OMP_MPI)/main_mpi.o: $(SRC_OMP_MPI)/main_mpi.c include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(DEBUG) $(MPICFLAGS) -c $< -o $@

$(SRC_CUDA)/main_cuda.o: $(SRC_CUDA)/main_cuda.c include/aco.h include/matrix.h include/solution.h
	$(CUDACC) $(CUDAFLAGS) -x cu -c $< -o $@

$(SRC_COMMON)/aco_common.o: $(SRC_COMMON)/aco_common.c $(SRC_COMMON)/aco_common.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/aco_seq.o: $(SRC_SEQ)/aco_seq.c $(SRC_COMMON)/aco_common.h include/aco.h include/matrix.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_OMP_MPI)/aco_openmp.o: $(SRC_OMP_MPI)/aco_openmp.c $(SRC_COMMON)/aco_common.h include/aco.h include/matrix.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) $(OMPFLAGS) -c $< -o $@

$(SRC_OMP_MPI)/aco_mpi_utils.o: $(SRC_OMP_MPI)/aco_mpi_utils.c $(SRC_OMP_MPI)/aco_mpi_utils.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_OMP_MPI)/aco_mpi.o: $(SRC_OMP_MPI)/aco_mpi.c $(SRC_COMMON)/aco_common.h $(SRC_OMP_MPI)/aco_mpi_utils.h include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(DEBUG) $(MPICFLAGS) $(OMPFLAGS) -c $< -o $@

$(SRC_COMMON)/solution.o: $(SRC_COMMON)/solution.c include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_COMMON)/matrix.o: $(SRC_COMMON)/matrix.c include/matrix.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(SRC_COMMON)/solution_cuda.o: $(SRC_COMMON)/solution.c include/solution.h
	$(CUDACC) $(CUDAFLAGS) -x cu -c $< -o $@

$(SRC_COMMON)/matrix_cuda.o: $(SRC_COMMON)/matrix.c include/matrix.h
	$(CUDACC) $(CUDAFLAGS) -x cu -c $< -o $@

$(SRC_CUDA)/aco_cuda.o: $(SRC_CUDA)/aco_cuda.cu $(SRC_CUDA)/aco_cuda_kernels.h $(SRC_CUDA)/aco_cuda_host_utils.h include/aco.h include/solution.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/aco_cuda_kernels.o: $(SRC_CUDA)/aco_cuda_kernels.cu $(SRC_CUDA)/aco_cuda_kernels.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/aco_cuda_host_utils.o: $(SRC_CUDA)/aco_cuda_host_utils.cu $(SRC_CUDA)/aco_cuda_host_utils.h include/solution.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SEQ_BIN): $(MAIN_SEQ_OBJ) $(SEQ_OBJ) $(CORE_OBJ)
	$(CC) $(DEBUG) $(CFLAGS) $^ $(LIBS) -o $@

$(OMP_BIN): $(MAIN_OMP_OBJ) $(OMP_OBJ) $(CORE_OBJ)
	$(CC) $(DEBUG) $(CFLAGS) $(OMPFLAGS) $^ $(LIBS) -o $@

$(MPI_BIN): $(MAIN_MPI_OBJ) $(MPI_OBJ) $(CORE_OBJ)
	$(MPICC) $(DEBUG) $(MPICFLAGS) $(OMPFLAGS) $^ $(LIBS) -o $@

$(CUDA_BIN): $(CUDA_CORE_OBJ) $(CUDA_OBJ)
	$(CUDACC) $(CUDAFLAGS) $^ $(CUDALIBS) -o $@

seq: $(SEQ_BIN)

omp: $(OMP_BIN)

mpi: $(MPI_BIN)

cuda: $(CUDA_BIN)

tests/test_aco.o: tests/test_aco.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(DEBUG) $(CFLAGS) -c $< -o $@

$(TEST_BIN): $(TEST_OBJ) $(SEQ_OBJ) $(OMP_OBJ) $(CORE_OBJ)
	$(CC) $(DEBUG) $(CFLAGS) $(OMPFLAGS) $^ $(LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

test-mpi: $(MPI_BIN)
	mpirun -np 2 ./$(MPI_BIN) 2 1

benchmark: seq omp mpi
	python3 scripts/benchmark_scaling.py

coverage:
	$(MAKE) clean
	$(MAKE) test CFLAGS="$(CFLAGS) $(COVERAGE_FLAGS)" LIBS="$(LIBS) $(COVERAGE_LIBS)"

debug:
	$(MAKE) DEBUG=-DDEBUG all

clean:
	rm -f $(SEQ_BIN) $(OMP_BIN) $(MPI_BIN) $(CUDA_BIN) $(TEST_BIN)
	find src tests -type f \( -name '*.o' -o -name '*.gcno' -o -name '*.gcda' \) -delete

.PHONY: all help clean debug test test-mpi benchmark coverage seq omp mpi cuda
