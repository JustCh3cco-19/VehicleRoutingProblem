CC=gcc
CUDACC?=/usr/local/cuda/bin/nvcc

# Set GPU arch
CUDA_ARCH?=75
CUDA_GENCODE=-arch=native
CUDA_FAST_MATH?=1
CUDA_MATH_FLAGS=
ifeq ($(CUDA_FAST_MATH),1)
CUDA_MATH_FLAGS=-use_fast_math
endif

SRC_SEQ=src_sequential
SRC_CUDA=src_cuda

CFLAGS=-O3 -Wall -Wextra -std=c11 -Iinclude -I$(SRC_SEQ) -I$(SRC_CUDA)
CUDAFLAGS=-O3 -Iinclude -I$(SRC_SEQ) -I$(SRC_CUDA) $(CUDA_GENCODE) $(CUDA_MATH_FLAGS)
LIBS=-lm
CUDALIBS=-lm -lcudart

SEQ_BIN=aco_vrp_seq
SEQ_INST_BIN=aco_vrp_seq_inst
CUDA_BIN=aco_vrp_cuda_vrp

CORE_OBJ=$(SRC_SEQ)/solution.o $(SRC_SEQ)/matrix.o $(SRC_SEQ)/instance_parser.o
CUDA_OBJ=$(SRC_CUDA)/aco_cuda.o $(SRC_CUDA)/aco_cuda_kernels.o $(SRC_CUDA)/aco_cuda_host_utils.o
CUDA_CORE_OBJ=$(SRC_CUDA)/main_vrp.o $(SRC_SEQ)/solution_cuda.o $(SRC_SEQ)/matrix_cuda.o $(SRC_SEQ)/instance_parser_cuda.o

all: $(SEQ_BIN) $(SEQ_INST_BIN) $(CUDA_BIN)

$(SRC_SEQ)/main.o: $(SRC_SEQ)/main.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/main_instance.o: $(SRC_SEQ)/main_instance.c include/aco.h include/matrix.h include/solution.h include/instance_parser.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/aco.o: $(SRC_SEQ)/aco.c include/aco.h include/matrix.h include/solution.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/solution.o: $(SRC_SEQ)/solution.c include/solution.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/matrix.o: $(SRC_SEQ)/matrix.c include/matrix.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/instance_parser.o: $(SRC_SEQ)/instance_parser.c include/instance_parser.h include/matrix.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC_SEQ)/solution_cuda.o: $(SRC_SEQ)/solution.c include/solution.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_SEQ)/matrix_cuda.o: $(SRC_SEQ)/matrix.c include/matrix.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_SEQ)/instance_parser_cuda.o: $(SRC_SEQ)/instance_parser.c include/instance_parser.h include/matrix.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/main_vrp.o: $(SRC_CUDA)/main_vrp.cu include/aco.h include/matrix.h include/solution.h include/instance_parser.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/aco_cuda.o: $(SRC_CUDA)/aco_cuda.cu $(SRC_CUDA)/aco_cuda_kernels.h $(SRC_CUDA)/aco_cuda_host_utils.h include/aco.h include/solution.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/aco_cuda_kernels.o: $(SRC_CUDA)/aco_cuda_kernels.cu $(SRC_CUDA)/aco_cuda_kernels.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SRC_CUDA)/aco_cuda_host_utils.o: $(SRC_CUDA)/aco_cuda_host_utils.cu $(SRC_CUDA)/aco_cuda_host_utils.h include/solution.h
	$(CUDACC) $(CUDAFLAGS) -c $< -o $@

$(SEQ_BIN): $(SRC_SEQ)/main.o $(SRC_SEQ)/aco.o $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

$(SEQ_INST_BIN): $(SRC_SEQ)/main_instance.o $(SRC_SEQ)/aco.o $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

$(CUDA_BIN): $(CUDA_CORE_OBJ) $(CUDA_OBJ)
	$(CUDACC) $(CUDAFLAGS) $^ $(CUDALIBS) -o $@

cuda-vrp: $(CUDA_BIN)

TEST_BIN=tests/test_aco

$(TEST_BIN): tests/test_aco.c $(SRC_SEQ)/aco.o $(CORE_OBJ)
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(SEQ_BIN) $(SEQ_INST_BIN) $(CUDA_BIN) $(TEST_BIN) src_sequential/*.o src_cuda/*.o tests/*.o

.PHONY: all clean cuda-vrp
