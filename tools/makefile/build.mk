all: $(BIN) $(OPENMP_MPI_BIN) $(CUDA_BIN)

seq: $(BIN)

src/main.o: src/main.c include/aco.h include/cli_common.h include/instance_parser.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SEQ_OBJ): $(SEQ_SRC) include/aco.h include/aco_config.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(ACO_SHARED_OBJ): $(ACO_SHARED_SRC) include/aco.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(ACO_CONFIG_OBJ): $(ACO_CONFIG_SRC) include/aco_config.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SOLUTION_OBJ): $(SOLUTION_SRC) include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(MATRIX_OBJ): $(MATRIX_SRC) include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(INSTANCE_PARSER_OBJ): $(INSTANCE_PARSER_SRC) include/instance_parser.h include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(CLI_COMMON_OBJ): $(CLI_COMMON_SRC) include/cli_common.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $^ $(LIBS) -o $@

openmp_mpi: $(OPENMP_MPI_BIN)

$(OPENMP_MPI_BIN): $(OPENMP_MPI_SRC) include/aco.h include/aco_config.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_SRC) $(LIBS) -o $@

cuda: $(CUDA_BIN)

$(CUDA_MAIN_OBJ): $(CUDA_MAIN_SRC) include/aco.h include/cli_common.h include/instance_parser.h include/matrix.h include/solution.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_ACO_OBJ): $(CUDA_ACO_SRC) include/aco.h include/aco_config.h include/aco_cuda_kernels.h include/matrix.h include/solution.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_KERNELS_OBJ): $(CUDA_KERNELS_SRC) include/aco_cuda_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_BIN): $(CUDA_OBJ)
	$(NVCC) $(CUDA_ARCH_FLAG) $^ -o $@

clean:
	rm -f $(BIN) $(OBJ) $(OPENMP_MPI_BIN) $(CUDA_BIN) $(LEGACY_BINARIES) \
		src/*.o src/seq/*.o src/openmp-mpi/*.o src/common/*.o src/cuda/*.o \
		src/cuda/*.ptx src/cuda/*.cubin src/cuda/*.fatbin \
		tests/*.o tests/*.out \
		$(COVERAGE_FILES) \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG
