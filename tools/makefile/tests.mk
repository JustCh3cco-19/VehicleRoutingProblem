$(SEQUENTIAL_TESTS_OBJ): $(SEQUENTIAL_TESTS_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SEQUENTIAL_TESTS_BIN): $(SEQUENTIAL_TESTS_OBJ) $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ)
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $^ $(LIBS) -o $@

$(C_COMPARE_CASE_BIN): $(C_COMPARE_CASE_SRC) $(SEQ_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(C_COMPARE_CASE_SRC) $(SEQ_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(LIBS) -o $@

$(C_COMPARE_CASE_MPI_BIN): $(C_COMPARE_CASE_MPI_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(C_COMPARE_CASE_MPI_SRC) $(PAR_SRC) $(ACO_SHARED_SRC) $(SOLUTION_SRC) $(MATRIX_SRC) $(LIBS) -o $@

$(OPENMP_MPI_TESTS_BIN): $(OPENMP_MPI_TESTS_BUILD_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_TESTS_BUILD_SRC) $(LIBS) -o $@

$(OPENMP_MPI_TESTS_HEAVY_BIN): $(OPENMP_MPI_TESTS_HEAVY_BUILD_SRC) include/aco.h include/matrix.h include/solution.h
	$(MPICC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) $(OMPFLAG) -DUSE_MPI $(OPENMP_MPI_TESTS_HEAVY_BUILD_SRC) $(LIBS) -o $@

sequential_tests: $(SEQUENTIAL_TESTS_BIN)
	$(call RUN_TEST_CMD,sequential_tests,./$(SEQUENTIAL_TESTS_BIN) --csv $(SCALING_DIR)/scaling_progressive_c.csv --input-log $(SCALING_DIR)/sequential_test_inputs.log)

openmp_mpi_tests: $(OPENMP_MPI_TESTS_BIN)
	$(call RUN_TEST_CMD,openmp_mpi_tests,OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(OPENMP_MPI_TESTS_BIN) --csv $(SCALING_DIR)/scaling_progressive_openmp_mpi_light.csv --input-log $(SCALING_DIR)/openmp_mpi_test_inputs_light.log $(LIGHT_CP_FLAGS) $(MPI_TEST_ARGS))

openmp_mpi_tests_heavy: $(OPENMP_MPI_TESTS_HEAVY_BIN)
	$(call RUN_TEST_CMD,openmp_mpi_tests_heavy,OMP_NUM_THREADS=$(MPI_OMP_THREADS) mpirun -np $(MPI_NP) ./$(OPENMP_MPI_TESTS_HEAVY_BIN) --csv $(SCALING_DIR)/scaling_progressive_openmp_mpi_heavy.csv --input-log $(SCALING_DIR)/openmp_mpi_test_inputs_heavy.log $(HEAVY_CP_FLAGS) $(MPI_TEST_ARGS))

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
