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
LOCAL_SEARCH_SRC=$(COMMON_DIR)/local_search.c

SEQ_OBJ=$(SEQ_DIR)/aco_sequential.o
ACO_SHARED_OBJ=$(COMMON_DIR)/aco_shared.o
SOLUTION_OBJ=$(COMMON_DIR)/solution.o
MATRIX_OBJ=$(COMMON_DIR)/matrix.o
INSTANCE_PARSER_OBJ=$(COMMON_DIR)/instance_parser.o
LOCAL_SEARCH_OBJ=$(COMMON_DIR)/local_search.o

OBJ=src/main.o $(SEQ_OBJ) $(ACO_SHARED_OBJ) $(SOLUTION_OBJ) $(MATRIX_OBJ) $(INSTANCE_PARSER_OBJ) $(LOCAL_SEARCH_OBJ)

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
CUDA_COMMON_OBJ=$(SOLUTION_OBJ) $(MATRIX_OBJ) $(LOCAL_SEARCH_OBJ)
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
SOLVE_SEQ_STAGNATION_ITERS?=0
SOLVE_SEQ_M?=0
SOLVE_SEQ_T?=0
SOLVE_MPI_RUNTIME_S?=0
SOLVE_MPI_STAGNATION_ITERS?=0
SOLVE_PYVRP_SEED?=1234
SOLVE_LIMIT?=0
SOLVE_CLIENTS?=
SOLVE_MPI_RANKS?=2
SOLVE_MPI_OMP_THREADS?=2
SOLVE_CUDA_IMPROVEMENT?=0.001
SOLVE_CUDA_VARIANT?=$(CUDA_VARIANT)
SOLVE_SEQ_RUNTIME_EFFECTIVE=$(if $(strip $(SOLVE_SEQ_RUNTIME)),$(SOLVE_SEQ_RUNTIME),$(SOLVE_SEQ_RUNTIME_S))

GEN_INST_DIR?=instances/test_aligned
GEN_CLIENTS?=500,1000,2000,4000,8000,16000
GEN_SEED_BASE?=19000
GEN_GRID?=100
GEN_SOLVER_SEED?=1234

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
	@printf "VehicleRoutingProblem - Make Help\n"
	@printf "=================================\n\n"
	@printf "COMANDI ESEMPIO:\n"
	@printf "  make generate_problems GEN_CLIENTS=500,1000,4000,8000\n"
	@printf "  make solve_pyvrp SOLVE_CLIENTS=500,1000 SOLVE_PYVRP_RUNTIME_S=20\n"
	@printf "  make solve_seq   SOLVE_CLIENTS=500,1000,2000 SOLVE_SEQ_RUNTIME_S=20 SOLVE_SEQ_T=500 SOLVE_SEQ_STAGNATION_ITERS=80\n"
	@printf "  make solve_cuda  SOLVE_CLIENTS=500,1000\n"
	@printf "  make solve_mpi   SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_RUNTIME_S=20 SOLVE_MPI_STAGNATION_ITERS=80\n"
	@printf "  make solve_all   SOLVE_CLIENTS=500,1000\n\n"
	@printf "TARGET GENERAZIONE PROBLEMI:\n"
	@printf "  %-42s | %s\n" "generate_problems" "Genera istanze .vrp + manifest.csv + manifest_openmp_mpi.csv"
	@printf "  %-42s | %s\n\n" "generate_clean" "Rimuove istanze/manifest generati in GEN_INST_DIR"
	@printf "TARGET BUILD:\n"
	@printf "  %-42s | %s\n" "all" "Compila il solver sequenziale -> aco_vrp_seq.out"
	@printf "  %-42s | %s\n" "openmp_mpi" "Compila il solver MPI+OpenMP -> aco_vrp_openmp_mpi.out"
	@printf "  %-42s | %s\n" "cuda" "Compila il solver CUDA -> aco_vrp_cuda.out"
	@printf "  %-42s | %s\n" "debug" "Ricompila con flag -DDEBUG"
	@printf "  %-42s | %s\n\n" "clean" "Pulisce binari, oggetti, coverage e artefatti CUDA intermedi"
	@printf "TARGET TEST / BENCHMARK ESISTENTI:\n"
	@printf "  %-42s | %s\n" "sequential_tests" "Esegue test progressivi C (sequenziale) su scala crescente"
	@printf "  %-42s | %s\n" "openmp_mpi_tests" "Esegue test OpenMP+MPI 'light' (fino a 16k clienti)"
	@printf "  %-42s | %s\n" "openmp_mpi_tests_heavy" "Esegue test OpenMP+MPI 'heavy' (24k..100k clienti)"
	@printf "  %-42s | %s\n" "openmp_mpi_tests_resume" "Riprende il profilo light dal checkpoint"
	@printf "  %-42s | %s\n" "openmp_mpi_tests_reset" "Resetta checkpoint light e riesegue"
	@printf "  %-42s | %s\n" "openmp_mpi_tests_heavy_resume" "Riprende il profilo heavy dal checkpoint"
	@printf "  %-42s | %s\n" "openmp_mpi_tests_heavy_reset" "Resetta checkpoint heavy e riesegue"
	@printf "  %-42s | %s\n" "c_compare_case" "Compila helper single-case sequenziale per confronto con PyVRP"
	@printf "  %-42s | %s\n\n" "c_compare_case_mpi" "Compila helper single-case MPI+OpenMP per confronto con PyVRP"
	@printf "TARGET SOLVE DA MANIFEST (PRODUZIONE DATI):\n"
	@printf "  %-42s | %s\n" "solve_prepare" "Crea cartelle output e verifica esistenza manifest"
	@printf "  %-42s | %s\n" "solve_pyvrp" "Legge SOLVE_MANIFEST, risolve con PyVRP, salva CSV + route txt"
	@printf "  %-42s | %s\n" "solve_seq" "Legge SOLVE_MANIFEST, esegue solver sequenziale, salva CSV + route txt"
	@printf "  %-42s | %s\n" "solve_cuda" "Legge SOLVE_MANIFEST, esegue solver CUDA, salva CSV + route txt"
	@printf "  %-42s | %s\n" "solve_mpi" "Legge SOLVE_MANIFEST_MPI, esegue solver MPI, salva CSV + route txt"
	@printf "  %-42s | %s\n\n" "solve_all" "Esegue in sequenza: solve_pyvrp, solve_seq, solve_cuda, solve_mpi"
	@printf "VARIABILI PER solve_*:\n"
	@printf "  %-60s | %s\n" "RESULTS_ROOT=results" "Root unica risultati (solve/scaling/detached)"
	@printf "  %-60s | %s\n" "SOLVE_OUT_DIR=$(RESULTS_ROOT)/solve_manifest" "Directory base output solve_*"
	@printf "  %-60s | %s\n" "SOLVE_CSV_DIR=$(SOLVE_OUT_DIR)/csv" "Directory CSV solve_*"
	@printf "  %-60s | %s\n" "SOLVE_SOLUTIONS_DIR=$(SOLVE_OUT_DIR)/solutions" "Directory route per backend"
	@printf "  %-60s | %s\n" "SOLVE_MANIFEST=.../manifest.csv" "Manifest usato da solve_pyvrp/solve_seq/solve_cuda"
	@printf "  %-60s | %s\n" "SOLVE_MANIFEST_MPI=.../manifest_openmp_mpi.csv" "Manifest usato da solve_mpi"
	@printf "  %-60s | %s\n" "SOLVE_CLIENTS=500,1000,4000" "Filtra per numero clienti n (colonna 4 del manifest)"
	@printf "  %-60s | %s\n" "SOLVE_LIMIT=5" "Limita alle prime N righe dopo i filtri (0 = tutte)"
	@printf "  %-60s | %s\n" "SOLVE_PYVRP_RUNTIME_S=10" "Budget tempo per istanza PyVRP (secondi)"
	@printf "  %-60s | %s\n" "SOLVE_SEQ_RUNTIME_S=0" "Budget tempo seq (secondi, 0=disattivo)"
	@printf "  %-60s | %s\n" "SOLVE_SEQ_RUNTIME=60" "Alias compatibile di SOLVE_SEQ_RUNTIME_S"
	@printf "  %-60s | %s\n" "SOLVE_SEQ_STAGNATION_ITERS=0" "Stop seq per N iterazioni senza miglioramento (0=off)"
	@printf "  %-60s | %s\n" "SOLVE_SEQ_M=0" "Override m (formiche) per solve_seq; 0=usa manifest"
	@printf "  %-60s | %s\n" "SOLVE_SEQ_T=0" "Override T (iterazioni) per solve_seq; 0=usa manifest"
	@printf "  %-60s | %s\n" "SOLVE_MPI_RUNTIME_S=0" "Budget tempo per istanza MPI+OpenMP (secondi, 0=disattivo)"
	@printf "  %-60s | %s\n" "SOLVE_MPI_STAGNATION_ITERS=0" "Stop MPI per N iterazioni senza miglioramento (0=off)"
	@printf "  %-60s | %s\n" "SOLVE_PYVRP_SEED=1234" "Seed PyVRP"
	@printf "  %-60s | %s\n" "SOLVE_MPI_RANKS=2" "Numero processi MPI per solve_mpi"
	@printf "  %-60s | %s\n" "SOLVE_MPI_OMP_THREADS=2" "Thread OpenMP per rank MPI"
	@printf "  %-60s | %s\n\n" "SOLVE_CUDA_IMPROVEMENT=0.001" "Soglia miglioramento early-stop CUDA"
	@printf "VARIABILI GENERAZIONE:\n"
	@printf "  %-60s | %s\n" "GEN_INST_DIR=instances/test_aligned" "Directory dove scrivere .vrp e manifest"
	@printf "  %-60s | %s\n" "GEN_CLIENTS=500,1000,2000" "Taglie n da generare (CSV separato da virgole)"
	@printf "  %-60s | %s\n" "GEN_SEED_BASE=19000" "Seed base; per ogni istanza incrementa di 1"
	@printf "  %-60s | %s\n" "GEN_GRID=100" "Griglia coordinate passata al generatore"
	@printf "  %-60s | %s\n" "GEN_SOLVER_SEED=1234" "Seed solver scritto nei manifest"
	@printf "  %-60s | %s\n\n" "Capacita generata" "Sempre n-K+3 con domanda unitaria per cliente"
	@printf "OUTPUT GENERATI DA solve_*:\n"
	@printf "  %-85s | %s\n" "$(SOLVE_CSV_DIR)/manifest_pyvrp_per_instance_results.csv" "CSV PyVRP con elapsed, memoria (max_rss_kb), costo"
	@printf "  %-85s | %s\n" "$(SOLVE_CSV_DIR)/manifest_seq_per_instance_results.csv" "CSV solver sequenziale"
	@printf "  %-85s | %s\n" "$(SOLVE_CSV_DIR)/manifest_cuda_per_instance_results.csv" "CSV solver CUDA"
	@printf "  %-85s | %s\n" "$(SOLVE_CSV_DIR)/manifest_openmp_mpi_per_instance_results.csv" "CSV solver MPI+OpenMP"
	@printf "  %-85s | %s\n\n" "$(SOLVE_SOLUTIONS_DIR)/<backend>/*.txt" "Route complete per ogni istanza/backend"
	@printf "OUTPUT GENERATI DA test/benchmark:\n"
	@printf "  %-60s | %s\n" "sequential_tests" "CSV: $(SCALING_DIR)/scaling_progressive_c.csv"
	@printf "  %-60s | %s\n" "openmp_mpi_tests" "CSV: $(SCALING_DIR)/scaling_progressive_openmp_mpi_light.csv"
	@printf "  %-60s | %s\n" "openmp_mpi_tests_heavy" "CSV: $(SCALING_DIR)/scaling_progressive_openmp_mpi_heavy.csv"
	@printf "  %-60s | %s\n" "TEST_MODE=background" "Log/PID/CMD in $(TEST_LOGS_DIR)/"
	@printf "  %-60s | %s\n\n" "Checkpoint MPI" "$(LIGHT_CHECKPOINT_PATH) / $(HEAVY_CHECKPOINT_PATH)"
	@printf "VARIABILI TEST DETACHED:\n"
	@printf "  %-60s | %s\n" "TEST_MODE=background|foreground" "background (default): nohup detached; foreground: output live"
	@printf "  %-60s | %s\n\n" "TEST_LOGS_DIR=$(RESULTS_ROOT)/detached" "Cartella log/pid/cmd per run detached"

src/main.o: src/main.c include/aco.h include/instance_parser.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SEQ_OBJ): $(SEQ_SRC) include/aco.h include/matrix.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(ACO_SHARED_OBJ): $(ACO_SHARED_SRC) include/aco.h include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(SOLUTION_OBJ): $(SOLUTION_SRC) include/solution.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(MATRIX_OBJ): $(MATRIX_SRC) include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(INSTANCE_PARSER_OBJ): $(INSTANCE_PARSER_SRC) include/instance_parser.h include/matrix.h
	$(CC) $(EXTRA_FLAGS) $(FLAGS) $(FORCE_OPT) $(PERF_FLAGS) -c $< -o $@

$(LOCAL_SEARCH_OBJ): $(LOCAL_SEARCH_SRC) include/local_search.h include/aco.h include/solution.h
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

$(CUDA_MAIN_OBJ): $(CUDA_MAIN_SRC) include/aco.h include/instance_parser.h include/matrix.h include/solution.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_ACO_OBJ): $(CUDA_ACO_SRC) include/aco.h include/aco_cuda_$(CUDA_VARIANT)_kernels.h include/matrix.h include/solution.h include/local_search.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_KERNELS_OBJ): $(CUDA_KERNELS_SRC) include/aco_cuda_$(CUDA_VARIANT)_kernels.h
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH_FLAG) -c $< -o $@

$(CUDA_BIN): $(CUDA_OBJ)
	$(NVCC) $(CUDA_ARCH_FLAG) $^ -o $@

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

generate_problems:
	@mkdir -p "$(GEN_INST_DIR)"
	@manifest_seq="$(GEN_INST_DIR)/manifest.csv"; \
	manifest_mpi="$(GEN_INST_DIR)/manifest_openmp_mpi.csv"; \
	echo "profile,name,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,capacity_formula" > "$$manifest_seq"; \
	echo "profile,name,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,capacity_formula" > "$$manifest_mpi"; \
	idx=0; \
	for n in $$(echo "$(GEN_CLIENTS)" | tr ',' ' '); do \
		idx=$$((idx + 1)); \
		seed=$$(( $(GEN_SEED_BASE) + idx - 1 )); \
		K=$$(( n / 1000 )); \
		if [ $$K -lt 8 ]; then K=8; fi; \
		if [ $$K -gt 128 ]; then K=128; fi; \
		if [ $$n -le 2000 ]; then m_seq=32; T_seq=20; m_mpi=64; T_mpi=40; \
		elif [ $$n -le 8000 ]; then m_seq=16; T_seq=10; m_mpi=32; T_mpi=20; \
		elif [ $$n -le 16000 ]; then m_seq=8; T_seq=6; m_mpi=16; T_mpi=12; \
		elif [ $$n -le 32000 ]; then m_seq=4; T_seq=4; m_mpi=8; T_mpi=8; \
		else m_seq=3; T_seq=3; m_mpi=4; T_mpi=6; fi; \
		name="n$${n}_k$${K}_s$${seed}"; \
		inst_path="$(GEN_INST_DIR)/$${name}.vrp"; \
		cap_formula="ceil(n/K)"; \
		$(PYTHON_BIN) scripts/generate_vrp_problem.py --name "$$name" --clients "$$n" --vehicles "$$K" --grid "$(GEN_GRID)" --seed "$$seed" --output "$$inst_path" || exit $$?; \
		echo "generated,$$name,$$inst_path,$$n,$$K,$$m_seq,$$T_seq,$(GEN_SOLVER_SEED),$$seed,grid$(GEN_GRID),$$cap_formula" >> "$$manifest_seq"; \
		echo "generated_mpi,$$name,$$inst_path,$$n,$$K,$$m_mpi,$$T_mpi,$(GEN_SOLVER_SEED),$$seed,grid$(GEN_GRID),$$cap_formula" >> "$$manifest_mpi"; \
		echo "[gen] $$name"; \
	done; \
	echo "wrote $$manifest_seq"; \
	echo "wrote $$manifest_mpi"

generate_clean:
	@rm -f "$(GEN_INST_DIR)"/n*_k*_s*.vrp
	@rm -f "$(GEN_INST_DIR)/manifest.csv" "$(GEN_INST_DIR)/manifest_openmp_mpi.csv"
	@echo "cleaned generated files in $(GEN_INST_DIR)"

solve_prepare:
	@mkdir -p $(SOLVE_OUT_DIR)
	@mkdir -p $(SOLVE_CSV_DIR)
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/pyvrp
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/seq
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/cuda
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/cuda_$(SOLVE_CUDA_VARIANT)
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/mpi
	@mkdir -p $(SCALING_DIR)
	@test -f "$(SOLVE_MANIFEST)" || (echo "missing manifest: $(SOLVE_MANIFEST)" && exit 1)
	@test -f "$(SOLVE_MANIFEST_MPI)" || (echo "missing manifest: $(SOLVE_MANIFEST_MPI)" && exit 1)

solve_pyvrp: solve_prepare
	@csv="$(SOLVE_CSV_DIR)/manifest_pyvrp_per_instance_results.csv"; \
	sol_dir="$(SOLVE_SOLUTIONS_DIR)/pyvrp"; \
	echo "name,profile,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,status,elapsed_s,max_rss_kb,best_cost,error" > "$$csv"; \
	py="$(PYTHON_BIN)"; \
	if [ -x "VRP/bin/python" ]; then py="VRP/bin/python"; fi; \
	( tail -n +2 "$(SOLVE_MANIFEST)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
		| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } ) | while IFS=, read -r profile name instance_path n K m T solver_seed instance_seed layout_id capacity_formula; do \
		sol_file="$$sol_dir/$${name}_pyvrp_solution.txt"; \
		rss_file=$$(mktemp); \
		start_ns=$$(date +%s%N); \
		out=$$(/usr/bin/time -f "%M" -o "$$rss_file" $$py -c "from pyvrp import Model, read, stop; import sys; p=sys.argv[1]; rt=float(sys.argv[2]); sd=int(sys.argv[3]); sol=sys.argv[4]; inst=read(p, round_func='round'); model=Model.from_data(inst); res=model.solve(stop.MaxRuntime(rt), seed=sd); best=res.best; f=open(sol,'w',encoding='utf-8'); [f.write(f'Route {i+1}: ' + ' '.join(map(str, r.visits())) + '\\n') for i,r in enumerate(best.routes())]; f.write(f'Cost: {best.distance():.6f}\\n'); f.close(); print(f'best_cost={best.distance():.6f}' if best.is_feasible() else 'best_cost=')" "$$instance_path" "$(SOLVE_PYVRP_RUNTIME_S)" "$(SOLVE_PYVRP_SEED)" "$$sol_file" 2>&1); \
		rc=$$?; \
		end_ns=$$(date +%s%N); \
		elapsed=$$(awk "BEGIN {printf \"%.6f\", ($$end_ns-$$start_ns)/1000000000}"); \
		rss_kb=$$(cat "$$rss_file" 2>/dev/null); \
		rm -f "$$rss_file"; \
		if [ $$rc -eq 0 ]; then \
			cost=$$(printf '%s\n' "$$out" | sed -n 's/^best_cost=//p' | tail -n1); \
			[ -n "$$cost" ] || cost=""; \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,ok,$$elapsed,$$rss_kb,$$cost," >> "$$csv"; \
		else \
			err=$$(printf '%s' "$$out" | tr '\n' ' ' | tr ',' ';'); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,error,$$elapsed,$$rss_kb,,$$err" >> "$$csv"; \
		fi; \
		echo "[pyvrp] $$name done"; \
	done; \
	echo "wrote $$csv"

solve_seq: solve_prepare all
	@csv="$(SOLVE_CSV_DIR)/manifest_seq_per_instance_results.csv"; \
	sol_dir="$(SOLVE_SOLUTIONS_DIR)/seq"; \
	echo "name,profile,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,status,elapsed_s,best_cost,error" > "$$csv"; \
	( tail -n +2 "$(SOLVE_MANIFEST)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
		| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } ) | while IFS=, read -r profile name instance_path n K m T solver_seed instance_seed layout_id capacity_formula; do \
		m_run="$$m"; \
		T_run="$$T"; \
		if [ "$(SOLVE_SEQ_M)" -gt 0 ]; then m_run="$(SOLVE_SEQ_M)"; fi; \
		if [ "$(SOLVE_SEQ_T)" -gt 0 ]; then T_run="$(SOLVE_SEQ_T)"; fi; \
		sol_file="$$sol_dir/$${name}_seq_solution.txt"; \
		time_file=$$(mktemp); \
		out=$$(/usr/bin/time -f "%e" -o "$$time_file" env ACO_SOLVER_TIMEOUT_SECONDS="$(SOLVE_SEQ_RUNTIME_EFFECTIVE)" ACO_SOLVER_STAGNATION_ITERS="$(SOLVE_SEQ_STAGNATION_ITERS)" ./aco_vrp_seq.out "$$instance_path" "$$K" "$$m_run" "$$T_run" "$$solver_seed" 2>&1); \
		rc=$$?; \
		elapsed=$$(cat "$$time_file" 2>/dev/null); \
		rm -f "$$time_file"; \
		[ -n "$$elapsed" ] || elapsed=""; \
		printf '%s\n' "$$out" > "$$sol_file"; \
		if [ $$rc -eq 0 ]; then \
			cost=$$(printf '%s\n' "$$out" | sed -n 's/^best cost: //p' | tail -n1); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m_run,$$T_run,$$solver_seed,$$instance_seed,$$layout_id,ok,$$elapsed,$$cost," >> "$$csv"; \
		else \
			err=$$(printf '%s' "$$out" | tr '\n' ' ' | tr ',' ';'); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m_run,$$T_run,$$solver_seed,$$instance_seed,$$layout_id,error,$$elapsed,,$$err" >> "$$csv"; \
		fi; \
		echo "[seq] $$name done"; \
	done; \
	echo "wrote $$csv"

solve_cuda: solve_prepare cuda
	@csv="$(SOLVE_CSV_DIR)/manifest_cuda_$(SOLVE_CUDA_VARIANT)_per_instance_results.csv"; \
	sol_dir="$(SOLVE_SOLUTIONS_DIR)/cuda_$(SOLVE_CUDA_VARIANT)"; \
		echo "name,profile,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,status,elapsed_s,best_cost,error" > "$$csv"; \
		( tail -n +2 "$(SOLVE_MANIFEST)" \
			| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
			| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } ) | while IFS=, read -r profile name instance_path n K m T solver_seed instance_seed layout_id capacity_formula; do \
			sol_file="$$sol_dir/$${name}_cuda_$(SOLVE_CUDA_VARIANT)_solution.txt"; \
			time_file=$$(mktemp); \
			out=$$(/usr/bin/time -f "%e" -o "$$time_file" env ACO_SOLVER_TIMEOUT_SECONDS="$(SOLVE_SEQ_RUNTIME_EFFECTIVE)" ACO_SOLVER_STAGNATION_ITERS="$(SOLVE_SEQ_STAGNATION_ITERS)" ./aco_vrp_cuda.out "$$instance_path" "$$K" "$$m" "$$T" "$$solver_seed" 2>&1); \
			rc=$$?; \
			elapsed=$$(cat "$$time_file" 2>/dev/null); \
			rm -f "$$time_file"; \
		[ -n "$$elapsed" ] || elapsed=""; \
		printf '%s\n' "$$out" > "$$sol_file"; \
		if [ $$rc -eq 0 ]; then \
			cost=$$(printf '%s\n' "$$out" | sed -n -e 's/^best cost: //p' -e 's/^Final Best Cost: //p' | tail -n1); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,ok,$$elapsed,$$cost," >> "$$csv"; \
		else \
			err=$$(printf '%s' "$$out" | tr '\n' ' ' | tr ',' ';'); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,error,$$elapsed,,$$err" >> "$$csv"; \
		fi; \
		echo "[cuda] $$name done"; \
	done; \
	echo "wrote $$csv"

solve_mpi: solve_prepare openmp_mpi
	@csv="$(SOLVE_CSV_DIR)/manifest_openmp_mpi_per_instance_results.csv"; \
	sol_dir="$(SOLVE_SOLUTIONS_DIR)/mpi"; \
	echo "name,profile,instance_path,n,K,m,T,solver_seed,instance_seed,layout_id,status,elapsed_s,best_cost,error" > "$$csv"; \
	( tail -n +2 "$(SOLVE_MANIFEST_MPI)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
			| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } ) | while IFS=, read -r profile name instance_path n K m T solver_seed instance_seed layout_id capacity_formula; do \
			sol_file="$$sol_dir/$${name}_mpi_solution.txt"; \
			time_file=$$(mktemp); \
			out=$$(/usr/bin/time -f "%e" -o "$$time_file" env ACO_SOLVER_TIMEOUT_SECONDS="$(SOLVE_MPI_RUNTIME_S)" ACO_SOLVER_STAGNATION_ITERS="$(SOLVE_MPI_STAGNATION_ITERS)" OMP_NUM_THREADS="$(SOLVE_MPI_OMP_THREADS)" mpirun -np "$(SOLVE_MPI_RANKS)" ./aco_vrp_openmp_mpi.out "$$instance_path" "$$K" "$$m" "$$T" "$$solver_seed" </dev/null 2>&1); \
			rc=$$?; \
			elapsed=$$(cat "$$time_file" 2>/dev/null); \
			rm -f "$$time_file"; \
			[ -n "$$elapsed" ] || elapsed=""; \
			printf '%s\n' "$$out" > "$$sol_file"; \
		if [ $$rc -eq 0 ]; then \
			cost=$$(printf '%s\n' "$$out" | sed -n 's/^best cost: //p' | tail -n1); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,ok,$$elapsed,$$cost," >> "$$csv"; \
		else \
			err=$$(printf '%s' "$$out" | tr '\n' ' ' | tr ',' ';'); \
			echo "$$name,$$profile,$$instance_path,$$n,$$K,$$m,$$T,$$solver_seed,$$instance_seed,$$layout_id,error,$$elapsed,,$$err" >> "$$csv"; \
		fi; \
		echo "[mpi] $$name done"; \
	done; \
	echo "wrote $$csv"

solve_all: solve_pyvrp solve_seq solve_cuda solve_mpi
	@echo "solve CSV files are in $(SOLVE_CSV_DIR)"
	@echo "solve route files are in $(SOLVE_SOLUTIONS_DIR)"

clean:
	rm -f $(BIN) $(OBJ) $(SEQUENTIAL_TESTS_BIN) $(SEQUENTIAL_TESTS_OBJ) $(OPENMP_MPI_BIN) $(CUDA_BIN) $(LEGACY_BINARIES) \
		$(OPENMP_MPI_TESTS_BIN) $(OPENMP_MPI_TESTS_HEAVY_BIN) \
		src/*.o src/seq/*.o src/openmp-mpi/*.o src/common/*.o src/cuda/*.o \
		src/cuda/*.ptx src/cuda/*.cubin src/cuda/*.fatbin \
		tests/*.o tests/*.out \
		$(COVERAGE_FILES) \
		report/*.aux report/*.log report/*.out

debug:
	$(MAKE) EXTRA_FLAGS=-DDEBUG

.PHONY: all help clean debug openmp_mpi cuda sequential_tests openmp_mpi_tests openmp_mpi_tests_heavy openmp_mpi_tests_resume openmp_mpi_tests_reset openmp_mpi_tests_heavy_resume openmp_mpi_tests_heavy_reset c_compare_case c_compare_case_mpi generate_problems generate_clean solve_prepare solve_pyvrp solve_seq solve_cuda solve_mpi solve_all
