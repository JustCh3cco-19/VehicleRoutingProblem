solve_prepare:
	@mkdir -p $(SOLVE_OUT_DIR)
	@mkdir -p $(SOLVE_CSV_DIR)
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/pyvrp
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/seq
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/cuda
	@mkdir -p $(SOLVE_SOLUTIONS_DIR)/mpi
	@test -f "$(SOLVE_MANIFEST)" || (echo "missing manifest: $(SOLVE_MANIFEST)" && exit 1)
	@test -f "$(SOLVE_MANIFEST_MPI)" || (echo "missing manifest: $(SOLVE_MANIFEST_MPI)" && exit 1)
	@test -f "$(SOLVE_MANIFEST_CUDA)" || (echo "missing manifest: $(SOLVE_MANIFEST_CUDA)" && exit 1)

solve_pyvrp: solve_prepare
	@SOLVE_CSV_DIR="$(SOLVE_CSV_DIR)" \
	SOLVE_SOLUTIONS_DIR="$(SOLVE_SOLUTIONS_DIR)" \
	SOLVE_MANIFEST="$(SOLVE_MANIFEST)" \
	SOLVE_CLIENTS="$(SOLVE_CLIENTS)" \
	SOLVE_LIMIT="$(SOLVE_LIMIT)" \
	SOLVE_PYVRP_RUNTIME_S="$(SOLVE_PYVRP_RUNTIME_S)" \
	SOLVE_PYVRP_SEED="$(SOLVE_PYVRP_SEED)" \
	PYTHON_BIN="$(PYTHON_BIN)" \
	bash tools/bash/solve_pyvrp.sh

solve_seq: solve_prepare
	@rows=$$(tail -n +2 "$(SOLVE_MANIFEST)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
		| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } \
		| wc -l); \
	if [ "$$rows" -gt 0 ]; then \
		$(MAKE) seq; \
	else \
		echo "[seq] no matching rows in manifest -> skipping build"; \
	fi
	@SOLVE_CSV_DIR="$(SOLVE_CSV_DIR)" \
	SOLVE_SOLUTIONS_DIR="$(SOLVE_SOLUTIONS_DIR)" \
	SOLVE_MANIFEST="$(SOLVE_MANIFEST)" \
	SOLVE_CLIENTS="$(SOLVE_CLIENTS)" \
	SOLVE_LIMIT="$(SOLVE_LIMIT)" \
	SOLVE_SEQ_M="$(SOLVE_SEQ_M)" \
	SOLVE_SEQ_REPEATS="$(SOLVE_SEQ_REPEATS)" \
	SOLVE_SEQ_RUNTIME_EFFECTIVE="$(SOLVE_SEQ_RUNTIME_EFFECTIVE)" \
	SOLVE_SEQ_STAGNATION_EPOCHS="$(SOLVE_SEQ_STAGNATION_EPOCHS)" \
	SOLVE_SEQ_MIN_REL_IMPROVEMENT="$(SOLVE_SEQ_MIN_REL_IMPROVEMENT)" \
	bash tools/bash/solve_seq.sh

solve_cuda: solve_prepare
	@rows=$$(tail -n +2 "$(SOLVE_MANIFEST_CUDA)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
		| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } \
		| wc -l); \
	if [ "$$rows" -gt 0 ]; then \
		$(MAKE) cuda; \
	else \
		echo "[cuda] no matching rows in manifest -> skipping build"; \
	fi
	@SOLVE_CSV_DIR="$(SOLVE_CSV_DIR)" \
	SOLVE_SOLUTIONS_DIR="$(SOLVE_SOLUTIONS_DIR)" \
	SOLVE_MANIFEST_CUDA="$(SOLVE_MANIFEST_CUDA)" \
	SOLVE_MANIFEST="$(SOLVE_MANIFEST_CUDA)" \
	SOLVE_CLIENTS="$(SOLVE_CLIENTS)" \
	SOLVE_LIMIT="$(SOLVE_LIMIT)" \
	SOLVE_SEQ_M="$(SOLVE_SEQ_M)" \
	SOLVE_CUDA_REPEATS="$(SOLVE_CUDA_REPEATS)" \
	SOLVE_CUDA_RUNTIME_S="$(SOLVE_CUDA_RUNTIME_S)" \
	SOLVE_CUDA_STAGNATION_EPOCHS="$(SOLVE_CUDA_STAGNATION_EPOCHS)" \
	SOLVE_CUDA_MIN_REL_IMPROVEMENT="$(SOLVE_CUDA_MIN_REL_IMPROVEMENT)" \
	SOLVE_SEQ_RUNTIME_EFFECTIVE="$(SOLVE_SEQ_RUNTIME_EFFECTIVE)" \
	SOLVE_SEQ_STAGNATION_EPOCHS="$(SOLVE_SEQ_STAGNATION_EPOCHS)" \
	SOLVE_SEQ_MIN_REL_IMPROVEMENT="$(SOLVE_SEQ_MIN_REL_IMPROVEMENT)" \
	bash tools/bash/solve_cuda.sh

solve_mpi: solve_prepare
	@rows=$$(tail -n +2 "$(SOLVE_MANIFEST_MPI)" \
		| { if [ -n "$(SOLVE_CLIENTS)" ]; then awk -F, -v list="$(SOLVE_CLIENTS)" 'BEGIN{split(list,a,","); for(i in a) wanted[a[i]]=1} ($$4 in wanted)'; else cat; fi; } \
		| { if [ "$(SOLVE_LIMIT)" -gt 0 ]; then head -n "$(SOLVE_LIMIT)"; else cat; fi; } \
		| wc -l); \
	if [ "$$rows" -gt 0 ]; then \
		$(MAKE) openmp_mpi; \
	else \
		echo "[mpi] no matching rows in manifest -> skipping build"; \
	fi
	@SOLVE_CSV_DIR="$(SOLVE_CSV_DIR)" \
	SOLVE_SOLUTIONS_DIR="$(SOLVE_SOLUTIONS_DIR)" \
	SOLVE_MANIFEST_MPI="$(SOLVE_MANIFEST_MPI)" \
	SOLVE_CLIENTS="$(SOLVE_CLIENTS)" \
	SOLVE_LIMIT="$(SOLVE_LIMIT)" \
	SOLVE_MPI_REPEATS="$(SOLVE_MPI_REPEATS)" \
	SOLVE_MPI_RUNTIME_S="$(SOLVE_MPI_RUNTIME_S)" \
	SOLVE_MPI_STAGNATION_EPOCHS="$(SOLVE_MPI_STAGNATION_EPOCHS)" \
	SOLVE_MPI_MIN_REL_IMPROVEMENT="$(SOLVE_MPI_MIN_REL_IMPROVEMENT)" \
	SOLVE_MPI_RANKS="$(SOLVE_MPI_RANKS)" \
	SOLVE_MPI_OMP_THREADS="$(SOLVE_MPI_OMP_THREADS)" \
	SOLVE_MPI_LAUNCHER="$(SOLVE_MPI_LAUNCHER)" \
	bash tools/bash/solve_mpi.sh

solve_all: solve_pyvrp solve_seq solve_cuda solve_mpi
	@echo "solve CSV files are in $(SOLVE_CSV_DIR)"
	@echo "solve route files are in $(SOLVE_SOLUTIONS_DIR)"

smoke:
	@bash tools/bash/smoke_test.sh all

smoke_seq:
	@bash tools/bash/smoke_test.sh seq

smoke_mpi:
	@bash tools/bash/smoke_test.sh mpi

smoke_cuda:
	@bash tools/bash/smoke_test.sh cuda

solve_memory_growth_non_cuda:
	@$(MAKE) solve_seq SOLVE_CLIENTS="$(if $(SOLVE_CLIENTS),$(SOLVE_CLIENTS),4000,8000,12000,16000,20000,24000,28000,32000)" SOLVE_SEQ_REPEATS="$(SOLVE_MEMORY_GROWTH_REPEATS)"
	@$(MAKE) solve_mpi SOLVE_CLIENTS="$(if $(SOLVE_CLIENTS),$(SOLVE_CLIENTS),4000,8000,12000,16000,20000,24000,28000,32000)" SOLVE_MPI_REPEATS="$(SOLVE_MEMORY_GROWTH_REPEATS)"
