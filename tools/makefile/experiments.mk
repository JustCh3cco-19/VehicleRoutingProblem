exp_strong_openmp:
	@threads_list="$(EXP_STRONG_OPENMP_THREADS)"; \
	if [ "$$threads_list" = "auto" ]; then \
		max_threads="$${SLURM_CPUS_PER_TASK:-$$(nproc)}"; \
		t=1; \
		threads_list=""; \
		while [ "$$t" -le "$$max_threads" ]; do \
			threads_list="$$threads_list $$t"; \
			t=$$((t*2)); \
		done; \
	fi; \
	for t in $$threads_list; do \
		echo "[exp_strong_openmp] omp_threads=$$t"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_strong_openmp" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_strong_openmp" \
			SOLVE_BATCH_ID="strong_openmp_clients_t$$t" \
			SOLVE_CLIENTS="$(EXP_STRONG_CLIENTS_SERIES)" \
			SOLVE_MPI_RANKS=1 \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_strong_openmp_n8000_m256:
	@threads_list="1 2 4 8 16"; \
	for t in $$threads_list; do \
		echo "[exp_strong_openmp_n8000_m256] mpi_ranks=1 omp_threads=$$t n=8000 m=256"; \
		$(MAKE) solve_mpi \
			SOLVE_MANIFEST_MPI="instances/test_aligned/manifest_openmp_mpi_n8000_m256.csv" \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_strong_openmp_n8000_m256" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_strong_openmp_n8000_m256" \
			SOLVE_BATCH_ID="strong_openmp_n8000_m256_t$$t" \
			SOLVE_CLIENTS="8000" \
			SOLVE_MPI_RANKS=1 \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_strong_mpi:
	@ranks_list="$(EXP_STRONG_MPI_RANKS)"; \
	if [ "$$ranks_list" = "auto" ]; then \
		max_nodes="$(EXP_MAX_CLUSTER_NODES)"; \
		if [ -n "$${SLURM_JOB_NUM_NODES:-}" ] && [ "$${SLURM_JOB_NUM_NODES}" -lt "$$max_nodes" ]; then \
			max_nodes="$${SLURM_JOB_NUM_NODES}"; \
		fi; \
		r=1; \
		ranks_list=""; \
		while [ "$$r" -le "$$max_nodes" ]; do \
			ranks_list="$$ranks_list $$r"; \
			r=$$((r*2)); \
		done; \
	fi; \
	for r in $$ranks_list; do \
		echo "[exp_strong_mpi] mpi_ranks=$$r omp_threads=$(EXP_STRONG_MPI_OMP_THREADS)"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_strong_mpi" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_strong_mpi" \
			SOLVE_BATCH_ID="strong_mpi_clients_r$$r" \
			SOLVE_CLIENTS="$(EXP_STRONG_CLIENTS_SERIES)" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$(EXP_STRONG_MPI_OMP_THREADS)" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_strong_hybrid:
	@pairs="$(EXP_STRONG_HYBRID_PAIRS)"; \
	if [ "$$pairs" = "auto" ]; then \
		max_threads="$${SLURM_CPUS_PER_TASK:-$$(nproc)}"; \
		max_nodes="$(EXP_MAX_CLUSTER_NODES)"; \
		if [ -n "$${SLURM_JOB_NUM_NODES:-}" ] && [ "$${SLURM_JOB_NUM_NODES}" -lt "$$max_nodes" ]; then \
			max_nodes="$${SLURM_JOB_NUM_NODES}"; \
		fi; \
		p=1; \
		pairs=""; \
		while [ "$$p" -le "$$max_threads" ] && [ "$$p" -le "$$max_nodes" ]; do \
			pairs="$$pairs $$p $$p"; \
			p=$$((p*2)); \
		done; \
	fi; \
	set -- $$pairs; \
	while [ "$$#" -gt 0 ]; do \
		r="$$1"; \
		t="$$2"; \
		shift 2; \
		echo "[exp_strong_hybrid] mpi_ranks=$$r omp_threads=$$t n=$(EXP_STRONG_HYBRID_N)"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_strong_hybrid" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_strong_hybrid" \
			SOLVE_BATCH_ID="strong_hybrid_clients_r$$r_t$$t" \
			SOLVE_CLIENTS="$(EXP_STRONG_CLIENTS_SERIES)" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_weak_openmp:
	@pairs="$(EXP_WEAK_OPENMP_PAIRS)"; \
	if [ "$$pairs" = "auto" ]; then \
		max_threads="$${SLURM_CPUS_PER_TASK:-$$(nproc)}"; \
		t=1; \
		pairs=""; \
		while [ "$$t" -le "$$max_threads" ]; do \
			n=$$(( $(EXP_WEAK_BASE_N_PER_WORKER) * $$t )); \
			pairs="$$pairs $$t $$n"; \
			t=$$((t*2)); \
		done; \
	fi; \
	set -- $$pairs; \
	while [ "$$#" -gt 0 ]; do \
		t="$$1"; \
		n="$$2"; \
		shift 2; \
		echo "[exp_weak_openmp] omp_threads=$$t n=$$n"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_weak_openmp" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_weak_openmp" \
			SOLVE_BATCH_ID="weak_openmp_n$$n_t$$t" \
			SOLVE_CLIENTS="$$n" \
			SOLVE_MPI_RANKS=1 \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_weak_mpi:
	@pairs="$(EXP_WEAK_MPI_PAIRS)"; \
	if [ "$$pairs" = "auto" ]; then \
		max_nodes="$(EXP_MAX_CLUSTER_NODES)"; \
		if [ -n "$${SLURM_JOB_NUM_NODES:-}" ] && [ "$${SLURM_JOB_NUM_NODES}" -lt "$$max_nodes" ]; then \
			max_nodes="$${SLURM_JOB_NUM_NODES}"; \
		fi; \
		r=1; \
		pairs=""; \
		while [ "$$r" -le "$$max_nodes" ]; do \
			n=$$(( $(EXP_WEAK_BASE_N_PER_WORKER) * $$r )); \
			pairs="$$pairs $$r $$n"; \
			r=$$((r*2)); \
		done; \
	fi; \
	set -- $$pairs; \
	while [ "$$#" -gt 0 ]; do \
		r="$$1"; \
		n="$$2"; \
		shift 2; \
		echo "[exp_weak_mpi] mpi_ranks=$$r n=$$n omp_threads=$(EXP_WEAK_MPI_OMP_THREADS)"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_weak_mpi" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_weak_mpi" \
			SOLVE_BATCH_ID="weak_mpi_n$$n_r$$r" \
			SOLVE_CLIENTS="$$n" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$(EXP_WEAK_MPI_OMP_THREADS)" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_weak_hybrid:
	@triplets="$(EXP_WEAK_HYBRID_TRIPLETS)"; \
	if [ "$$triplets" = "auto" ]; then \
		max_threads="$${SLURM_CPUS_PER_TASK:-$$(nproc)}"; \
		max_nodes="$(EXP_MAX_CLUSTER_NODES)"; \
		if [ -n "$${SLURM_JOB_NUM_NODES:-}" ] && [ "$${SLURM_JOB_NUM_NODES}" -lt "$$max_nodes" ]; then \
			max_nodes="$${SLURM_JOB_NUM_NODES}"; \
		fi; \
		p=1; \
		triplets=""; \
		while [ "$$p" -le "$$max_threads" ] && [ "$$p" -le "$$max_nodes" ]; do \
			n=$$(( $(EXP_WEAK_BASE_N_PER_WORKER) * $$p * $$p )); \
			triplets="$$triplets $$p $$p $$n"; \
			p=$$((p*2)); \
		done; \
	fi; \
	set -- $$triplets; \
	while [ "$$#" -gt 0 ]; do \
		r="$$1"; \
		t="$$2"; \
		n="$$3"; \
		shift 3; \
		echo "[exp_weak_hybrid] mpi_ranks=$$r omp_threads=$$t n=$$n"; \
		$(MAKE) solve_mpi \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_weak_hybrid" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_weak_hybrid" \
			SOLVE_BATCH_ID="weak_hybrid_n$$n_r$$r_t$$t" \
			SOLVE_CLIENTS="$$n" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS=1 \
			SOLVE_MPI_RUNTIME_S=0 \
			SOLVE_MPI_STAGNATION_EPOCHS="$(EXP_STAGNATION_EPOCHS)" \
			SOLVE_MPI_MIN_REL_IMPROVEMENT="$(EXP_MIN_REL_IMPROVEMENT)"; \
	done

exp_all: exp_strong_openmp exp_weak_openmp exp_strong_mpi exp_strong_hybrid exp_weak_mpi exp_weak_hybrid
	@echo "Experiments completed."
	@echo "CSV dirs:"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_strong_openmp"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_weak_openmp"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_strong_mpi"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_strong_hybrid"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_weak_mpi"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_weak_hybrid"

# CUDA experiments
exp_cuda_scaling_input:
	@echo "[exp_cuda_scaling_input] input scaling with fixed m=$(EXP_CUDA_SCALING_M_FIXED)"
	$(MAKE) solve_cuda \
		SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_scaling_input" \
		SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_cuda_scaling_input" \
		SOLVE_CLIENTS="$(EXP_CUDA_SCALING_CLIENTS)" \
		SOLVE_CUDA_VARIANT="$(EXP_CUDA_VARIANT)" \
		SOLVE_CUDA_REPEATS="$(EXP_CUDA_REPEATS)" \
		SOLVE_SEQ_M="$(EXP_CUDA_SCALING_M_FIXED)" \
		SOLVE_CUDA_RUNTIME_S="$(EXP_CUDA_RUNTIME_S)" \
		SOLVE_CUDA_STAGNATION_EPOCHS="$(EXP_CUDA_STAGNATION_EPOCHS)" \
		SOLVE_CUDA_MIN_REL_IMPROVEMENT="$(EXP_CUDA_MIN_REL_IMPROVEMENT)" \
		SOLVE_BATCH_ID="cuda_scaling_input_m$(EXP_CUDA_SCALING_M_FIXED)"

exp_cuda_scaling_ants:
	@echo "[exp_cuda_scaling_ants] m scaling on clients=$(EXP_CUDA_CLIENTS_FUNCTIONAL)"
	@for m in $(EXP_CUDA_M_LIST); do \
		echo "[exp_cuda_scaling_ants] m=$$m"; \
		$(MAKE) solve_cuda \
			SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_scaling_ants" \
			SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_cuda_scaling_ants" \
			SOLVE_CLIENTS="$(EXP_CUDA_CLIENTS_FUNCTIONAL)" \
			SOLVE_CUDA_VARIANT="$(EXP_CUDA_VARIANT)" \
			SOLVE_CUDA_REPEATS="$(EXP_CUDA_REPEATS)" \
			SOLVE_SEQ_M="$$m" \
			SOLVE_CUDA_RUNTIME_S="$(EXP_CUDA_RUNTIME_S)" \
			SOLVE_CUDA_STAGNATION_EPOCHS="$(EXP_CUDA_STAGNATION_EPOCHS)" \
			SOLVE_CUDA_MIN_REL_IMPROVEMENT="$(EXP_CUDA_MIN_REL_IMPROVEMENT)" \
			SOLVE_BATCH_ID="cuda_scaling_ants_m$$m"; \
	done

exp_seq_for_cuda:
	@echo "[exp_seq_for_cuda] sequenziale (non scaling)"
	$(MAKE) solve_seq \
		SOLVE_CSV_DIR="$(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_vs_seq" \
		SOLVE_SOLUTIONS_DIR="$(RESULTS_ROOT)/solve_manifest/solutions/exp_cuda_vs_seq" \
		SOLVE_CLIENTS="$(EXP_CUDA_CLIENTS_FUNCTIONAL)" \
		SOLVE_SEQ_REPEATS="$(EXP_CUDA_REPEATS)" \
		SOLVE_SEQ_RUNTIME_S="$(EXP_CUDA_RUNTIME_S)" \
		SOLVE_SEQ_STAGNATION_EPOCHS="$(EXP_CUDA_STAGNATION_EPOCHS)" \
		SOLVE_SEQ_MIN_REL_IMPROVEMENT="$(EXP_CUDA_MIN_REL_IMPROVEMENT)" \
		SOLVE_BATCH_ID="seq_baseline_for_cuda"
exp_cuda_all: exp_cuda_scaling_input exp_cuda_scaling_ants exp_seq_for_cuda
	@echo "CUDA scaling + test sequenziale completed."
	@echo "CSV dirs:"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_scaling_input"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_scaling_ants"
	@echo "  $(RESULTS_ROOT)/solve_manifest/csv/exp_cuda_vs_seq"

# Practical campaign entry-points (batch-friendly)
exp_practical_cpu:
	@tag="$${PRACTICAL_TAG:-$$(date +%Y%m%d_%H%M%S)}"; \
	root="$(RESULTS_ROOT)/practical_campaign/$${tag}/cpu"; \
	echo "[exp_practical_cpu] root=$$root"; \
	$(PYTHON_BIN) scripts/run_practical_experiments.py \
		--launcher srun \
		--skip-cuda \
		--results-root "$$root" \
		$(PRACTICAL_COMMON_ARGS) \
		$(PRACTICAL_CPU_ARGS)

exp_practical_gpu:
	@tag="$${PRACTICAL_TAG:-$$(date +%Y%m%d_%H%M%S)}"; \
	root="$(RESULTS_ROOT)/practical_campaign/$${tag}/gpu"; \
	echo "[exp_practical_gpu] root=$$root"; \
	$(PYTHON_BIN) scripts/run_practical_experiments.py \
		--launcher srun \
		--only-cuda-sections \
		--openmp-threads 32 \
		--mpi-ranks 1 \
		--hybrid-pairs 1x32 \
		--quality-mpi-ranks 1 \
		--quality-mpi-threads 32 \
		--quality-hybrid 1x32 \
		--results-root "$$root" \
		$(PRACTICAL_COMMON_ARGS) \
		$(PRACTICAL_GPU_ARGS)
