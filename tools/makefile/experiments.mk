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
			SOLVE_BATCH_ID="strong_openmp_n$(EXP_STRONG_OPENMP_N)_t$$t" \
			SOLVE_CLIENTS="$(EXP_STRONG_OPENMP_N)" \
			SOLVE_MPI_RANKS=1 \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
			SOLVE_BATCH_ID="strong_mpi_n$(EXP_STRONG_MPI_N)_r$$r" \
			SOLVE_CLIENTS="$(EXP_STRONG_MPI_N)" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$(EXP_STRONG_MPI_OMP_THREADS)" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
			SOLVE_BATCH_ID="strong_hybrid_n$(EXP_STRONG_HYBRID_N)_r$$r_t$$t" \
			SOLVE_CLIENTS="$(EXP_STRONG_HYBRID_N)" \
			SOLVE_MPI_RANKS="$$r" \
			SOLVE_MPI_OMP_THREADS="$$t" \
			SOLVE_MPI_LAUNCHER="$(EXP_MPI_LAUNCHER)" \
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
			SOLVE_MPI_REPEATS="$(EXP_REPEATS)" \
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
