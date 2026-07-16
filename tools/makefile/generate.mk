generate_problems:
	@mkdir -p "$(GEN_INST_DIR)"
	@manifest_seq="$(GEN_INST_DIR)/manifest.csv"; \
	manifest_mpi="$(GEN_INST_DIR)/manifest_openmp_mpi.csv"; \
	manifest_cuda="$(GEN_INST_DIR)/manifest_cuda.csv"; \
	rm -f "$(GEN_INST_DIR)"/n*_k*_s*.vrp "$$manifest_seq" "$$manifest_mpi" "$$manifest_cuda"; \
	header="profile,name,instance_path,n,K,m,solver_seed,instance_seed,layout_id,capacity_formula"; \
	echo "$$header" > "$$manifest_seq"; \
	echo "$$header" > "$$manifest_mpi"; \
	echo "$$header" > "$$manifest_cuda"; \
	target_cppv="$(GEN_TARGET_CUSTOMERS_PER_VEHICLE)"; \
	min_vehicles="$(GEN_MIN_VEHICLES)"; \
	max_vehicles="$(GEN_MAX_VEHICLES)"; \
	slack_percent="$(GEN_CAPACITY_SLACK_PERCENT)"; \
	m_cuda="$(GEN_CUDA_M)"; \
	idx=0; \
	for n in $$(echo "$(GEN_CLIENTS)" | tr ',' ' '); do \
		idx=$$((idx + 1)); \
		seed=$$(( $(GEN_SEED_BASE) + idx - 1 )); \
		if [ "$$target_cppv" -lt 1 ]; then target_cppv=1; fi; \
		K=$$(( (n + target_cppv - 1) / target_cppv )); \
		if [ $$K -lt $$min_vehicles ]; then K=$$min_vehicles; fi; \
		if [ $$K -gt $$max_vehicles ]; then K=$$max_vehicles; fi; \
		if [ $$n -le 2000 ]; then m_seq=32; m_mpi=64; \
		elif [ $$n -le 8000 ]; then m_seq=16; m_mpi=32; \
		elif [ $$n -le 16000 ]; then m_seq=8; m_mpi=16; \
		elif [ $$n -le 32000 ]; then m_seq=4; m_mpi=8; \
		else m_seq=3; m_mpi=4; fi; \
		name="n$${n}_k$${K}_s$${seed}"; \
		inst_path="$(GEN_INST_DIR)/$${name}.vrp"; \
		cap_formula="ceil((1+$$slack_percent/100)*n/K)"; \
		$(PYTHON_BIN) tools/python/generate_vrp_problem.py --name "$$name" --clients "$$n" --vehicles "$$K" --grid "$(GEN_GRID)" --seed "$$seed" --capacity-slack-percent "$$slack_percent" --output "$$inst_path" || exit $$?; \
		echo "generated,$$name,$$inst_path,$$n,$$K,$$m_seq,$(GEN_SOLVER_SEED),$$seed,grid$(GEN_GRID),$$cap_formula" >> "$$manifest_seq"; \
		echo "generated_mpi,$$name,$$inst_path,$$n,$$K,$$m_mpi,$(GEN_SOLVER_SEED),$$seed,grid$(GEN_GRID),$$cap_formula" >> "$$manifest_mpi"; \
		echo "generated_cuda,$$name,$$inst_path,$$n,$$K,$$m_cuda,$(GEN_SOLVER_SEED),$$seed,grid$(GEN_GRID),$$cap_formula" >> "$$manifest_cuda"; \
		echo "[gen] $$name"; \
	done; \
	tmp_seq="$$(mktemp)"; \
	tmp_mpi="$$(mktemp)"; \
	tmp_cuda="$$(mktemp)"; \
	head -n 1 "$$manifest_seq" > "$$tmp_seq"; \
	tail -n +2 "$$manifest_seq" | sort -u -t, -k4,4n -k8,8n >> "$$tmp_seq"; \
	mv "$$tmp_seq" "$$manifest_seq"; \
	head -n 1 "$$manifest_mpi" > "$$tmp_mpi"; \
	tail -n +2 "$$manifest_mpi" | sort -u -t, -k4,4n -k8,8n >> "$$tmp_mpi"; \
	mv "$$tmp_mpi" "$$manifest_mpi"; \
	head -n 1 "$$manifest_cuda" > "$$tmp_cuda"; \
	tail -n +2 "$$manifest_cuda" | sort -u -t, -k4,4n -k8,8n >> "$$tmp_cuda"; \
	mv "$$tmp_cuda" "$$manifest_cuda"; \
	echo "wrote $$manifest_seq"; \
	echo "wrote $$manifest_mpi"; \
	echo "wrote $$manifest_cuda"

generate_clean:
	@rm -f "$(GEN_INST_DIR)"/n*_k*_s*.vrp
	@rm -f "$(GEN_INST_DIR)/manifest.csv" "$(GEN_INST_DIR)/manifest_openmp_mpi.csv" "$(GEN_INST_DIR)/manifest_cuda.csv"
	@echo "cleaned generated files in $(GEN_INST_DIR)"

generate_problems_big:
	@$(MAKE) generate_problems GEN_CLIENTS="4000,8000,12000,16000,20000,24000,28000,32000"
