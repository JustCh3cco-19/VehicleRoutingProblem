# Parallel VRP Implementation Analysis (Seq, OpenMP+MPI, CUDA)

Autori: `<nome1>`, `<nome2>`  
Data: `<YYYY-MM-DD>`

## 1 Introduction

Descrivi CVRP + ACO, obiettivo del progetto, differenza tra backend:
- Sequential baseline (`src/seq/`)
- OpenMP+MPI (`src/openmp-mpi/`)
- CUDA (`src/cuda/`)

### 1.0.1 Algorithm Structure

Inserisci pseudocodice ACO-CVRP (stile “Algorithm 1” del PDF):
1. inizializzazione feromoni `tau0`
2. ciclo iterativo
3. costruzione soluzioni delle formiche
4. selezione `best_epoch`
5. aggiornamento `global_best`
6. evaporazione + deposito
7. criteri di stop (tempo/stagnazione)

### 1.0.2 Limitations

Esempi coerenti con il vostro caso:
- Complessità ~ `O(iter * m * N * cand_k)` con fallback globali
- Memoria `O(N^2)` per matrice feromoni (soprattutto CUDA)
- Sensibilità a `m`, `rho`, timeout, stagnation, improvement threshold

### 1.1 Sequential Code Analysis

Riferimenti principali da citare:
- Loop principale ACO seq: `src/seq/aco_sequential.c` (funzione `aco_vrp_run_with_timer`)
- Costruzione soluzione formica: `build_ant_solution`
- Aggiornamento feromoni (`best_epoch` + `global_best`): `iter_deposit_weight=0.3`, `global_deposit_weight=0.7`

## 2 MPI + OpenMP Implementation

### 2.1 Workload Distribution

Descrivi:
- partizionamento formiche per rank (`local_m`, `ant_offset`)
- parallelismo intra-rank con OpenMP

### 2.2 Fully Local Operations

Snippet suggeriti:
- ciclo di costruzione e valutazione locale ant
- riduzione thread-local -> `iter_best` locale

### 2.3 Ranks Synchronization

Snippet/concetti:
- `MPI_Allreduce` su costo migliore iterazione
- selezione `global_best_ant`
- broadcast soluzione vincente (`MPI_Bcast` + pack/unpack)

### 2.4 Parallel Update Step

Descrivi:
- evaporazione parallela OpenMP
- deposito su `iter_best` e `global_best`
- clamp `tau_min/tau_max`

### 2.5 Outside Iterative Loop

Descrivi:
- inizializzazione matrici/strutture condivise
- cleanup, gestione errori allocazione, stop conditions

### 2.6 MPI + OpenMP Performance Analysis

Sottosezioni stile PDF:
- 2.6.1 Speedup analysis
- 2.6.2 Execution time across datasets
- 2.6.3 Parallel efficiency
- 2.6.4 Scaling quality

Input dati:
- `results/solve_manifest/csv/manifest_seq_per_instance_results.csv`
- `results/solve_manifest/csv/manifest_openmp_mpi_per_instance_results.csv`

## 3 CUDA Implementation

Usa la stessa struttura del PDF K-means, ma adattata al vostro ACO.

### 3.1 Memory Model and Parameters

Da citare:
- v5: matrice `tau` float (`src/cuda/aco_cuda_v5.cu`)
- v6: tau quantizzata `uint8` log-space (`src/cuda/aco_cuda_v6.cu`, `src/cuda/aco_cuda_v6_kernels.cu`)

### 3.2 Kernel Pipeline

#### 3.2.1 Kernel A: reset stato formiche
- `launch_reset_ant_state_v5/v6`

#### 3.2.2 Kernel B: costruzione soluzioni
- `launch_construct_solutions_v5/v6`

#### 3.2.3 Kernel C: evaporazione + deposito
- `launch_evaporate_tau_v5/v6`
- `launch_deposit_solution_v5/v6`

### 3.3 Host-Orchestrated Iteration

Da mettere in evidenza:
- selezione best iteration lato host (`select_iter_best_host`)
- update global best e telemetria (v5)
- stop per timeout/stagnazione

### 3.4 CUDA Performance Analysis

Sottosezioni stile PDF:
- 3.4.1 Speedup across datasets
- 3.4.2 Execution time across datasets
- 3.4.3 Efficiency (vs baseline seq)
- 3.4.4 Scaling and memory wall

Input dati:
- `results/solve_manifest/csv/manifest_cuda_<variant>_per_instance_results.csv`
- `docs/cuda_v6_scaling_report.md` (sezione narrativa)

## 4 Overall Performance

### 4.1 Smallest vs Largest Instances
- tempo assoluto
- speedup vs seq
- efficienza relativa

### 4.2 Overall Comparison Across Implementations
- best speedup per backend
- robustezza su diverse `n`
- qualità scaling

## 5 Conclusions

Punti consigliati:
- quando conviene seq
- quando conviene openmp-mpi
- quando conviene cuda (v5/v6)
- limiti attuali e next steps

## References

Includi:
- repo e file sorgente
- eventuali paper ACO/CVRP
- tool di profiling usati

---

## Appendix A: Snippets to Extract (Ready-to-Cite)

Estrarre blocchi da:
- `src/seq/aco_sequential.c`
- `src/openmp-mpi/aco_parallel.c`
- `src/cuda/aco_cuda_v5.cu`
- `src/cuda/aco_cuda_v6.cu`
- `src/cuda/aco_cuda_v5_kernels.cu`
- `src/cuda/aco_cuda_v6_kernels.cu`

Ordine consigliato snippet:
1. loop iterativo solver
2. costruzione soluzione ant
3. selezione best epoch
4. update global best
5. evaporazione/deposito
6. stop conditions

