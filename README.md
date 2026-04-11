# VehicleRoutingProblem

Repository per risolvere istanze VRP con ACO usando 4 backend:
- `pyvrp` (Python, riferimento)
- `seq` (C sequenziale)
- `openmp-mpi` (C + OpenMP/MPI)
- `cuda` (CUDA)

Questa guida descrive lo stato attuale della repo, inclusa la nuova organizzazione completa di `tools/`.

## Requisiti
- `make`, `gcc`
- `mpicc`, `mpirun` (solo backend MPI)
- `nvcc` (solo backend CUDA)
- `python3`
- ambiente Python con `pyvrp`

Nota: se esiste `VRP/bin/python`, i flussi `solve_pyvrp` lo usano automaticamente.

## Struttura repo (stato corrente)

Codice solver:
- `src/seq/`
- `src/openmp-mpi/`
- `src/cuda/`
- `src/common/`

Header:
- `include/`

Script analisi/benchmark storici:
- `scripts/` (solo benchmark/plot/profiling, non più utility make principali)

Tooling operativo centralizzato:
- `tools/makefile/` -> moduli `.mk` inclusi dal `Makefile`
- `tools/bash/` -> script bash operativi (`solve_*`, `run_and_validate.sh`)
- `tools/python/` -> utility Python operative (`generate_vrp_problem.py`, `solve_pyvrp_runner.py`, `validate_pyvrp.py`)
- `tools/batch/` -> submit/job script Slurm per eseguire i target `solve_*` su cluster

Entry-point build:
- `Makefile` (root) include i moduli da `tools/makefile/*.mk`

## Makefile: organizzazione modulare

Il `Makefile` root include:
- `tools/makefile/vars.mk`
- `tools/makefile/help.mk`
- `tools/makefile/build.mk`
- `tools/makefile/generate.mk`
- `tools/makefile/solve.mk`
- `tools/makefile/experiments.mk`
- `tools/makefile/phony.mk`

Default goal:
- `.DEFAULT_GOAL := all`

## Generazione istanze

Comando base:

```bash
make generate_problems
```

Output:
- `instances/test_aligned/*.vrp`
- `instances/test_aligned/manifest.csv`
- `instances/test_aligned/manifest_openmp_mpi.csv`
- `instances/test_aligned/manifest_cuda.csv`

Nota:
- `generate_problems` ripulisce prima i `.vrp`/manifest esistenti nella `GEN_INST_DIR`, poi rigenera tutto da zero.
- i manifest generati vengono ordinati automaticamente per `n` crescente (poi `instance_seed`).

Profilo grande predefinito:

```bash
make generate_problems_big
```

Pulizia:

```bash
make generate_clean
```

Variabili principali:
- `GEN_INST_DIR`
- `GEN_CLIENTS`
- `GEN_SEED_BASE`
- `GEN_GRID`
- `GEN_SOLVER_SEED`
- `GEN_TARGET_CUSTOMERS_PER_VEHICLE`
- `GEN_MIN_VEHICLES`
- `GEN_MAX_VEHICLES`
- `GEN_CUDA_M` (default `256`, usato nel manifest CUDA)

Implementazione usata da make:
- `tools/python/generate_vrp_problem.py`

## Solve da manifest

Target:

```bash
make solve_pyvrp
make solve_seq
make solve_cuda
make solve_mpi
make solve_all
```

Prepare/validazione path:
- `solve_prepare` crea cartelle output `solve_*` e verifica `SOLVE_MANIFEST` / `SOLVE_MANIFEST_MPI` / `SOLVE_MANIFEST_CUDA`
- non crea directory `scaling`

### Ripetizioni per istanza
Supportate direttamente:
- `SOLVE_SEQ_REPEATS`
- `SOLVE_MPI_REPEATS`
- `SOLVE_CUDA_REPEATS`

Ogni run scrive `run_id` nel CSV e route file separato (`*_runN_solution.txt`).

### Memoria non-CUDA
Per `seq` e `mpi` viene registrato `max_rss_kb` (via `/usr/bin/time`).

### Variabili solve principali
- `SOLVE_OUT_DIR`
- `SOLVE_CSV_DIR`
- `SOLVE_SOLUTIONS_DIR`
- `SOLVE_MANIFEST`
- `SOLVE_MANIFEST_MPI`
- `SOLVE_MANIFEST_CUDA`
- `SOLVE_CLIENTS`
- `SOLVE_LIMIT`
- `SOLVE_PYVRP_RUNTIME_S`
- `SOLVE_PYVRP_SEED`
- `SOLVE_SEQ_RUNTIME_S`
- `SOLVE_SEQ_RUNTIME` (alias)
- `SOLVE_SEQ_M`
- `SOLVE_SEQ_STAGNATION_EPOCHS`
- `SOLVE_SEQ_MIN_REL_IMPROVEMENT` (percentuale; es. `0.1` = `0.1%`)
- `SOLVE_MPI_RUNTIME_S`
- `SOLVE_MPI_STAGNATION_EPOCHS`
- `SOLVE_MPI_MIN_REL_IMPROVEMENT` (percentuale; es. `0.1` = `0.1%`)
- `SOLVE_MPI_RANKS`
- `SOLVE_MPI_OMP_THREADS`
- `SOLVE_MPI_LAUNCHER` (`auto|mpirun|srun`)
- `SOLVE_CUDA_VARIANT`

Nota mapping env nel solver C:
- i target passano `SOLVE_*_STAGNATION_EPOCHS` come `ACO_SOLVER_STAGNATION_ITERS`
- i target passano `SOLVE_*_MIN_REL_IMPROVEMENT` come `ACO_SOLVER_IMPROVE_EPS`
- per `seq` e `mpi`, il valore da Makefile viene convertito da percentuale a frazione (`val/100`)

### Comando composito crescita memoria non-CUDA

```bash
make solve_memory_growth_non_cuda
```

Usa un profilo clienti grande (default `4000..32000`) e ripetizioni con `SOLVE_MEMORY_GROWTH_REPEATS`.

## Esperimenti scaling (`exp_*`)

I target `exp_*` sono wrapper sopra `make solve_mpi` pensati per campagne di benchmark riproducibili.
Ogni target scrive in una directory CSV dedicata (append-only), quindi i risultati non si mescolano con i CSV generali.

Target disponibili:
- `make exp_strong_openmp`
  - strong scaling OpenMP intra-node
  - mantiene `n` fisso (`EXP_STRONG_OPENMP_N`, default `2000`)
  - usa `MPI_RANKS=1`, varia `OMP_THREADS` (`EXP_STRONG_OPENMP_THREADS`, default `1 2 4 8 16`)
- `make exp_weak_openmp`
  - weak scaling OpenMP intra-node
  - usa coppie `(threads, n)` da `EXP_WEAK_OPENMP_PAIRS` (default `1 2000 2 4000 4 8000 8 16000 16 32000`)
  - mantiene `MPI_RANKS=1`
- `make exp_strong_mpi`
  - strong scaling MPI inter-node
  - mantiene `n` fisso (`EXP_STRONG_MPI_N`, default `16000`)
  - varia `MPI_RANKS` (`EXP_STRONG_MPI_RANKS`, default `1 2 4 8`)
  - mantiene `OMP_THREADS` fisso (`EXP_STRONG_MPI_OMP_THREADS`, default `8`)
- `make exp_strong_hybrid`
  - strong scaling ibrido OpenMP+MPI
  - mantiene `n` fisso (`EXP_STRONG_HYBRID_N`, default `16000`)
  - varia coppie `(ranks, threads)` con `EXP_STRONG_HYBRID_PAIRS` (default `auto`, scala proporzionalmente con `r=t` in potenze di 2)
- `make exp_weak_mpi`
  - weak scaling MPI
  - usa coppie `(ranks, n)` da `EXP_WEAK_MPI_PAIRS` (default `auto`, `n` proporzionale ai rank)
  - mantiene `OMP_THREADS` fisso (`EXP_WEAK_MPI_OMP_THREADS`, default `8`)
- `make exp_weak_hybrid`
  - weak scaling ibrido OpenMP+MPI
  - usa triple `(ranks, threads, n)` da `EXP_WEAK_HYBRID_TRIPLETS` (default `auto`, con `r=t` e `n` proporzionale a `r*t`)
- `make exp_all`
  - esegue in sequenza tutti i target sopra.

Impostazioni comuni ai target `exp_*`:
- `EXP_REPEATS` (default `5`): ripetizioni per punto sperimentale
- `EXP_STAGNATION_EPOCHS` (default `500`)
- `EXP_MIN_REL_IMPROVEMENT` (default `0.001`)
- `EXP_MPI_LAUNCHER` (default `mpirun`)
- `EXP_MAX_CLUSTER_NODES` (default `4`)
- `EXP_WEAK_BASE_N_PER_WORKER` (default `2000`)
- timeout disattivato (`SOLVE_MPI_RUNTIME_S=0`) per non troncare i run su tempo.

Output CSV separati:
- `results/solve_manifest/csv/exp_strong_openmp/manifest_openmp_mpi_per_instance_results.csv`
- `results/solve_manifest/csv/exp_weak_openmp/manifest_openmp_mpi_per_instance_results.csv`
- `results/solve_manifest/csv/exp_strong_mpi/manifest_openmp_mpi_per_instance_results.csv`
- `results/solve_manifest/csv/exp_strong_hybrid/manifest_openmp_mpi_per_instance_results.csv`
- `results/solve_manifest/csv/exp_weak_mpi/manifest_openmp_mpi_per_instance_results.csv`
- `results/solve_manifest/csv/exp_weak_hybrid/manifest_openmp_mpi_per_instance_results.csv`

Ogni riga nel CSV MPI include anche:
- `mpi_ranks`
- `omp_threads`
- `batch_id` (identificatore del batch lanciato)

Esempi:

```bash
make exp_strong_openmp
make exp_strong_hybrid EXP_STRONG_HYBRID_PAIRS="1 1 1 2 2 2 2 4 4 4"
make exp_weak_mpi EXP_WEAK_MPI_PAIRS="1 2000 2 4000 4 8000" EXP_REPEATS=3
make exp_weak_hybrid
```

## Output risultati

Base:
- `RESULTS_ROOT` (default `results`)
- `SOLVE_OUT_DIR` (default `results/solve_manifest`)

CSV principali:
- `results/solve_manifest/csv/manifest_pyvrp_per_instance_results.csv`
- `results/solve_manifest/csv/manifest_seq_per_instance_results.csv`
- `results/solve_manifest/csv/manifest_cuda_<variant>_per_instance_results.csv`
- `results/solve_manifest/csv/manifest_openmp_mpi_per_instance_results.csv`

Route:
- `results/solve_manifest/solutions/pyvrp/*.txt`
- `results/solve_manifest/solutions/seq/*.txt`
- `results/solve_manifest/solutions/cuda_<variant>/*.txt`
- `results/solve_manifest/solutions/mpi/*.txt`

## Tool bash utili

- `tools/bash/run_and_validate.sh`
- `tools/bash/solve_pyvrp.sh`
- `tools/bash/solve_seq.sh`
- `tools/bash/solve_cuda.sh`
- `tools/bash/solve_mpi.sh`

## Tool python utili

- `tools/python/generate_vrp_problem.py`
- `tools/python/solve_pyvrp_runner.py`
- `tools/python/validate_pyvrp.py`

## Esecuzione su cluster (Slurm)

Script disponibili:
- `tools/batch/submit_solve.sh`
- `tools/batch/run_solve.sbatch`

Note cluster:
- il job inizializza automaticamente l'ambiente con `source /home/guest/init-hpc.sh` (se presente)
- QoS di default nel job: `students_limit` (override con `--qos`)
- default submit per target:
  - `solve_seq` (e altri non-MPI): `--nodes=1 --ntasks=1 --cpus-per-task=32`
  - `solve_mpi`: `--nodes=4 --ntasks=4 --cpus-per-task=32`
- launcher MPI in `solve_mpi`: configurabile con `SOLVE_MPI_LAUNCHER`; per il tuo workflow usa `mpirun`
- per CUDA sul tuo cluster usa `CUDA_ARCH=sm_75` nei `--make-args`

Esempi:

```bash
tools/batch/submit_solve.sh --target solve_seq \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=60"
```

```bash
tools/batch/submit_solve.sh --target solve_mpi --cpus 32 --mem 64G \
  --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun SOLVE_MPI_REPEATS=3"
```

```bash
tools/batch/submit_solve.sh --target solve_cuda --partition gpu --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_VARIANT=v4 CUDA_ARCH=sm_75"
```

Caricamento moduli cluster (opzionale):

```bash
tools/batch/submit_solve.sh --target solve_all \
  --module-loads "gcc/13.2 openmpi/4.1 cuda/12.2"
```

## Comandi rapidi

### Locale

Build:

```bash
make all
make openmp_mpi
make cuda
```

Solve all (filtro clienti):

```bash
make solve_all SOLVE_CLIENTS=500,1000
```

Solve seq con ripetizioni:

```bash
make solve_seq SOLVE_CLIENTS=4000 SOLVE_SEQ_REPEATS=3
```

Solve mpi con parametri runtime:

```bash
make solve_mpi SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_RUNTIME_S=20
```

Help completo:

```bash
make help
```

Pulizia:

```bash
make clean
```

PyVRP in locale:

```bash
make solve_pyvrp
```

```bash
make solve_pyvrp SOLVE_CLIENTS=500,1000 SOLVE_PYVRP_RUNTIME_S=30
```

### Locale vs Cluster (comandi pronti)

Locale, con progressione `seq -> mpi -> cuda`:

```bash
make solve_seq SOLVE_CLIENTS=4000,8000,16000,32000,64000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 && \
make solve_mpi SOLVE_CLIENTS=4000,8000,16000,32000,64000 SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=4 && \
make solve_cuda SOLVE_CLIENTS=4000,8000,16000,32000,64000 SOLVE_CUDA_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300
```

Locale, solo sequenziale su tutte le taglie:

```bash
make solve_seq SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300
```

Locale, verifica risorse CPU disponibili (per scegliere `SOLVE_MPI_RANKS` e `SOLVE_MPI_OMP_THREADS`):

```bash
nproc
lscpu | egrep 'CPU\(s\)|Core\(s\) per socket|Socket\(s\)|Thread\(s\) per core'
```

Locale, impostazione automatica `MPI_RANKS`/`OMP_THREADS` da CPU locali:

```bash
cores=$(lscpu -p=Core | grep -v '^#' | sort -u | wc -l); \
threads=$(( $(nproc) / cores )); \
make solve_mpi SOLVE_CLIENTS=4000,8000,16000,32000,64000 SOLVE_MPI_RANKS=$cores SOLVE_MPI_OMP_THREADS=$threads
```

Cluster (Slurm), sequenziale:

```bash
tools/batch/submit_solve.sh --target solve_seq \
  --make-args "SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300"
```

Cluster (Slurm), MPI:

```bash
tools/batch/submit_solve.sh --target solve_mpi \
  --make-args "SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun"
```

Cluster (Slurm), CUDA:

```bash
tools/batch/submit_solve.sh --target solve_cuda --partition gpu --gres gpu:1 \
  --make-args "SOLVE_CUDA_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_CUDA_VARIANT=v6 CUDA_ARCH=sm_75"
```

Cluster, esecuzione in coda `seq -> mpi -> cuda`:

```bash
tools/batch/submit_solve.sh --target solve_seq \
  --make-args "SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300" && \
tools/batch/submit_solve.sh --target solve_mpi \
  --make-args "SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun" && \
tools/batch/submit_solve.sh --target solve_cuda --partition gpu --gres gpu:1 \
  --make-args "SOLVE_CUDA_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_CUDA_VARIANT=v6 CUDA_ARCH=sm_75"
```

### Cluster (Slurm)

Submit PyVRP:

```bash
tools/batch/submit_solve.sh --target solve_pyvrp \
  --qos students_limit \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_PYVRP_RUNTIME_S=30"
```

Submit sequenziale:

```bash
tools/batch/submit_solve.sh --target solve_seq \
  --qos students_limit \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=30"
```

Submit MPI:

```bash
tools/batch/submit_solve.sh --target solve_mpi \
  --qos students_limit --cpus 32 \
  --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=mpirun SOLVE_MPI_RUNTIME_S=30"
```

Submit CUDA:

```bash
tools/batch/submit_solve.sh --target solve_cuda \
  --partition gpu --gres gpu:1 --qos students_limit \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_VARIANT=v4 CUDA_ARCH=sm_75"
```

Log job cluster:
- `results/slurm/*.out`
- `results/slurm/*.err`
