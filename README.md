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
- `solve_prepare` crea cartelle output `solve_*` e verifica `SOLVE_MANIFEST` / `SOLVE_MANIFEST_MPI`
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
  - `solve_mpi`: `--nodes=4 --ntasks=4 --cpus-per-task=8`
- per CUDA sul tuo cluster usa `CUDA_ARCH=sm_75` nei `--make-args`

Esempi:

```bash
tools/batch/submit_solve.sh --target solve_seq \
  --make-args "SOLVE_CLIENTS=500,1000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=60"
```

```bash
tools/batch/submit_solve.sh --target solve_mpi --cpus 32 --mem 64G \
  --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_REPEATS=3"
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
  --make-args "SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_RUNTIME_S=30"
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
