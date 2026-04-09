# VehicleRoutingProblem

Repository per risolvere istanze VRP con ACO usando 4 backend:
- `pyvrp` (Python, riferimento)
- `seq` (C sequenziale)
- `openmp-mpi` (C + MPI/OpenMP)
- `cuda` (CUDA)

Questa guida è operativa: copia/incolla i comandi e ottieni CSV + route.

## Requisiti minimi
- `make`, `gcc`
- `mpicc`, `mpirun` (solo MPI)
- `nvcc` (solo CUDA)
- `python3`
- ambiente Python con `pyvrp` (se presente `VRP/bin/python`, il Makefile lo usa automaticamente)

## 1) Generare i problemi (dal Makefile)
Default:

```bash
make generate_problems
```

Questo comando crea:
- `instances/test_aligned/*.vrp`
- `instances/test_aligned/manifest.csv`
- `instances/test_aligned/manifest_openmp_mpi.csv`

Esempi utili:

```bash
# taglie custom
make generate_problems GEN_CLIENTS=500,1000,4000,8000

# capacità "illimitata" (usa --unlimited del generatore)
make generate_problems GEN_CAPACITY_MODE=unlimited

# output in una directory diversa
make generate_problems GEN_INST_DIR=instances/my_batch GEN_SEED_BASE=25000
```

Pulizia file generati:

```bash
make generate_clean
```

Variabili generazione:
- `GEN_INST_DIR=instances/test_aligned`
- `GEN_CLIENTS=500,1000,2000,4000,8000,16000`
- `GEN_SEED_BASE=19000`
- `GEN_GRID=100`
- `GEN_SOLVER_SEED=1234`
- `GEN_CAPACITY_MODE=formula|unlimited`

Colonne principali del manifest:
- `n` = numero clienti
- `K` = numero veicoli
- `m` = formiche
- `T` = iterazioni
- `solver_seed` = seed solver

## 2) Risolvere tutto (quick start)
Esegui tutti i backend in una volta:

```bash
make solve_all
```

Questo comando:
1. legge i manifest
2. lancia `pyvrp`, `seq`, `cuda`, `mpi`
3. salva CSV risultati
4. salva le route complete per ogni istanza/backend

## 3) Eseguire solo una parte (filtro per clienti)
Esempi:

```bash
# solo istanze con n=500 e n=1000
make solve_all SOLVE_CLIENTS=500,1000

# solo n=4000, con runtime pyvrp più alto
make solve_pyvrp SOLVE_CLIENTS=4000 SOLVE_PYVRP_RUNTIME_S=30

# seq con budget 20s + stop per stagnazione in epoche (80 iterazioni senza miglioramento)
make solve_seq SOLVE_CLIENTS=500,1000 SOLVE_SEQ_RUNTIME_S=20 SOLVE_SEQ_T=500 SOLVE_SEQ_STAGNATION_ITERS=80

# mpi con budget 20s + stop per stagnazione in epoche
make solve_mpi SOLVE_CLIENTS=4000,8000 SOLVE_MPI_RUNTIME_S=20 SOLVE_MPI_STAGNATION_ITERS=80

# cuda
make solve_cuda SOLVE_CLIENTS=500,1000

# prime 3 istanze dopo il filtro
make solve_seq SOLVE_CLIENTS=500,1000,2000 SOLVE_LIMIT=3
```

## Dove finiscono i risultati
Output base:
- `RESULTS_ROOT` (default `results`)
- `SOLVE_OUT_DIR` (default `results/solve_manifest`)
- `SOLVE_CSV_DIR` (default `results/solve_manifest/csv`)
- `SOLVE_SOLUTIONS_DIR` (default `results/solve_manifest/solutions`)

CSV:
- `$(SOLVE_CSV_DIR)/manifest_pyvrp_per_instance_results.csv`
- `$(SOLVE_CSV_DIR)/manifest_seq_per_instance_results.csv`
- `$(SOLVE_CSV_DIR)/manifest_cuda_per_instance_results.csv`
- `$(SOLVE_CSV_DIR)/manifest_openmp_mpi_per_instance_results.csv`

Route complete:
- `$(SOLVE_SOLUTIONS_DIR)/pyvrp/*.txt`
- `$(SOLVE_SOLUTIONS_DIR)/seq/*.txt`
- `$(SOLVE_SOLUTIONS_DIR)/cuda/*.txt`
- `$(SOLVE_SOLUTIONS_DIR)/mpi/*.txt`

Note:
- nel CSV PyVRP c’è anche `max_rss_kb` (picco memoria RSS in KB per istanza)
- se MPI fallisce per ambiente (PMIx/OpenMPI), vedrai `status=error` nel CSV MPI

## Target principali Makefile
Generazione istanze:

```bash
make generate_problems
make generate_clean
```

Build:

```bash
make all          # seq
make openmp_mpi   # mpi+openmp
make cuda         # cuda
```

Solve da manifest:

```bash
make solve_pyvrp
make solve_seq
make solve_cuda
make solve_mpi
make solve_all
```

Aiuto completo:

```bash
make help
```

## Variabili utili (solve_*)
- `SOLVE_OUT_DIR=...` cartella output
- `SOLVE_CSV_DIR=...` cartella CSV
- `SOLVE_SOLUTIONS_DIR=...` cartella route
- `SOLVE_MANIFEST=...` manifest per `solve_pyvrp/solve_seq/solve_cuda`
- `SOLVE_MANIFEST_MPI=...` manifest per `solve_mpi`
- `SOLVE_CLIENTS=500,1000,...` filtro per `n`
- `SOLVE_LIMIT=N` limita il numero di righe (0 = tutte)
- `SOLVE_PYVRP_RUNTIME_S=10`
- `SOLVE_SEQ_RUNTIME_S=0` (0 = disattivo)
- `SOLVE_SEQ_RUNTIME=60` (alias compatibile)
- `SOLVE_SEQ_STAGNATION_ITERS=0` (0 = off, stop per epoche senza miglioramento)
- `SOLVE_SEQ_M=0` (override `m` solo per `solve_seq`)
- `SOLVE_SEQ_T=0` (override `T` solo per `solve_seq`)
- `SOLVE_MPI_RUNTIME_S=0` (0 = disattivo)
- `SOLVE_MPI_STAGNATION_ITERS=0` (0 = off, stop per epoche senza miglioramento)
- `SOLVE_PYVRP_SEED=1234`
- `SOLVE_MPI_RANKS=2`
- `SOLVE_MPI_OMP_THREADS=2`
- `SOLVE_CUDA_IMPROVEMENT=0.001`

Esempio completo:

```bash
make solve_all \
  SOLVE_OUT_DIR=results/run_01 \
  SOLVE_MANIFEST=instances/test_aligned/manifest.csv \
  SOLVE_MANIFEST_MPI=instances/test_aligned/manifest_openmp_mpi.csv \
  SOLVE_CLIENTS=500,1000,2000 \
  SOLVE_PYVRP_RUNTIME_S=20 \
  SOLVE_MPI_RANKS=4 \
  SOLVE_MPI_OMP_THREADS=8
```

## Modalità test legacy (scaling)
Restano disponibili i target storici:
- `make sequential_tests`
- `make openmp_mpi_tests`
- `make openmp_mpi_tests_heavy`

Output legacy scaling:
- `$(RESULTS_ROOT)/scaling/scaling_progressive_c.csv`
- `$(RESULTS_ROOT)/scaling/scaling_progressive_openmp_mpi_light.csv`
- `$(RESULTS_ROOT)/scaling/scaling_progressive_openmp_mpi_heavy.csv`
- checkpoint MPI in `$(RESULTS_ROOT)/scaling/`
- log detached in `$(RESULTS_ROOT)/detached/`

Con supporto detached:
- `TEST_MODE=background` (default)
- `TEST_MODE=foreground`
- `TEST_LOGS_DIR=...`

## Pulizia
```bash
make clean
```

Rimuove binari, oggetti, coverage e artefatti CUDA intermedi (`.ptx`, `.cubin`, `.fatbin`).
