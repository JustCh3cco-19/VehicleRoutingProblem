# VehicleRoutingProblem

Implementazione in C di Ant Colony Optimization (ACO) per Vehicle Routing Problem (VRP), con due eseguibili:
- `aco_vrp_seq.out` (sequenziale)
- `aco_vrp_openmp_mpi.out` (ibrido OpenMP + MPI)

Il solver usa layout hardware-aware (matrici allineate 64-byte, candidate list per-rank, workspace persistente per-thread/per-run, MMAS bounds, sincronizzazione MPI per epoche).

## Requisiti
- `gcc`
- `make`
- `mpicc` e `mpirun` (solo target MPI/OpenMP)
- `python3` + `pandas`, `numpy`, `matplotlib` (report scaling)
- `pyvrp` (solo confronto C vs PyVRP)
- `sbatch` (opzionale, per esecuzione su cluster SLURM)

## Struttura progetto
- `src/seq/aco_sequential.c`: solver ACO sequenziale
- `src/openmp-mpi/aco_parallel.c`: solver ACO OpenMP+MPI
- `src/common/matrix.c`: allocazione matrici dense allineate/paddate
- `src/common/solution.c`: storage contiguo flat delle soluzioni
- `tests/sequential_tests.c`: scaling progressivo C
- `tests/openmp_mpi_tests.c`: scaling OpenMP+MPI light (fino a 16k clienti)
- `tests/openmp_mpi_tests_heavy.c`: scaling OpenMP+MPI heavy (da 24k a 100k clienti)
- `tests/openmp_mpi_scaling_report.py`: report/plot da CSV MPI
- `tests/c_compare_case.c`: runner C single-scenario per confronto con PyVRP
- `tests/pyvrp_compare.py`: confronto C vs PyVRP con gap su CSV
- `scripts/run_openmp_mpi_tests.sbatch`: job batch SLURM

## Build
Sequenziale:
```sh
make
```

Ibrido OpenMP+MPI:
```sh
make openmp_mpi
```

Help Makefile:
```sh
make help
```

## Esecuzione binari solver
### Sequenziale
```sh
./aco_vrp_seq.out [n K m T seed]
```

Esempio:
```sh
./aco_vrp_seq.out 200 16 0 100 1234
```

Note:
- `m=0` abilita auto-scaling delle formiche.
- senza argomenti, usa un esempio minimale interno.

### OpenMP + MPI
```sh
OMP_NUM_THREADS=2 mpirun -np 2 ./aco_vrp_openmp_mpi.out [n K m T seed]
```

## Test mode (foreground/background)
I target `sequential_tests`, `openmp_mpi_tests` e `openmp_mpi_tests_heavy` usano:
- `TEST_MODE=background` (default): run detached con `nohup`
- `TEST_MODE=foreground`: output live nel terminale
- `TEST_LOGS_DIR=...`: directory log detached (`.log`, `.pid`, `.cmd`)

Esempi:
```sh
make sequential_tests TEST_MODE=foreground
make openmp_mpi_tests TEST_MODE=foreground MPI_NP=4 MPI_OMP_THREADS=8
make openmp_mpi_tests_heavy TEST_MODE=foreground MPI_NP=2 MPI_OMP_THREADS=32
make sequential_tests TEST_LOGS_DIR=/tmp/vrp_logs
```

Output runtime semplificato (esempio):
- `[RUN xx/yy] ...`
- `[INPUT] solver : n=... K=... m=... T=... alpha=... beta=... rho=... tau0=... Q=... seed=...`
- `[INPUT] instance : seed=... layout_id=...`
- `[OK] ...` oppure `[SKIP] ...` oppure `[FAIL] ...`

## Sequential scaling runner
Build + run:
```sh
make sequential_tests
```

CSV default:
- `results/scaling_progressive_c.csv`

Esecuzione manuale:
```sh
./tests/sequential_tests.out --memory-utilization 0.70 --c-max-n 100000
```

Opzioni CLI:
- `--csv PATH`
- `--input-log PATH` (default: `results/sequential_test_inputs.log`)
- `--memory-utilization X` con `0 < X <= 1`
- `--c-max-n N`
- `--force` (ignora soglia memoria stimata)
- `--auto-ants` (default, passa `m=0` al solver)
- `--fixed-ants` (usa `m` dallo scenario)

Nota memoria: il runner stima `c` + `tau` dense e matrici candidate (`candidate_idx`, `eta_beta`, `score`) con overhead di sicurezza.

## OpenMP+MPI scaling runner
Build + run via Make (light <= 16k):
```sh
make openmp_mpi_tests
```

Build + run via Make (heavy > 16k):
```sh
make openmp_mpi_tests_heavy
```

Il target usa:
- `MPI_NP` (default `2`)
- `MPI_OMP_THREADS` (default `2`)

Comando equivalente:
```sh
OMP_NUM_THREADS=${MPI_OMP_THREADS} mpirun -np ${MPI_NP} ./tests/openmp_mpi_tests.out
```

CSV default:
- `results/scaling_progressive_openmp_mpi_light.csv`
- `results/scaling_progressive_openmp_mpi_heavy.csv`

Esecuzione manuale:
```sh
mpirun -np 2 ./tests/openmp_mpi_tests.out --memory-utilization 0.70 --c-max-n 100000
mpirun -np 2 ./tests/openmp_mpi_tests_heavy.out --memory-utilization 0.70 --c-max-n 100000
```

Opzioni CLI:
- `--csv PATH`
- `--input-log PATH` (default: `results/openmp_mpi_test_inputs.log`)
- `--checkpoint PATH` (default: `results/openmp_mpi_tests_light.checkpoint` o `..._heavy.checkpoint`)
- `--resume` (riparte da checkpoint, append a CSV/log)
- `--memory-utilization X` con `0 < X <= 1`
- `--time-budget-minutes X` (default `30`)
- `--estimate-safety X` (default `1.25`)
- `--c-max-n N`
- `--enforce-c-max-n` (salta `n > c_max_n`)
- `--force` (ignora soglia memoria stimata)
- `--auto-ants` (default, passa `m=0` al solver)
- `--fixed-ants` (usa `m` dallo scenario)

Durante i test, il runner stampa esplicitamente gli input passati al solver per ogni run (`n,K,m,T,alpha,beta,rho,tau0,Q,seed`) e salva gli stessi dati nel file `--input-log`.

## Esecuzione SLURM
Submit diretto:
```sh
./scripts/submit_openmp_mpi_tests.sh
```

Con argomenti passati al test runner:
```sh
./scripts/submit_openmp_mpi_tests.sh "--memory-utilization 0.70 --c-max-n 100000 --enforce-c-max-n"
```

Profilo heavy:
```sh
./scripts/submit_openmp_mpi_tests.sh "--memory-utilization 0.70 --c-max-n 100000 --enforce-c-max-n" heavy
```

Con checkpoint mode esplicito (`fresh|resume|reset`):
```sh
./scripts/submit_openmp_mpi_tests.sh "--time-budget-minutes 30" heavy resume
./scripts/submit_openmp_mpi_tests.sh "--time-budget-minutes 30" heavy reset
```

Run pulita e ripresa dopo blocco:
```sh
# 1) run da zero (pulisce checkpoint precedente)
./scripts/submit_openmp_mpi_tests.sh "--time-budget-minutes 30 --memory-utilization 0.70 --estimate-safety 1.25" heavy reset

# 2) se il job viene interrotto/scade, riprende dal checkpoint
./scripts/submit_openmp_mpi_tests.sh "--time-budget-minutes 30 --memory-utilization 0.70 --estimate-safety 1.25" heavy resume
```

Target Make utili:
```sh
make openmp_mpi_tests_resume TEST_MODE=foreground
make openmp_mpi_tests_reset TEST_MODE=foreground
make openmp_mpi_tests_heavy_resume TEST_MODE=foreground
make openmp_mpi_tests_heavy_reset TEST_MODE=foreground
```

Lo script batch:
- compila `tests/openmp_mpi_tests.out` (o `tests/openmp_mpi_tests_heavy.out` con `TEST_PROFILE=heavy`)
- imposta `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`
- esegue `mpirun -n $SLURM_NTASKS ...`
- salva output/error in `results/slurm/`

## Report scaling OpenMP+MPI
Da CSV a report markdown + grafici:
```sh
python3 tests/openmp_mpi_scaling_report.py \
  --csv results/scaling_progressive_openmp_mpi.csv \
  --out-dir results
```

Output principali:
- `results/openmp_mpi_scaling_report.md`
- `results/openmp_mpi_runtime_vs_n.png`
- `results/openmp_mpi_throughput_vs_n.png`
- `results/openmp_mpi_empirical_exponent.png`
- `results/openmp_mpi_memory_vs_n.png`

## Confronto C vs PyVRP
Build helper C:
```sh
make c_compare_case
make c_compare_case_mpi
```

Confronto rapido su un solo scenario (solo sequenziale):
```sh
python3 tests/pyvrp_compare.py --mode seq --single-n 500 --enforce-c-max-n --c-max-n 500
```

Confronto sequenziale + parallelo (MPI/OpenMP):
```sh
python3 tests/pyvrp_compare.py --mode both --single-n 500 --mpi-ranks-list 1,2,4 --omp-threads 2 --enforce-c-max-n --c-max-n 500
```

Nota: lo script prova automaticamente a rieseguirsi con `VRP/bin/python` se presente.

Output CSV default:
- `results/c_vs_pyvrp_compare.csv`
- `results/c_vs_pyvrp_scaling_report.md`

Opzioni utili:
- `--mode seq|mpi|both` (default: `both`)
- `--mpi-ranks-list 1,2,4` (usato in modalità MPI)
- `--omp-threads N` (thread OpenMP per rank in modalità MPI)
- `--fixed-ants` (default usa `m=0` auto)
- `--timeout N` (secondi, stop PyVRP a runtime; `0` usa iterazioni `T`)
- `--helper-bin-seq PATH` (default `tests/c_compare_case.out`)
- `--helper-bin-mpi PATH` (default `tests/c_compare_case_mpi.out`)
- `--no-build-helpers` (non lancia `make` automatico)

Metriche scaling nel CSV/report:
- **Strong scaling**: `strong_speedup`, `strong_efficiency` (stesso `n`, rank diversi)
- **Weak scaling**: `weak_efficiency` con `n_per_rank` circa costante

## Debug e pulizia
Build debug:
```sh
make debug
```

Pulizia:
```sh
make clean
```
