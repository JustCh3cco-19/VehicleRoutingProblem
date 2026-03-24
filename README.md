# VehicleRoutingProblem

Implementazione di Ant Colony Optimization (ACO) per il Vehicle Routing Problem (VRP), con supporto:
- sequenziale (default)
- ibrido MPI + OpenMP

Il `main` esegue un esempio minimale senza argomenti e stampa il costo migliore trovato.

## Requisiti
- `gcc`
- `make`
- `mpicc` e `mpirun` (solo per versione ibrida / test MPI)
- `python3` (per script test/scaling)
- ambiente `VRP` con PyVRP (solo per baseline/golden e scaling PyVRP)

## Avvio rapido
Compilazione ed esecuzione sequenziale:
```sh
make
./aco_vrp_seq.out
```

## Esecuzione ibrida MPI + OpenMP
Compilazione:
```sh
make openmp_mpi
```

Esecuzione (esempio con 2 rank e 2 thread OpenMP per rank):
```sh
OMP_NUM_THREADS=2 mpirun -np 2 ./aco_vrp_openmp_mpi.out
```

## Test (background di default)
I target di test partono in background (`TEST_MODE=background`) per uso su cluster.
I log sono salvati in `results/detached/` con file `.log`, `.pid`, `.cmd`.

Per esecuzione in foreground:
```sh
make <target> TEST_MODE=foreground
```

Per cambiare directory dei log:
```sh
make <target> TEST_LOGS_DIR=/path/to/logs
```

Target disponibili:
- `sequential_tests`
- `openmp_mpi_tests`

## Sequential Tests (fino a 100k clienti)
Runner C dedicato per scenari progressivi fino a `n=100000`:
```sh
make sequential_tests
```

Output CSV predefinito:
- `results/scaling_progressive_c.csv`

Esecuzione manuale:
```sh
./tests/sequential_tests.out --memory-utilization 0.70 --c-max-n 100000
```

Opzioni utili:
- `--memory-utilization X` frazione RAM usabile prima di fare skip (`0 < X <= 1`)
- `--c-max-n N` limite massimo clienti eseguiti
- `--csv PATH` percorso CSV output
- `--force` forza l'esecuzione anche oltre la soglia memoria stimata

Nota: la parte C usa matrici dense (`c`, `eta`, `tau`), quindi i casi molto grandi possono essere saltati automaticamente per evitare out-of-memory.

## OpenMP + MPI Tests
Runner OpenMP+MPI progressivo:
```sh
make openmp_mpi_tests
```

Esecuzione su cluster con SLURM (batch, puoi disconnetterti):
```sh
./scripts/submit_openmp_mpi_tests.sh
```

Con argomenti opzionali passati al binario:
```sh
./scripts/submit_openmp_mpi_tests.sh "--memory-utilization 0.70 --c-max-n 100000"
```

## Debug
Compila con `-DDEBUG`:
```sh
make debug
```

## Pulizia
Rimuove eseguibili, oggetti e file di coverage:
```sh
make clean
```
