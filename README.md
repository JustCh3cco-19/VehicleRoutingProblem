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

## Test di correttezza (C vs PyVRP golden offline)
Compila ed esegue i test file-based:
```sh
make test
```

Il test confronta il solver C con una baseline PyVRP offline in:
- `tests/files/golden_pyvrp.csv`

Quindi i test funzionano anche su cluster dove PyVRP non e installato (purche il CSV golden sia gia presente).

Rigenerazione golden in locale (venv `VRP`):
```sh
python3 tests/generate_pyvrp_golden.py
```

## Test MPI + OpenMP
Compila ed esegue un test parallelo:
```sh
make test_mpi
```

Test race-oriented su piu casi/rank:
```sh
make test_mpi_race
```

## Scaling progressivo fino a 100k clienti (PyVRP)
Esegue scenari progressivi fino a `n=100000` con PyVRP, usando automaticamente `VRP/bin/python`.

Comando rapido:
```sh
make scaling_tests
```

Output CSV predefinito:
- `results/scaling_progressive_pyvrp.csv`

Esecuzione manuale (opzioni principali):
```sh
python3 tests/pyvrp_tests.py --memory-utilization 0.70 --pyvrp-max-n 100000
```

Opzioni utili:
- `--memory-utilization X` frazione RAM usabile prima di fare skip (`0 < X <= 1`)
- `--pyvrp-max-n N` limite massimo di clienti da eseguire
- `--csv PATH` percorso CSV output
- `--force` disabilita skip basato su memoria stimata

## Scaling progressivo fino a 100k clienti (solver C)
Nuovo runner C dedicato per scenari progressivi fino a `n=100000`:
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

## Esperimenti (correttezza + scaling)
Esegue benchmark con ripetizioni e genera CSV/grafici in `results/`:
```sh
make experiments
```

## Report PDF
Genera il report in `report/report.pdf`:
```sh
make report
```

## Coverage
Compila con flag di coverage ed esegue i test (genera `*.gcno` e `*.gcda`):
```sh
make coverage
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
