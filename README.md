# VehicleRoutingProblem

Implementazione dell'Ant Colony Optimization (ACO) per il
Vehicle Routing Problem (VRP) con supporto:
- sequenziale (default)
- ibrido MPI + OpenMP (`aco_vrp`)

Il `main` esegue un esempio minimale senza argomenti e stampa il costo migliore
trovato.

## Requisiti
- `gcc`
- `make`
- `mpicc` e `mpirun` (solo per versione ibrida/test MPI)

## Avvio rapido
Compilazione ed esecuzione:
```sh
make
./aco_vrp_seq
```

## Esecuzione ibrida MPI + OpenMP
Compilazione:
```sh
make hybrid
```

Esecuzione (esempio con 2 rank e 2 thread OpenMP per rank):
```sh
OMP_NUM_THREADS=2 mpirun -np 2 ./aco_vrp_hybrid
```

## Test
Compila ed esegue i test:
```sh
make test
```
Il test confronta il solver C con una baseline PyVRP offline (`tests/files/golden_pyvrp.csv`),
quindi e adatto anche al cluster senza installare PyVRP.

Rigenera il golden in locale (venv `VRP`) con:
```sh
python3 tests/generate_pyvrp_golden.py
```

## Test MPI + OpenMP
Compila ed esegue un test parallelo:
```sh
make test_mpi
```

## Esperimenti (correttezza + scaling)
Esegue automaticamente benchmark con ripetizioni e genera CSV/grafici in
`results/`:
```sh
make experiments
```

## Scaling progressivo fino a 40k clienti (solo PyVRP)
Esegue test progressivi con PyVRP fino a `n=40000`, usando automaticamente il
venv `VRP` (`VRP/bin/python`) e con skip automatico dei casi che eccedono la
memoria disponibile stimata:
```sh
make scaling_tests
```

Output CSV predefinito:
`results/scaling_progressive_pyvrp.csv`

Esecuzione manuale (opzioni principali):
```sh
python3 tests/pyvrp_tests.py --memory-utilization 0.70 --pyvrp-max-n 40000
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
