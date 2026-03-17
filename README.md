# VehicleRoutingProblem

Implementazione dell'Ant Colony Optimization (ACO) per il
Vehicle Routing Problem (VRP) con supporto:
- sequenziale (default)
- ibrido MPI + OpenMP (stessa funzione `aco_vrp_sequential`)

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
