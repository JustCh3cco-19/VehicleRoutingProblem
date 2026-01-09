# VehicleRoutingProblem

Implementazione sequenziale dell'Ant Colony Optimization (ACO) per il
Vehicle Routing Problem (VRP). Il `main` esegue un esempio minimale senza
argomenti e stampa il costo migliore trovato.

## Requisiti
- `gcc`
- `make`

## Avvio rapido
Compilazione ed esecuzione:
```sh
make
./aco_vrp_seq
```

## Test
Compila ed esegue i test:
```sh
make test
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
