# Documentazione `aco_cuda.cu`

## Descrizione Generale
`aco_cuda.cu` implementa l'orchestrazione principale del risolutore CUDA per il Vehicle Routing Problem. Contiene la funzione `aco_vrp_cuda` che gestisce l'intero ciclo di vita dell'algoritmo ACO sulla GPU, coordinando l'allocazione della memoria, il lancio dei kernel e il trasferimento dei dati tra host e device.

## Funzione `aco_vrp_cuda`

### Firma
```c
int aco_vrp_cuda(int n, int K, int m, int T, double **c,
                 double alpha, double beta, double rho,
                 double tau0, double Q, unsigned int seed,
                 bool use_uniform_pheromone, int convergence_iter,
                 Solution *best_solution, double *best_cost);
```

### Parametri Principali
- `n`: Numero di clienti (escluso il deposito).
- `K`: Numero di veicoli disponibili.
- `m`: Numero di formiche (parallelismo a livello di formica).
- `T`: Numero massimo di iterazioni.
- `c`: Matrice dei costi (distanze).
- `alpha`, `beta`, `rho`, `tau0`, `Q`: Parametri specifici ACO.
- `convergence_iter`: Soglia per l'arresto anticipato.
- `best_solution`, `best_cost`: Puntatori per restituire il risultato all'host.

## Flusso di Esecuzione

### 1. Preparazione e Allocazione (Host & Device)
- **Appiattimento Matrice**: La matrice 2D dell'host viene convertita in un array 1D (`h_c_flat`) per facilitare il trasferimento e l'accesso in memoria device.
- **Allocazione Device**: Vengono allocati i buffer sulla GPU per:
  - Costi (`d_c`), Visibilità (`d_eta`), Feromone (`d_tau`), Score precalcolati (`d_scores`).
  - Liste candidati (`d_candidates`) per ottimizzare la scelta del prossimo nodo.
  - Stati del generatore di numeri casuali (`d_curand_states`) per ogni formica.
  - Rotte, lunghezze e costi prodotti dalle formiche.

### 2. Inizializzazione
- Copia della matrice dei costi sul device.
- Lancio di `launch_init_matrices` per inizializzare `d_eta` (1/costo) e `d_tau` (tau0).
- Precalcolo delle liste dei candidati (`launch_precompute_candidate_lists`).
- Inizializzazione degli stati cuRAND (`init_curand_states`).

### 3. Loop Iterativo (Generazione Soluzioni)
Per ogni iterazione fino a `T`:
1. **Precompute Scores**: Calcola `tau^alpha * eta^beta` per ogni arco (`launch_precompute_scores`).
2. **Costruzione Soluzioni**: Ogni formica costruisce una soluzione completa in parallelo (`launch_construct_solutions`). Questo è il cuore del parallelismo CUDA, dove ogni thread gestisce una formica.
3. **Sincronizzazione e Riduzione**: 
   - `cudaDeviceSynchronize` per attendere il completamento di tutte le formiche.
   - Copia dei costi sull'host per trovare l'ape (formica) migliore dell'iterazione attuale.
4. **Aggiornamento Best**: Se la formica migliore dell'iterazione migliora il record globale, la sua rotta viene copiata dal device all'host.
5. **Aggiornamento Feromone**:
   - Evaporazione globale (`launch_evaporate_pheromones`).
   - Deposito basato sulla formica migliore dell'iterazione (`launch_deposit_pheromones`).
6. **Controllo Convergenza**: Se non ci sono miglioramenti per `convergence_iter` iterazioni, l'algoritmo termina anticipatamente.

### 4. Cleanup
- Deallocazione di tutta la memoria device tramite `cudaFree`.
- Deallocazione dei buffer temporanei sull'host.

## Considerazioni Tecniche

### Parallelismo
L'algoritmo sfrutta il parallelismo massivo della GPU assegnando una formica a ciascun thread. Il numero di formiche `m` determina il numero totale di thread attivi durante la fase di costruzione.

### Ottimizzazioni
- **Liste Candidati**: Riduce lo spazio di ricerca per il prossimo nodo da visitare, limitando i controlli ai nodi più vicini.
- **Precompute Scores**: Evita di ricalcolare le potenze (costose) all'interno del ciclo di scelta del nodo per ogni formica, spostando il calcolo a livello di arco una volta per iterazione.
