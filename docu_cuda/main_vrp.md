# Documentazione `main_vrp.cu`

## Descrizione Generale
`main_vrp.cu` è il punto di ingresso (entry point) dell'applicazione VRP Solver basata su CUDA. Questo file funge da driver principale: gestisce il parsing degli argomenti da riga di comando, carica l'istanza del problema, inizializza i parametri dell'algoritmo Ant Colony Optimization (ACO), invoca il risolutore CUDA e infine stampa i risultati ottenuti.

## Struttura del Codice

### Inclusione delle Header
Il file include le seguenti librerie e header di progetto:
- `aco.h`: Dichiarazione della funzione principale del risolutore CUDA (`aco_vrp_cuda`).
- `instance_parser.h`: Funzioni per il caricamento delle istanze VRP (formato TSPLIB).
- `matrix.h`: Utility per la gestione delle matrici di adiacenza (costi).
- `solution.h`: Strutture e funzioni per la gestione delle soluzioni VRP.
- Librerie standard: `stdio.h`, `stdlib.h`.

### Funzione `main`
La funzione `main` segue un flusso di esecuzione lineare:

1. **Validazione degli Argomenti**: Verifica che siano stati passati almeno i parametri obbligatori.
   - Utilizzo: `./aco_vrp_cuda <instance.vrp> <K> <m> <T> [seed] [convergence]`
2. **Parsing dei Parametri**:
   - `instance.vrp`: Percorso del file dell'istanza.
   - `K`: Numero di veicoli (capacità flotta).
   - `m`: Numero di formiche (popolazione per iterazione).
   - `T`: Numero massimo di iterazioni.
   - `seed` (opzionale): Seme per la generazione di numeri casuali (default: 1234).
   - `convergence` (opzionale): Numero di iterazioni senza miglioramento prima dell'uscita anticipata (default: 100).
3. **Caricamento Istanza**: Utilizza `vrp_load_tsplib_euc2d_matrix` per leggere le coordinate dei nodi e calcolare la matrice dei costi euclidei.
4. **Inizializzazione Parametri ACO**: Vengono definiti i parametri standard dell'algoritmo:
   - `alpha = 1.0`, `beta = 2.0`: Influenze relative di feromone ed euristica.
   - `rho = 0.5`: Coefficiente di evaporazione.
   - `tau0 = 1.0`: Livello iniziale di feromone.
   - `Q = 100.0`: Costante per il deposito di feromone.
5. **Esecuzione del Risolutore**: Viene allocata una struttura `Solution` e chiamata la funzione `aco_vrp_cuda`.
6. **Output dei Risultati**: 
   - Stampa delle rotte per ciascuno dei `K` veicoli.
   - Stampa del costo totale della migliore soluzione trovata.
7. **Cleanup**: Deallocazione della memoria per la soluzione e la matrice dei costi.

## Dettagli di Implementazione Rilevanti

### Gestione delle Rotte
Il codice itera sulle rotte della soluzione `best` e stampa i nodi visitati da ogni veicolo. I nodi identificati con `0` (deposito) vengono saltati nella stampa interna per chiarezza, assumendo che ogni rotta inizi e finisca implicitamente al deposito.

### Integrazione CUDA
Sebbene `main_vrp.cu` sia codice C host standard, la sua compilazione con `nvcc` permette di collegarsi alle funzioni kernel e alle utility definite in `aco_cuda.cu`. La funzione `aco_vrp_cuda` incapsula tutta la logica di gestione della memoria device e il lancio dei kernel.

## Error Handling
- Il programma termina con errore se il numero di argomenti è insufficiente.
- Gestisce il fallimento del caricamento dell'istanza.
- Gestisce il fallimento dell'esecuzione del solver CUDA, liberando la memoria già allocata prima di uscire.
