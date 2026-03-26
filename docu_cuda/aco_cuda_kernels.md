# Documentazione `aco_cuda_kernels.cu` e `.h`

## Descrizione Generale
Questo modulo contiene l'implementazione dei kernel CUDA che eseguono il lavoro computazionale pesante dell'algoritmo ACO. Il design è ottimizzato per sfruttare l'architettura della GPU, con un particolare focus sull'uso delle operazioni a livello di warp (warp-centric) per la costruzione delle soluzioni.

## Memoria Costante (`__constant__`)
Per ottimizzare l'accesso ai parametri globali dell'ACO che non cambiano durante l'esecuzione del kernel, vengono utilizzati i seguenti simboli in memoria costante:
- `d_alpha`, `d_beta`: Esponenti per il calcolo della probabilità.
- `d_rho`: Coefficiente di evaporazione.
- `d_tau0`: Valore iniziale del feromone.

## Kernel di Inizializzazione e Precalcolo

### `init_curand_states_kernel`
Inizializza un generatore di numeri casuali `curandState` per ogni formica. Ogni formica ha un proprio stato per garantire l'indipendenza statistica durante le scelte probabilistiche.

### `init_matrices_kernel`
Inizializza le matrici device:
- `eta` (Visibilità): calcolata come `1.0 / (costo + EPS)`.
- `tau` (Feromone): inizializzata al valore `tau0`.

### `precompute_scores_kernel`
Calcola in parallelo per ogni arco `(i, j)` il valore `tau[i][j]^alpha * eta[i][j]^beta`. Questo evita ricalcoli costosi di potenze durante la costruzione delle rotte.

### `precompute_candidate_lists_kernel`
Per ogni nodo, identifica i `MAX_CANDIDATES` (32) nodi più vicini. Questo limita lo spazio di ricerca durante la scelta del prossimo nodo, migliorando drasticamente le prestazioni senza sacrificare eccessivamente la qualità della soluzione.

## Utility a Livello di Warp
Vengono utilizzate funzioni intrinseche di CUDA (`__shfl_down_sync`, `__shfl_up_sync`) per eseguire riduzioni e prefix-sum all'interno di un warp senza ricorrere alla memoria condivisa:
- `warp_reduce_sum`: Calcola la somma totale di un valore tra i thread del warp.
- `warp_prefix_sum`: Esegue una scansione inclusiva (somma prefissa) utile per il campionamento probabilistico.

## Il Kernel di Costruzione: `construct_solutions_warp_v4_kernel`
Questo è il kernel più complesso e critico per le prestazioni.

### Strategia Warp-Centric
- Ogni formica è gestita da un **intero Warp** (32 thread).
- **Shared Memory**: Utilizzata per memorizzare una bitmask dei nodi visitati per ogni formica (warp), garantendo accessi veloci e atomici all'interno del blocco.
- **Parallelismo dei Veicoli**: La costruzione delle rotte per i `K` veicoli avviene in modo ciclico.

### Algoritmo di Scelta del Nodo
1. **Fase Candidati**: I 32 thread del warp controllano i nodi nella lista dei candidati del nodo corrente.
2. **Campionamento Probabilistico**:
   - Viene calcolata la somma locale degli score per i candidati non visitati.
   - Si genera un numero casuale.
   - Si usa la `warp_prefix_sum` per selezionare il nodo corrispondente al valore casuale (tecnica della roulette wheel).
3. **Fallback**: Se nessun candidato è disponibile o valido, il warp esegue una scansione parallela su **tutti** i nodi dell'istanza per trovare i nodi rimanenti.

## Kernel di Gestione Feromone

### `evaporate_pheromones_kernel`
Applica l'evaporazione su tutti gli archi della matrice `tau` in parallelo: `tau = tau * (1 - rho)`.

### `deposit_pheromones_kernel`
Ogni thread gestisce un veicolo della migliore soluzione dell'iterazione e deposita il feromone sugli archi percorsi utilizzando `atomicAdd`. Questo garantisce che gli aggiornamenti alla matrice globale siano sicuri anche se più rotte passano per lo stesso arco.

## Funzioni di Lancio (`launch_...`)
Queste funzioni (definite in `.h` e implementate in `.cu`) gestiscono la configurazione della griglia (blocchi e thread) e il passaggio dei parametri ai kernel, agendo come interfaccia tra l'orchestrazione host e l'esecuzione device.
- Nota: `launch_construct_solutions` calcola dinamicamente la dimensione della memoria condivisa necessaria per le bitmask in base al numero di nodi `n`.
