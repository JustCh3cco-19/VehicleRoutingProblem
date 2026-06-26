# Implementation Improvements Roadmap

Questo documento raccoglie gli interventi che implementerei per migliorare correttezza,
robustezza, confrontabilita' sperimentale e manutenibilita' dei backend ACO/CVRP.

## Priorita' Alta

### 1. Allineare costo e soluzione globale nel backend OpenMP/MPI

File: `src/openmp-mpi/aco_parallel.c`

Problema: il backend riduce solo il costo minimo con `MPI_Allreduce`, ma poi ogni rank
puo' copiare la propria soluzione locale. Se il costo migliore appartiene a un altro
rank, `best_cost` e `best_sol` possono non rappresentare la stessa soluzione.

Da implementare:
- riduzione di una coppia `{cost, rank}` con tie-break deterministico;
- broadcast della soluzione del rank vincente a tutti i rank;
- aggiornamento di `best_sol` solo dopo aver ricevuto la soluzione corrispondente;
- test su almeno 2 rank con seed fisso e validazione della soluzione finale.

### 2. Rendere finita la terminazione di default

File:
- `src/seq/aco_sequential.c`
- `src/openmp-mpi/aco_parallel.c`
- `src/cuda/aco_cuda.cu`

Problema: alcuni backend possono non terminare se non vengono impostati timeout o
stagnation via variabili d'ambiente. Nel sequenziale e CUDA il default di
`ACO_SOLVER_STAGNATION_EPOCHS` e' 0; nel backend MPI il timeout viene caricato ma non
usato nel loop principale.

Da implementare:
- default coerente per tutti i backend, ad esempio `ACO_SOLVER_STAGNATION_EPOCHS=100`
  se non specificato;
- rispetto di `ACO_SOLVER_TIMEOUT_SECONDS` anche in OpenMP/MPI;
- comportamento documentato quando `ACO_SOLVER_FIXED_EPOCHS` e' impostato;
- smoke test che verifica la terminazione senza variabili d'ambiente.

### 3. Gestire gli errori di allocazione in modo sistematico

File:
- `src/openmp-mpi/aco_parallel.c`
- `src/cuda/aco_cuda.cu`
- `src/common/solution.c`
- `src/common/matrix.c`

Problema: diverse allocazioni non sono controllate o usano ritorni anticipati che
perdono risorse gia' allocate.

Da implementare:
- controlli espliciti dopo `malloc`, `calloc`, `aligned_alloc`, `cudaMalloc`;
- cleanup unico con `goto cleanup` nelle funzioni lunghe;
- ritorni di errore dai backend CPU invece di API `void`, dove possibile;
- messaggi d'errore con contesto: backend, dimensione problema, buffer fallito;
- test o modalita' fault-injection leggera per percorsi di errore critici.

### 4. Validare sempre la soluzione prodotta

File:
- `src/common/solution.c`
- `include/solution.h`
- `src/main.c`
- `src/cuda/main_vrp.cu`

Problema: `solution_validate` controlla struttura e copertura clienti, ma non verifica
la capacita' dei veicoli. Inoltre non viene usata sistematicamente dopo ogni backend.

Da implementare:
- nuova funzione `solution_validate_cvrp(const Solution *, int n, int K, int capacity, ...)`;
- controllo che ogni route serva al massimo `capacity` clienti;
- validazione finale in CLI sequenziale/MPI e CUDA;
- status non-zero se il backend produce una soluzione invalida;
- test con istanze piccole e casi limite.

## Priorita' Media

### 5. Rendere confrontabili le candidate list tra backend

File:
- `src/seq/aco_sequential.c`
- `src/openmp-mpi/aco_parallel.c`
- `src/cuda/aco_cuda.cu`
- `include/aco.h`

Problema: il sequenziale usa `candidate_k = n`, quindi valuta tutti i clienti, mentre
OpenMP/MPI e CUDA usano candidate list limitate. Questo rende meno pulito il confronto
prestazionale e puo' cambiare la dinamica dell'algoritmo.

Da implementare:
- parametro comune per `candidate_k`;
- default documentato per ogni scala di `n`;
- possibilita' di forzare `candidate_k` via env, ad esempio `ACO_CANDIDATE_K`;
- report nei log del valore effettivamente usato;
- benchmark A/B su qualita' e tempo.

### 6. Migliorare il parser TSPLIB-like

File:
- `src/common/instance_parser.c`
- `include/instance_parser.h`

Problema: il parser legge gli id dei nodi ma usa l'ordine di lettura come indice. Questo
funziona solo con file ordinati e depot atteso in posizione 0.

Da implementare:
- indicizzazione robusta tramite id letto dal file;
- verifica di id duplicati o fuori range;
- supporto esplicito per depot section se presente;
- sostituzione di `sqrt(pow(dx, 2) + pow(dy, 2))` con `hypot(dx, dy)`;
- messaggi d'errore con numero di sezione e id problematico.

### 7. Rendere `route_append` esplicita sugli errori

File:
- `src/common/solution.c`
- `include/solution.h`
- chiamanti nei backend

Problema: `route_append` fallisce silenziosamente se la route e' piena. Questo puo'
nascondere soluzioni tronche o incoerenti.

Da implementare:
- cambiare firma in `bool route_append(Route *r, int node)`;
- propagare il fallimento nella costruzione soluzione;
- invalidare l'ant/soluzione quando una route supera la capacita' di storage;
- aggiungere assert o controlli in debug.

### 8. Chiarire la semantica della deposizione CUDA

File:
- `src/cuda/aco_cuda_kernels.cu`
- `src/cuda/aco_cuda.cu`
- `docs/engineering/cuda_v6_qa.md`

Problema: CUDA deposita feromone con `atomicMax_uint8` su valori quantizzati. Questo non
equivale a una somma additiva classica: mantiene il massimo valore quantizzato tra update
concorrenti.

Da implementare:
- decidere se questa e' una scelta algoritmica intenzionale o un compromesso tecnico;
- documentare il razionale se viene mantenuta;
- altrimenti implementare una deposizione additiva con accumulo temporaneo o formato piu'
  adatto;
- confrontare qualita' soluzione e tempo su istanze piccole/medie;
- aggiungere contatori diagnostici per saturazione di `q_tau_max`.

## Priorita' Media-Bassa

### 9. Unificare configurazione e logging

File:
- `src/seq/aco_sequential.c`
- `src/openmp-mpi/aco_parallel.c`
- `src/cuda/aco_cuda.cu`
- `src/main.c`
- `src/cuda/main_vrp.cu`

Da implementare:
- struttura comune `AcoRuntimeConfig`;
- parsing unico di env var e default;
- stampa compatta di configurazione effettiva: backend, `n`, `K`, ants, seed,
  candidate list, timeout, stagnation, alpha/beta/rho/Q;
- livello log configurabile, ad esempio `ACO_LOG_LEVEL=quiet|info|debug`;
- output CSV opzionale per evitare parsing fragile di stdout.

### 10. Ridurre duplicazione tra backend

File:
- `src/common/aco_shared.c`
- `include/aco.h`
- backend in `src/seq`, `src/openmp-mpi`, `src/cuda`

Problema: esistono helper duplicati per power fast-path, timer directives, improvement
test e candidate tuning. Inoltre `aco_shared.c` espone una score cache che non sembra
usata dai backend principali.

Da implementare:
- spostare helper comuni in moduli piccoli e realmente usati;
- rimuovere o reintegrare `AcoScoreCache`;
- evitare API pubbliche non usate;
- mantenere differenze backend solo dove sono davvero necessarie.

### 11. Migliorare documentazione inline

File: molti file C/CUDA in `src/` e `include/`

Problema: molti commenti Doxygen sono generati e poco utili, ad esempio descrizioni del
tipo `Executes ...` e `Function parameter`.

Da implementare:
- rimuovere commenti ovvi o generici;
- documentare invece invarianti reali: layout memoria, ownership, thread-safety,
  limiti di dimensione, semantica delle env var;
- descrivere le scelte algoritmiche non standard, soprattutto nel backend CUDA.

## Priorita' Bassa

### 12. Migliorare layout e stile del codice OpenMP/MPI

File: `src/openmp-mpi/aco_parallel.c`

Da implementare:
- spezzare `aco_vrp_run` in funzioni piu' piccole;
- isolare inizializzazione, loop iterativo, sincronizzazione MPI, evaporazione/deposito;
- ridurre righe molto lunghe;
- usare nomi coerenti tra `cand_k`, `candidate_k`, `cap`, `vehicle_capacity_customers`;
- mantenere commenti solo dove aiutano a capire la sincronizzazione.

### 13. Aggiungere test automatici minimi

File/aree:
- `tools/bash/smoke_test.sh`
- eventuale nuova cartella `tests/`
- Makefile

Da implementare:
- test sequenziale su istanza piccola;
- test OpenMP con piu' thread;
- test MPI con 2 rank se MPI disponibile;
- test CUDA se `nvcc` e GPU sono disponibili;
- validazione soluzione finale;
- controllo che ogni backend termini con default;
- confronto deterministico con seed fisso dove possibile.

### 14. Migliorare benchmark e riproducibilita'

File:
- `tools/python/*.py`
- `docs/usage/practical_experiment_campaign.md`

Da implementare:
- registrare commit hash, compiler version, CUDA version, MPI implementation;
- registrare configurazione runtime effettiva;
- distinguere chiaramente tempo di parsing, setup, kernel/loop, cleanup;
- salvare anche validita' soluzione e capacita' rispettata;
- usare mediana e deviazione standard quando `repeats > 1`.

## Ordine Consigliato Di Implementazione

1. Correggere costo/soluzione globale in OpenMP/MPI.
2. Rendere finita e uniforme la terminazione di default.
3. Aggiungere validazione CVRP finale.
4. Sistemare gestione errori e cleanup CUDA/OpenMP-MPI.
5. Uniformare candidate list e configurazione.
6. Migliorare parser e `route_append`.
7. Decidere e documentare la semantica della deposizione CUDA.
8. Ripulire documentazione inline e codice condiviso.
9. Aggiungere smoke test e benchmark piu' riproducibili.

## Criteri Di Completamento

Un miglioramento e' considerato completo quando:
- compila per i backend interessati;
- la soluzione finale e' validata strutturalmente e rispetto alla capacita';
- il comportamento e' documentato se cambia un default o una scelta algoritmica;
- esiste almeno uno smoke test o comando di verifica ripetibile;
- i log riportano abbastanza informazioni per riprodurre il run.
