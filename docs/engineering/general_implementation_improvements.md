# General Implementation Improvements

Questo documento riassume miglioramenti possibili nell'implementazione generale
del solver VRP, analizzando `src/` e `include/`. Le priorita' sono orientate a
correttezza, scalabilita', manutenibilita' e benchmark riproducibili.

Regola di manutenzione: quando un miglioramento viene implementato, rimuoverlo
da questo documento. Il file deve contenere solo interventi ancora aperti.

## Priorita' Alta

### 1. Ridurre Il Costo O(n^2) Dove Non Necessario

CPU e MPI materializzano matrici complete delle distanze/feromoni. Per istanze
da `64000` o `100000` clienti, una matrice densa diventa rapidamente il collo di
bottiglia di memoria. CUDA usa feromone quantizzato `uint8_t`, ma resta una
matrice densa `(n+1)^2`.

Intervento consigliato:

- rendere candidate-list e pheromone sparse una scelta ufficiale;
- mantenere solo `k` archi candidati per nodo, piu' eventuale fallback;
- calcolare distanze EUC_2D on demand quando possibile;
- introdurre una policy runtime: dense per piccoli `n`, sparse per grandi `n`.

Beneficio: il progetto scala davvero verso le dimensioni generate dal Makefile.

## Priorita' Media

### 2. Unificare Candidate List Tra Backend

Il sequenziale usa `choose_candidate_count(n) { return n; }`, quindi di fatto
usa tutti i nodi. MPI usa tuning basato su L3 cache. CUDA fissa `cand_k = 32`.
Queste differenze rendono difficili confronti scientifici tra backend.

Intervento consigliato:

- usare `AcoRuntimeConfig.candidate_k` in tutti i backend;
- documentare default diversi solo se motivati;
- aggiungere nel CSV benchmark il valore effettivo di `candidate_k`.

Beneficio: confronti piu' equi e tuning controllabile.

### 3. Rendere Determinismo E Seed Piu' Espliciti

Il seed e' passato via CLI e salvato in `AcoRuntimeConfig`, ma ogni backend lo
usa con logiche diverse. MPI/OpenMP e CUDA possono produrre differenze legate a
ordine delle riduzioni e scheduling.

Intervento consigliato:

- definire una policy ufficiale di reproducibility;
- distinguere "deterministico bitwise" da "statisticamente riproducibile";
- aggiungere modalita' benchmark deterministica: schedule fisso, reduction
  deterministica, stesso numero formiche quando possibile;
- stampare nel report seed, backend, risorse e candidate_k.

Beneficio: risultati benchmark piu' ripetibili e differenze spiegabili.

### 4. Separare Log Di Progresso Da Output Soluzione

I backend stampano progresso su `stderr`, mentre la soluzione finale va su
`stdout`. Questo e' corretto, ma ci sono ancora stampe hardcoded nei backend
MPI/CUDA (`ACO Parallel Starting`, completamento, `CUDA Solver starting`).

Intervento consigliato:

- introdurre livello di logging in `AcoRuntimeConfig`;
- rendere silenziabili tutte le stampe non essenziali;
- mantenere `stdout` parsabile solo per route e costo;
- inviare diagnostica e progressi sempre a `stderr`.

Beneficio: script benchmark piu' robusti e output pulito.

### 5. Migliorare Modularita' Del Backend MPI/OpenMP

`src/openmp-mpi/aco_parallel.c` contiene molte responsabilita' in un solo file:
matrici float, candidate tuning, workspace, costruzione soluzione, sparse sync,
loop ACO e API backend.

Intervento consigliato:

- separare in moduli:
  - `aco_mpi_matrix.c`;
  - `aco_mpi_candidates.c`;
  - `aco_mpi_workspace.c`;
  - `aco_mpi_sync.c`;
  - `aco_mpi_solver.c`;
- mantenere il file API piccolo e leggibile;
- aggiungere test unitari su candidate list e sync sparse.

Beneficio: debug piu' semplice e meno rischio di regressioni.

### 6. Controllare Allocazioni Nel Backend MPI

`matrix_create_float()` alloca struttura, data e rows senza controlli intermedi.
Se una allocazione fallisce, il codice puo' dereferenziare puntatori nulli.

Intervento consigliato:

- rendere `matrix_create_float()` robusta come `matrix_create()`;
- gestire overflow `size_t` prima di calcolare buffer grandi;
- restituire `ACO_ERR_ALLOCATION` in modo consistente.

Beneficio: comportamento definito su istanze grandi o memoria insufficiente.

### 7. Evitare Silenziamento Di Errori In `route_append()`

`route_append()` ignora l'append se la route e' piena. Questo evita crash, ma
puo' nascondere soluzioni troncate.

Intervento consigliato:

- far ritornare `bool` o `AcoStatus`;
- propagare errore quando una route eccede la capacita' del buffer;
- aggiungere assert o contatori diagnostici nei backend.

Beneficio: errori di costruzione route visibili subito.

### 8. Rendere Il Parser TSPLIB Meno Rigido

Il parser richiede `VEHICLES`, `CAPACITY`, `DEMAND_SECTION` e demand unitaria per
tutti i clienti. Questo va bene per le istanze generate dal progetto, ma limita
compatibilita' con dataset CVRPLIB/TSPLIB reali.

Intervento consigliato:

- supportare demand arbitrarie;
- accettare istanze senza `VEHICLES` quando `K` arriva da CLI;
- validare `DEPOT_SECTION`;
- usare gli id nodo letti dal file invece di assumere ordine 1..n+1;
- restituire errori strutturati invece di solo `int`.

Beneficio: benchmark su istanze standard esterne piu' facile.

## Priorita' Bassa Ma Utile

### 9. Ripulire Commenti Generati

Molti commenti hanno forma `@brief Executes ...` e parametri generici
`Function parameter`. Non aiutano la manutenzione.

Intervento consigliato:

- rimuovere commenti ovvi;
- documentare solo invarianti, ownership, complessita' e motivazioni;
- aggiornare i commenti pubblici negli header.

Beneficio: codice piu' leggibile e documentazione meno rumorosa.

### 10. Unificare Tipi Di Matrice

Esistono `Matrix` double in common, `MatrixFloat` privata in MPI e buffer CUDA.
Questa differenza e' giustificata dai backend, ma manca una policy chiara.

Intervento consigliato:

- documentare quando usare `double`, `float`, `uint8_t` quantizzato;
- introdurre funzioni comuni per stride/alignment;
- centralizzare controlli di overflow e allocazione allineata.

Beneficio: meno duplicazione e meno bug di memoria.

### 11. Rendere Il Main Solo Orchestrazione

`src/main.c` gestisce parsing modalita', caricamento istanza, demo matrix,
allocazione soluzione, esecuzione solver e output. Sta crescendo.

Intervento consigliato:

- creare `AcoCliOptions`;
- spostare parsing comando in `cli_common`;
- rimuovere o separare la modalita' demo `[n K m seed]`;
- aggiungere opzioni esplicite per `alpha`, `beta`, `rho`, `tau0`, `Q`.

Beneficio: CLI piu' chiara e meno branching nel main.

### 12. Spostare Il Main CUDA Nella Root Di `src`

`src/cuda/main_vrp.cu` e' un entrypoint CLI, non logica del backend CUDA. Tenerlo
dentro `src/cuda/` mescola orchestrazione e backend.

Intervento consigliato:

- spostare `src/cuda/main_vrp.cu` in `src/main_cuda.cu`;
- aggiornare Makefile e dipendenze;
- lasciare in `src/cuda/` solo implementazione backend e kernel;
- mantenere naming coerente con `src/main.c`.

Beneficio: struttura piu' chiara tra entrypoint e backend.

## Miglioramenti Specifici Per Benchmark

### 13. Registrare Metadata Completi In Output CSV

Per confronti seri servono piu' informazioni nei risultati.

Campi consigliati:

- backend;
- commit hash;
- compiler e flags;
- host/GPU;
- `n`, `K`, capacity, `m`;
- candidate_k effettivo;
- seed;
- timeout/stagnation/min improvement;
- rank MPI, thread OpenMP;
- CUDA arch;
- runtime, best cost, validazione, exit status.

### 14. Aggiungere Test Di Regressione Su Correttezza

Gli smoke test sono utili, ma servono casi piu' mirati.

Test consigliati:

- istanza piccola con soluzione nota;
- capacita' impossibile;
- demand non unitaria quando supportata;
- route overflow;
- confronto costo stampato vs costo ricalcolato;
- backend che deve restituire `ACO_ERR_NO_SOLUTION`.

### 15. Profilazione Guidata

Prima di ottimizzare ancora, conviene produrre dati stabili.

Strumenti consigliati:

- `perf`/`gprof` per CPU;
- MPI profiling per tempo in comunicazione;
- Nsight Systems/Compute per CUDA;
- contatori interni gia' presenti in `CudaIterStats`;
- metriche cache per candidate list e fallback.

## Roadmap Suggerita

1. Unificare `candidate_k` tramite `AcoRuntimeConfig`.
2. Rendere logging configurabile e completamente su `stderr`.
3. Spostare il main CUDA in `src/main_cuda.cu`.
4. Implementare modalita' sparse per istanze grandi.
5. Separare `aco_parallel.c` in moduli piu' piccoli.
6. Estendere parser TSPLIB/CVRPLIB.
7. Rafforzare benchmark CSV e test di regressione.
