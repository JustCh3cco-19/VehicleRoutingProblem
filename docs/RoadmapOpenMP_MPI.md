# Roadmap Evolutiva OpenMP-MPI: Verso l'HPC (V2)

Questo documento traccia l'evoluzione dell'implementazione parallela del solutore ACO per il Constrained VRP, documentando i problemi individuati, le soluzioni architetturali adottate e i risultati degli esperimenti di scaling.

## 1. Analisi delle Criticità V1 (Baseline)

Nella versione iniziale sono stati identificati diversi colli di bottiglia che limitavano sia lo **Strong Scaling** che il **Weak Scaling**:

*   **False Sharing Catastrofico:** La telemetria scriveva continuamente in un array di strutture non allineate, causando collisioni costanti nelle cache L1/L2.
*   **Overhead Fork-Join:** La regione parallela veniva aperta e chiusa a ogni epoca, introducendo latenze fisse significative.
*   **Serializzazione del Deposito:** Il deposito dei feromoni era eseguito da un singolo thread (`#pragma omp single`), rendendo la fase finale di ogni epoca un collo di bottiglia seriale (Legge di Amdahl).
*   **Latenza MPI Elevata:** Veniva effettuata una chiamata `MPI_Iallreduce` per ogni riga della matrice ($N+1$ chiamate), saturando la rete con pacchetti piccoli.

## 2. Architettura V2: I Pilastri del Miglioramento

L'architettura V2 è stata riprogettata seguendo i principi del calcolo ad alte prestazioni (HPC):

### A. Persistent Threading (Zero Fork-Join)
La regione parallela viene aperta una sola volta all'esterno del loop delle epoche. I thread sono sincronizzati tramite barriere leggere, eliminando l'overhead di creazione/distruzione.

### B. Cache-Awareness & Auto-tuning
*   **Padding & Alignment:** Tutte le strutture dati private sono allineate a 64 byte (cache line size).
*   **L3 Auto-tuning:** All'avvio, il solutore rileva la dimensione della cache L3 del sistema e adatta dinamicamente il parametro $K_{cand}$ (vicini prossimi) affinché la matrice degli score risieda interamente in cache, massimizzando la banda di memoria.

### C. Atomic Pheromone Update
Il deposito dei feromoni è ora **totalmente parallelo**. I thread scrivono simultaneamente sulla matrice globale utilizzando istruzioni atomiche hardware (`#pragma omp atomic`), eliminando la sezione seriale della V1.

### D. Ottimizzazione Network (Contiguous MPI)
La sincronizzazione MPI è stata ridotta a una **singola operazione collettiva** (`MPI_Allreduce`) sull'intero blocco di memoria contiguo della matrice dei feromoni, riducendo la latenza di rete di ordini di grandezza.

## 3. Esperimenti e Validazione Empirica

### Esperimento 1: Granularità dello Scheduling (N=2000, Threads=24)
Abbiamo confrontato diverse strategie di assegnazione delle formiche ai thread per identificare il punto di equilibrio tra bilanciamento del carico e overhead di sincronizzazione.

| Strategia | Tempo Medio (s) | Speedup vs Baseline | Conclusione |
| :--- | :--- | :--- | :--- |
| `dynamic, 1` | 30.5 s | 1.0x | Eccessiva contesa sui lock interni di OpenMP. |
| `dynamic, 4` | 15.0 s | 2.0x | La riduzione delle richieste di task dimezza il tempo. |
| **`guided, 1`** | **4.1 s** | **~7.4x** | **Vincitore.** Granularità adattiva: blocchi grandi all'inizio, piccoli alla fine. |

### Esperimento 2: Strong Scaling e Rumore di Convergenza (End-to-End)
In questo primo test abbiamo misurato il tempo di esecuzione totale fino alla terminazione naturale dell'algoritmo. Questo esperimento evidenzia come la logica di business (stagnazione/timeout) interagisca con il parallelismo.

| Threads | Tempo Min (s) | Speedup (Rel) | Efficiency |
| :--- | :--- | :--- | :--- |
| 1 | 20.50 s | 1.0x | 100% |
| 2 | 18.09 s | 1.1x | 56% |
| 4 | 9.71 s | 2.1x | 52% |
| 8 | 5.30 s | 3.9x | 48% |
| 16 | 3.82 s | 5.4x | 33% |
| 24 | 2.23 s | 9.2x | 38% |

#### Analisi dei Risultati:
*   **Speedup Massimo:** Abbiamo registrato un'accelerazione di **9.2x** su 24 core fisici.
*   **Rumore Stocastico:** Si è notata una forte varianza tra le run dello stesso test (es. a 4 thread). Questo è dovuto alla natura dell'ACO: se una run trova una soluzione ottima rapidamente, il programma termina prima a causa della logica di stagnazione, mascherando le performance pure dell'hardware.
*   **Conclusione:** L'architettura V2 permette una scalabilità significativa, ma la variabilità algoritmica rende difficile misurare l'efficienza hardware pura in questo scenario.

### Esperimento 3: Scientific Hardware Scaling (FIXED EPOCHS = 100)
Per isolare lo scaling dell'hardware dal comportamento stocastico dell'algoritmo, abbiamo forzato il solutore a eseguire esattamente 100 iterazioni, misurando il **Tempo medio per Epoca (ms)**.

| Threads | Tempo/Epoca (ms) | Speedup (Real) | Efficiency |
| :--- | :--- | :--- | :--- |
| 1 | 308.4 ms | 1.0x | 100% |
| 2 | 163.8 ms | 1.9x | 94% |
| 4 | 83.3 ms | 3.7x | 92% |
| 8 | 44.7 ms | 6.9x | 86% |
| 16 | 26.9 ms | 11.4x | 71% |
| 24 | 21.0 ms | **14.6x** | **60%** |

#### Analisi dei Risultati:
*   **Precisione Scientifica:** Grazie alle epoche fisse, la varianza tra le run è scesa sotto l'1%, confermando la stabilità del Dataflow.
*   **Scaling Eccellente:** Fino a 8 core il sistema è quasi lineare (efficienza > 85%).
*   **Saturazione:** A 24 core l'efficienza scende al 60%. Questo è dovuto alla saturazione della banda di memoria (molti core che leggono le matrici) e alla contesa atomica nel deposito feromoni. Tuttavia, uno speedup di **14.6x** su 24 core è un risultato eccellente per questa classe di algoritmi.

### Esperimento 4: Scaling Up (N=8000, Ants=768, Threads=24)
Abbiamo testato la V2 su un'istanza 4 volte più grande per verificare se la granularità del `chunk` minimo nello scheduling `guided` influenzi le performance su problemi pesanti.

| Strategia | Tempo/Epoca (ms) | Osservazioni |
| :--- | :--- | :--- |
| `guided, 1` | 173.3 ms | Ottimo bilanciamento, overhead trascurabile. |
| **`guided, 4`** | **170.4 ms** | **Punto di ottimo.** Leggera riduzione della contesa sui task. |
| `guided, 16` | 179.5 ms | Inizio del "Tail Effect" (alcuni thread aspettano alla barriera). |

#### Analisi della Granularità vs Dimensione:
Dagli esperimenti emerge una regola chiara per lo scheduling OpenMP nell'ACO:
1.  **Istanze Piccole (N < 2000):** L'overhead di OpenMP è critico. `guided,1` o `dynamic,4` sono necessari per non sprecare cicli CPU in lock hardware.
2.  **Istanze Grandi (N > 8000):** Il calcolo della singola formica diventa così oneroso (~0.2ms - 1ms) da "annegare" l'overhead di scheduling. In questo regime, quasi ogni strategia dinamica/guided performa bene, ma `guided,4` offre il miglior compromesso tra stabilità e bilanciamento.

### Esperimento 5: Impatto dello Scheduling su Scale Diverse (N=2k -> 16k)
Abbiamo misurato come il divario di performance tra `dynamic` e `guided` si amplifichi drasticamente all'aumentare della taglia del problema.

| Taglia (N) | Miglior Dynamic (ms) | Miglior Guided (ms) | Divario (Speedup) |
| :--- | :--- | :--- | :--- |
| 2.000 | 124 ms | 54 ms | 2.3x |
| 4.000 | 366 ms | 95 ms | 3.8x |
| 8.000 | 1.322 ms | 223 ms | 5.9x |
| **16.000** | **5.066 ms** | **739 ms** | **6.8x** (vs dynamic,4) |
| 16.000 | 18.157 ms | 739 ms | **24.5x** (vs dynamic,1) |

#### Conclusioni Finali sullo Scheduling:
*   Lo scheduling **`guided`** è l'unico in grado di mantenere l'overhead sotto controllo quando la contesa sulla coda dei task aumenta.
*   A 16.000 clienti, l'uso di uno scheduling errato (`dynamic,1`) può rendere il solutore **24 volte più lento** a parità di hardware.
*   La scelta definitiva per la versione di produzione della V2 è **`guided,4`** per la massima stabilità su tutte le taglie.

## 4. Conclusioni e Best Practice

Per massimizzare lo scaling della versione OpenMP-MPI, si raccomanda di:
1.  Utilizzare sempre lo scheduling **`guided`** per la costruzione stocastica dei percorsi.
2.  Garantire il binding dei thread ai core fisici (`OMP_PROC_BIND=close`) per sfruttare l'auto-tuning della cache L3.
3.  Preferire istanze con un numero di veicoli $K$ sufficientemente alto da saturare il parallelismo nella fase di deposito feromoni.

## Prossimi Passi (V3)
*   **Overlap Comunicazione/Computazione:** Implementazione di `MPI_Iallreduce` non bloccante per iniziare lo scambio dell'epoca $i$ mentre si calcolano gli score dell'epoca $i+1$.
*   **Vectorization (AVX-512):** Sfruttare le unità SIMD per il calcolo della matrice degli score.
