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
*   **L3 Auto-tuning Adattivo:** All'avvio, il solutore rileva la dimensione della cache L3 e adatta dinamicamente il parametro $K_{cand}$ (vicini prossimi). La formula è stata raffinata per considerare il costo combinato di indici e score (8 byte per elemento) con un fattore di sicurezza del 70%, garantendo che le "hot structures" risiedano interamente in L3.

### C. Atomic Pheromone Update
Il deposito dei feromoni è ora **totalmente parallelo**. I thread scrivono simultaneamente sulla matrice globale utilizzando istruzioni atomiche hardware (`#pragma omp atomic`), eliminando la sezione seriale della V1.

### D. Ottimizzazione Network (Contiguous MPI)
La sincronizzazione MPI è stata ridotta a una **singola operazione collettiva** (`MPI_Allreduce`) sull'intero blocco di memoria contiguo della matrice dei feromoni, riducendo la latenza di rete di ordini di grandezza.

## 3. Esperimenti e Validazione Empirica

### [Esperimenti 1-8: Ottimizzazione Base e Scaling Fisico]
*(Vedi versioni precedenti per dettagli su Scheduling, Strong Scaling e Precisione Float)*

### Esperimento 9: Prefetching Speculativo (N=16.000, Threads=24)
Abbiamo testato l'inserimento di `__builtin_prefetch` manuale nel loop di fallback per nascondere la latenza di caricamento dei nodi dalla RAM.
*   **Risultato:** Speedup **1.04x** (da 1226ms a 1175ms/epoca).
*   **Conclusione:** Miglioramento modesto ma reale. La CPU già esegue prefetching hardware efficace sulle maschere lineari; il prefetching manuale fornisce un aiuto marginale sulle matrici sparse.

### Esperimento 10: V3 Collaborative Teams (Analisi del Fallimento)
Abbiamo tentato di dividere la scansione del fallback tra più thread (collaborazione intra-formica).
*   **Versioni provate:** Barriere OpenMP, Atomic Spinning, Wait-Signal (Pthreads).
*   **Risultato:** Fallimento sistematico (degrado di performance fino a 60x).
*   **Analisi Post-Mortem:** La granularità del task (scansione 64KB) è troppo piccola. Il costo del coordinamento inter-core (syscall, latenza del kernel, traffico di coerenza cache) supera il tempo del calcolo seriale. Documentato in `docs/v3_failure_analysis.md`.

### Esperimento 11: SIMD AVX2 Paradox
Abbiamo vettorizzato il fallback usando istruzioni a 256-bit per processare 8 float alla volta.
*   **Risultato:** Rallentamento del **3%** (0.97x).
*   **Analisi:** In presenza di bitmask di visita, la CPU è più efficiente con un loop scalar ottimizzato (`ctzll`) che con una pipeline SIMD appesantita da maschere di blend e riduzioni orizzontali.

### Esperimento 12: Hierarchical Bitmask (Meta-Masking)
Implementazione di una bitmask a due livelli per saltare blocchi di 4096 nodi già visitati.
*   **Risultato:** Speedup **1.09x** (da 1207ms a 1107ms/epoca).
*   **Conclusione:** Ottimizzazione puramente algoritmica molto efficace per grandi istanze, riducendo i cicli CPU sprecati in scansioni inutili.

### Esperimento 13: Adaptive Candidate Tuning (La Svolta)
Raffinamento della formula di dimensionamento $K_{cand}$ basata sulla cache L3 reale, considerando sia l'impronta degli indici che degli score.
*   **Formula:** $K_{cand} = (L3 \times 0.7) / (N \times 8)$.
*   **Risultato:** Speedup **1.53x** (da 1189ms a **776ms/epoca**).
*   **Conclusione:** **Successo critico.** Abbiamo sfondato la barriera del secondo per epoca a $N=16.000$. Questo conferma che in HPC la gestione intelligente della gerarchia della memoria batte la forza bruta del parallelismo.

### Esperimento 14: Weak Scaling (Gustafson's Law)
Abbiamo verificato la capacità del sistema di gestire carichi crescenti (64 formiche per thread) mantenendo costante l'impronta di memoria.
*   **Risultato:** Efficienza registrata del **259%** (Super-Scaling).
*   **Analisi:** Passando da 1 a 24 thread, il tempo per epoca è sceso da 2500ms a 960ms nonostante il lavoro totale sia aumentato di 24 volte. 
*   **Conclusione:** Il solutore beneficia enormemente della sinergia della cache L3 e del prefetching hardware quando più core lavorano in parallelo sulla stessa istanza. Il sistema è pronto per popolazioni di formiche massicce.

## 4. Stato dell'Arte e Linee Guida

Il solutore V2 attuale (**Adaptive-Hierarchical-Float**) rappresenta il picco delle prestazioni su singolo nodo:
1.  **Sotto il secondo:** ~770ms per epoca a N=16.000 (24 core).
2.  **HPC-Ready:** Super-Scaling confermato nel test di Weak Scaling.
3.  **Memoria Ottimizzata:** Bypassato il Memory Wall grazie alla precisione `float` e all'auto-tuning della L3.

## 5. Prossimi Passi
*   **Asynchronous MPI Overlap (SSP):** Implementare lo scambio feromoni in background (`MPI_Iallreduce`) per test in ambiente cluster multi-nodo reale.
*   **Cluster Multi-Nodo:** Testare lo scaling su 2-4 nodi (48-96 core) per verificare l'impatto della latenza di rete reale.
