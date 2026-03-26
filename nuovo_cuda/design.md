# Design del Nuovo Solver CUDA VRP

## 1. Analisi del Problema e Vincoli Architetturali

L'obiettivo di questa nuova implementazione è superare l'approccio generico delle versioni precedenti, integrando esplicitamente la natura del Vehicle Routing Problem (VRP) con le caratteristiche fisiche dell'architettura GPU.

### Dinamica dei Parametri di Input
A differenza delle versioni passate, il design si basa su una gerarchia chiara tra i parametri del problema:

*   **Clienti ($N$):** Il carico computazionale principale. Assumiamo che $N$ sia sempre almeno un ordine di grandezza superiore al numero di veicoli ($N \gg K$). Questo rapporto deve guidare la strategia di decomposizione del lavoro tra i thread.
*   **Veicoli ($K$):** Rappresentano il numero di rotte indipendenti da costruire. Essendo $K \ll N$, la gestione dei veicoli deve essere ottimizzata per evitare overhead di sincronizzazione eccessivi.

### Parametri Interni (Controllo dell'Algoritmo)
Il solver deve imparare a gestire dinamicamente o in modo ottimizzato i parametri che governano il comportamento dell'Ant Colony Optimization (ACO):

*   **Formiche ($m$):** Non è più un parametro di input. Viene derivato automaticamente dall'hardware per saturare gli Streaming Multiprocessors (SM) e massimizzare l'occupancy (es. $m = Numero\_SM \times 32$).
*   **Iterazioni ($T$):** Sostituite da un budget temporale e da criteri di convergenza dinamici.

## 2. Vincoli Specifici del Problema

Per questa specifica implementazione, il problema adotta una configurazione semplificata ma rigorosa per quanto riguarda il carico:

*   **Domanda Unitaria ($d_i = 1$):** Ogni cliente (nodo $i \in \{1 \dots N\}$) richiede esattamente 1 unità di capacità. Il deposito (nodo 0) ha domanda 0.
*   **Capacità Omogenea dei Veicoli ($C$):** Ogni veicolo ha la stessa capacità massima, calcolata dinamicamente come:
    $$C = N - K + 3$$
    Dove $N$ è il numero di clienti e $K$ il numero di veicoli.

### Impatto sull'Implementazione Kernel
Questa configurazione permette ottimizzazioni significative:
1.  **Riduzione Traffico Memoria**: Non è necessario caricare un array `demands[N]` dalla Global Memory. Il valore "1" è trattato come un letterale (immediato) nelle istruzioni di calcolo.
2.  **Semplificazione Stato Veicolo**: Il controllo della capacità residua nel kernel `Construct_Solutions` diventa un semplice confronto tra il numero di nodi visitati dal veicolo attuale e la costante $C$.
3.  **Bilanciamento del Carico**: Con $C = N - K + 3$, i veicoli hanno una capacità totale combinata ($K \cdot C$) che supera di poco il numero totale di clienti $N$ (specialmente per $K$ piccoli), forzando l'algoritmo a cercare rotte dense e ottimizzate.

## 3. Struttura Logica (Pseudo-codice)

L'algoritmo si articola in una fase di setup e un ciclo iterativo dove la maggior parte del lavoro viene delegato alla GPU.

### A. Inizializzazione (Host & Device)
1. Caricamento dati e allocazione memoria (Matrice dei costi, feromone, visibilità).
2. **Pre-processing GPU (K1)**:
   * Inizializzazione stati casuali (uno per ogni formica).
   * Calcolo `eta = 1/costo`.
   * **Inizializzazione Liste Candidati**: Setup statico basato sulla distanza geografica per la prima iterazione.

### B. Ciclo Principale (Iterazioni)
Per ogni iterazione $t$:

1. **Selezione Dinamica dei Candidati (K2)**:
   * Per ogni nodo sorgente, identifica i 32 nodi con score $S_{ij} = \tau_{ij}^\alpha \cdot \eta_{ij}^\beta$ più alto tramite riduzione parallela nel Warp.
   * *Ottimizzazione*: Ricalcolato ad ogni iterazione per adattarsi all'evoluzione del feromone ("scie" dinamiche).

2. **Costruzione Soluzioni (K4 - Kernel Core)**:
   Ogni formica (mappata su un Warp) esegue:
   * **Setup**: Pulisce la Bitmask dei visitati in Shared Memory.
   * **Loop di Visita ($N$ passi)**:
     * Determinazione veicolo attivo ($v = 0 \dots K-1$).
     * **Selezione Prossimo Nodo**:
       1. Consulta la **Lista Candidati Dinamica** (top 32 per score).
       2. Calcola le probabilità pesate usando $S_{ij}$ pre-calcolati.
       3. Se nessun candidato è libero, esegue un **Fallback Globale Stocastico**: scansiona tutti i nodi, ricalcola $S_{ij}$ al volo e applica una Roulette Wheel parallela a livello di Warp (**Compute is free**).
     * **Aggiornamento Stato**:
       * Marca il nodo come visitato nella Bitmask.
       * Aggiorna la rotta del veicolo $v$ e il costo parziale.
       * Passa al veicolo successivo ($v = (v+1) \bmod K$).
   * **Chiusura**: Ritorno forzato al deposito per tutti i veicoli e calcolo del costo totale.

3. **Valutazione e Riduzione**:
   * Trova la formica migliore dell'iterazione (Riduzione parallela).
   * Se migliore del record globale, aggiorna `BestSolution`.

4. **Aggiornamento Feromone (K5)**:
   * **Evaporazione**: Applica $\tau = \tau \cdot (1-\rho)$ su tutta la matrice.
   * **Deposito**: La formica migliore deposita feromone sugli archi percorsi ($Q/L$).

### C. Finalizzazione
1. Recupero della `BestSolution` dall'host.
2. Deallocazione risorse.

## 4. Principi di Ottimizzazione Logica

Il design del nuovo solver si basa sulla massimizzazione dell'efficienza della GPU attraverso due direttrici principali:

### A. Massimizzazione del Throughput (Formiche)
Dato che la latenza di un'iterazione rimane pressoché costante fino alla saturazione degli Streaming Multiprocessor (SM), la strategia è:
*   **Saturazione Hardware**: Identificare il numero ottimale di formiche ($m$) che garantisce la massima *Occupancy*. L'obiettivo è avere abbastanza formiche da nascondere le latenze di accesso alla memoria globale.
*   **Autotuning**: $m$ è calcolato in base alle proprietà del device (SM count) per garantire scalabilità su diverse architetture.

### B. Riduzione del Carico per Iterazione
Per permettere un elevato numero di iterazioni, fondamentale per la convergenza del feromone, ogni iterazione deve essere logica e snella:
*   **Esecuzione Parallela**: Sfruttare la natura indipendente delle formiche per eseguire la costruzione delle soluzioni simultaneamente.
*   **Feedback Rapido**: Ridurre al minimo le dipendenze e le sincronizzazioni non necessarie per mantenere il ciclo di aggiornamento del feromone il più frequente possibile.

## 5. Unità Logica di Esecuzione: Formica = Warp

La scelta del livello di astrazione per la singola formica è il pilastro del design hardware-aware. Abbiamo optato per il modello **Formica = Warp (32 thread)** invece del modello **Formica = Blocco**.

### Motivazioni della Scelta "Warp-Centric"

1.  **Efficienza della Sincronizzazione**:
    *   **Warp**: Utilizza primitive di registro (`__shfl_sync`, `__shfl_down_sync`) per scambiare dati e calcolare riduzioni (es. somma dei pesi per la roulette wheel). Queste istruzioni hanno latenze di pochissimi cicli di clock.
    *   **Blocco**: Richiederebbe l'uso costante di `__syncthreads()`. In un algoritmo con $N$ passi decisionali (es. 10.000), l'overhead di migliaia di barriere hardware svuoterebbe le pipeline di calcolo, rallentando drasticamente l'esecuzione.

2.  **Massimizzazione dell'Occupancy**:
    *   Un Warp è l'unità minima di scheduling della GPU. Molti Warp indipendenti (formiche) possono risiedere sullo stesso Streaming Multiprocessor (SM). 
    *   Se una formica (Warp) resta in attesa di un dato dalla Global Memory, lo scheduler può istantaneamente far subentrare un'altra formica, nascondendo la latenza della DRAM. Un modello a blocchi ridurrebbe il numero di unità "in volo", esponendo l'SM a lunghi tempi di inattività.

3.  **Gestione delle Risorse (Registri e Shared Memory)**:
    *   Assegnare un intero blocco a una formica aumenterebbe la pressione sui registri per thread, causando un "performance cliff".
    *   Il modello Warp permette di distribuire meglio il carico, mantenendo ogni unità "leggera" e veloce.

4.  **Assenza di Divergenza tra Warp**:
    *   Ogni formica (Warp) corre alla sua velocità. Se una formica finisce prima la sua rotta, può procedere o terminare senza dover aspettare le altre (cosa obbligatoria in un blocco a causa delle barriere).

### Gestione del Parallelismo Interno
Nonostante la formica sia un Warp, i 32 thread cooperano per:
*   **Lettura Coalescente**: Caricare simultaneamente 32 candidati o 32 score dalla memoria.
*   **Parallelismo SIMT**: Valutare in un unico ciclo di clock lo stato di visita di 32 nodi diversi.
*   **Riduzione Veloce**: Calcolare la probabilità totale di transizione in $O(\log_2 32)$ passi di shuffle.

## 6. Gestione delle Risorse e Occupancy

Per istanze di larga scala ($N=20.000$, centinaia di veicoli $K$), il kernel di costruzione delle soluzioni rischia di essere limitato dal numero di registri (*Register-Heavy*), causando un crollo dell'occupancy e delle prestazioni. Il design adotta due strategie di "scarico" per mantenere l'efficienza elevata.

### A. Trade-off: Shared Memory vs Registri
Per ridurre la pressione sui registri e permettere a più Warp (formiche) di risiedere simultaneamente su ogni SM, spostiamo lo stato dinamico del problema nella **Shared Memory**:

*   **Bitmask dei Visitati**: Allocata in Shared Memory per ogni Warp ($\approx 2,5$ KB per 20.000 nodi). Consente lookup e update in $O(1)$ con latenza minima.
*   **Stato della Flotta ($K$)**: Le informazioni sui veicoli (nodo attuale, capacità residua, costo parziale) vengono spostate dai registri alla Shared Memory ($\approx 1,5$ KB per $K=128$).
*   **Obiettivo Tecnico**: Ridurre il consumo di registri da circa 80 a **40 per thread**, puntando a un'occupancy del **70-80%**. Il leggero aumento della latenza della Shared Memory è ampiamente compensato dal raddoppio delle formiche "in volo" che nascondono le latenze della memoria globale.

### B. Buffer dei Numeri Casuali (Pre-computing)
L'integrazione di `curand` nel kernel principale è onerosa ($\approx 15$-$20$ registri per thread). Il design prevede la separazione della generazione stocastica:

1.  **Kernel "Random Gen" (K3)**: Un kernel leggero genera un buffer di numeri `float` casuali in Global Memory prima della fase di costruzione.
2.  **Consumo di Memoria**: Per 1024 formiche e 20.000 passi, il buffer occupa $\approx 80$ MB, un impatto trascurabile sulla VRAM moderna.
3.  **Vantaggio**: Il kernel `Construct_Solutions` legge semplicemente i valori pre-calcolati. L'accesso sequenziale favorisce il caching in L2, eliminando il costo computazionale e di registri della generazione *on-the-fly*.

### C. Riepilogo Allocazione Memoria ($N=20.000$)

| Risorsa | Tipo | Dimensione Stimata | Posizionamento |
| :--- | :--- | :--- | :--- |
| **Matrice Score** | `float` | 1,6 GB | Global Memory (L2 Cached) |
| **Matrice Costi** | `float` | 1,6 GB | Global Memory |
| **Random Buffer** | `float` | 80 MB | Global Memory |
| **Bitmask** | `uint32` | 2,5 KB / ant | Shared Memory |
| **Stato Veicoli** | `mixed` | 1,5 KB / ant | Shared Memory |
| **Parametri ACO** | `double` | < 1 KB | Constant Memory |

## 7. Architettura dei Kernel (Pseudo-codice Hardware-Aware)

L'implementazione si articola in cinque kernel coordinati dall'Host, progettati per minimizzare i viaggi in DRAM e massimizzare l'uso della Cache L2 e dei registri.

### K1: `Initialize_ACO` (Setup Statico)
*Eseguito una sola volta. Prepara le strutture dati "L2-friendly".*
*   **Grid**: $N+1$ blocchi (uno per ogni nodo $i$).
*   **Logica**: Ogni blocco identifica i 32 vicini più prossimi (Candidate List) per il nodo $i$.
*   **Output**: `cand_list[N+1][32]` e `cand_eta[N+1][32]`.

### K2: `Precompute_Candidate_Scores` (Dynamic Selection)
*Eseguito all'inizio di ogni iterazione per aggiornare il vicinato promettente.*
*   **Grid**: $N+1$ blocchi.
*   **Logica**: Identifica i 32 nodi con score $\tau^\alpha \cdot \eta^\beta$ più alto tramite riduzione parallela nel Warp.
*   **Vantaggio**: Permette alle formiche di seguire "scie" di feromone anche su archi non inizialmente vicini geograficamente.

### K3: `Generate_Randoms`
*Eseguito una volta per iterazione.*
*   **Logica**: Popola un buffer globale di numeri `float` casuali per eliminare il costo computazionale e di registri di `curand` dal kernel core.

### K4: `Construct_Solutions` (Il Motore Warp-Centric)
*Il cuore del sistema: 1 Warp = 1 Formica.*
*   **Setup**: Pulisce la Bitmask e inizializza lo Stato Flotta in **Shared Memory**.
*   **Loop di Costruzione ($N$ passi)**:
    1.  **Lettura Stato**: Recupera il veicolo attivo dalla Shared Memory (Broadcast).
    2.  **Caricamento Candidati**: I 32 thread leggono i 32 score dei candidati dinamici (Hit in L2 Cache).
    3.  **Filtraggio**: Applica la Bitmask (Shared Memory) per azzerare i nodi già visitati.
    4.  **Roulette Wheel**: Calcola la probabilità totale e sceglie il vincitore tramite **Warp Shuffle** (Registri).
    5.  **Global Fallback Stocastico**: Se i 32 candidati sono occupati, esegue una scansione completa, ricalcola lo score `powf` on-the-fly e seleziona il nodo via Roulette Wheel parallela (**Compute is free**).
    6.  **Aggiornamento**: Scrive il vincitore nella Bitmask e nello Stato Veicolo (Shared Memory).
    7.  **Output**: Buffering dei nodi scelti e scrittura in Global Memory.

### K5: `Pheromone_Evolution`
*Kernel separati per il throughput di memoria.*
*   **Evaporazione**: Kernel lineare e coalescente su tutta la matrice $\tau$.
*   **Deposito**: La formica migliore dell'iterazione deposita feromone tramite `atomicAdd` sugli archi percorsi (Scatter controllato).

## 8. Strategia della Grid e Occupancy Target

La configurazione della Grid (blocchi e thread) è dettata dai limiti fisici degli Streaming Multiprocessor (SM) per massimizzare il parallelismo.

### K4: `Construct_Solutions` (Limitato da Shared Memory)
*   **Configurazione**: 128 thread per blocco (4 Warp/Formiche per blocco).
*   **Uso Risorse**: Ogni Warp occupa ~4 KB di Shared Memory. Un blocco occupa quindi ~16 KB.
*   **Occupancy Target**: Puntiamo a circa **24 Warp attivi per SM**. 
    *   Con ~96 KB di Shared Memory occupata (6 blocchi da 128 thread), saturiamo la capacità della memoria veloce dell'SM, garantendo che ci siano sempre abbastanza formiche "in volo" per nascondere le latenze della DRAM.
*   **Numero Formiche ($m$)**: Calcolato dinamicamente come `Numero_SM * 32`.

### K2: `Precompute_Candidate_Scores` (Limitato da Throughput)
*   **Configurazione**: 128 thread per blocco.
*   **Grid**: $N+1$ blocks (uno per ogni nodo).
*   **Logica**: Blocchi che permettono allo scheduler di distribuire il carico uniformemente su tutti gli SM, saturando la banda verso la Cache L2.

### K5: `Pheromone_Evolution` (Limitato da Banda Memoria)
*   **Configurazione**: 512 o 1024 thread per blocco.
*   **Logica**: Blocchi grandi per massimizzare la coalescenza delle letture/scritture lineari sulla matrice $\tau$, sfruttando l'intera ampiezza del bus di memoria.

## 9. Controllo di Flusso e Terminazione

Il solver adotta una strategia di terminazione ibrida per bilanciare qualità e tempo:

*   **Timer Host (Hard Limit)**: Il solver riceve un `timeout_minutes`. Se il tempo trascorso supera questo limite, il solver termina immediatamente dopo l'iterazione corrente e restituisce la migliore soluzione trovata.
*   **Early Stoppage (Soft Limit)**: Ispirato al training delle reti neurali, il solver monitora il miglioramento relativo della soluzione migliore globale:
    *   **Pazienza**: Fissata a **100 iterazioni**.
    *   **Soglia di Miglioramento**: Parametro di input (`improvement_threshold`, es. 0.001 per lo 0.1%).
    *   **Logica**: Se per 100 iterazioni consecutive il miglioramento relativo è inferiore della soglia, il solver termina anticipatamente per convergenza.

---
**Conclusione del Design**: Questa architettura trasforma il VRP in un flusso di dati altamente ottimizzato, dove la gerarchia della memoria GPU e le logiche di controllo dinamiche collaborano per massimizzare il rendimento hardware.
