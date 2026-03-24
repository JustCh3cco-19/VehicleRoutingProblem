# CUDA Implementation V3: Warp-Scan & GPU-Deposit (Best Possible)

## 1. Architettura ad Alte Prestazioni
La terza versione rappresenta l'apice dell'ottimizzazione per questa architettura, risolvendo tutti i colli di bottiglia identificati nelle versioni precedenti.

### Innovazioni Tecniche Definitive
- **Parallel Choice (Warp-Scan)**: La selezione del nodo è ora completamente parallelizzata. Usiamo un **Parallel Prefix Sum** a livello di Warp per gestire la roulette wheel selection. Questo ha trasformato il collo di bottiglia $O(N)$ in un'operazione $O(N/32)$ estremamente efficiente.
- **Voto di Gruppo (`__ballot_sync` & `__ffs`)**: Per decidere quale nodo scegliere, il warp esegue un "voto" elettronico immediato per individuare il thread vincitore, eliminando ogni forma di ricerca sequenziale.
- **Zero-Copy Pheromone Update**: Il deposito del feromone è stato spostato interamente sulla GPU tramite il kernel `deposit_pheromones_kernel`. Abbiamo eliminato il trasferimento di 8MB di dati via PCIe (Host-Device) che avveniva in ogni singola iterazione.
- **Atomic Operations**: Usiamo `atomicAdd` sulla memoria della GPU per gestire i contributi dei feromoni in modo sicuro e veloce.

## 2. Analisi dello Scaling
Grazie a queste modifiche, l'algoritmo mostra finalmente uno **scaling quasi lineare**. La GPU non è più limitata dalla logica di controllo sequenziale, ma può sprigionare tutta la sua potenza di calcolo parallelo.

## 3. Risultati Finali e Confronto
Benchmark eseguito con 1024 formiche e 100 iterazioni.

| N (Problema) | Tempo V3 (ms) | Tempo V2 (ms) | Tempo V1 (ms) | Miglioramento vs V1 |
| :--- | :--- | :--- | :--- | :--- |
| 50 | 3.51 ms | 3.22 ms | 3.27 ms | - |
| 100 | 2.19 ms | 4.53 ms | 4.70 ms | 2.1x |
| 250 | 4.21 ms | 17.89 ms | 24.59 ms | 5.8x |
| 500 | 7.69 ms | 57.46 ms | 93.38 ms | 12.1x |
| 1000 | **21.43 ms** | 220.74 ms | 366.23 ms | **17.1x** |

## 4. Limitazioni Identificate: Il Bug della Doppia Visita (Warp Synchronization)

Durante test intensivi (512+ formiche, 2000+ iterazioni), è emerso un bug critico di correttezza legato alla parallelizzazione a livello di warp.

### Analisi Tecnica del Problema
Il solver CUDA può occasionalmente produrre soluzioni non valide in cui uno o più clienti vengono visitati più volte (es. `Route 1: ... 8 1 1`).

**Root Cause (Analisi in Profondità):**
Il problema risiede nella **mancata sincronizzazione della memoria globale** all'interno del warp tra le iterazioni del ciclo di costruzione del percorso (`while`).
1. Il thread 0 del warp seleziona il `next_node` e aggiorna l'array `my_visited` in memoria globale: `my_visited[next_node] = true`.
2. Il valore di `next_node` viene propagato agli altri thread tramite `__shfl_sync`.
3. Tuttavia, la scrittura in memoria globale di `my_visited` **non è garantita come visibile** a tutti gli altri thread del warp all'inizio dell'iterazione successiva, quando viene ricalcolato il denominatore (`denom`) e viene eseguito il parallel scan.
4. Se un thread legge `false` per un nodo già visitato, quel nodo rientra nel calcolo delle probabilità e può essere selezionato nuovamente.

**Conseguenze:**
- Soluzioni non fattibili (`Infeasible`) secondo i vincoli VRP.
- Fallimento della validazione tramite `pyvrp`.
- Instabilità dei risultati con carichi di lavoro elevati (molte formiche/iterazioni).

### Strategia di Risoluzione (In Fase di Implementazione)
Per garantire la correttezza atomica delle visite, sono necessarie le seguenti modifiche al kernel `construct_solutions_warp_v3_kernel`:
1. **`__syncwarp()` forzato**: Inserire una barriera di sincronizzazione dopo ogni aggiornamento dell'array `visited`.
2. **Shared Memory per `visited`**: Spostare l'array `visited` (attualmente in memoria globale per ogni formica) nella **Shared Memory** del blocco per ridurre la latenza e sfruttare la coerenza automatica dei dati a livello di warp/blocco.
3. **Sincronizzazione Fallback**: Assicurarsi che le sezioni di "fallback" (quando `denom <= 0`) siano eseguite in modo atomico o protetto da sincronizzazioni warp-level.

Questa scoperta evidenzia che, sebbene la Versione 3 sia estremamente veloce, richiede una gestione più rigorosa della coerenza della memoria per essere considerata affidabile in produzione.
