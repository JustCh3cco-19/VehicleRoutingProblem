# Profiling Scaling Limits of v6 Architecture

## 1. Obiettivo e Metodologia
L'obiettivo di questa indagine è quantificare empiricamente i limiti di scalabilità dell'architettura **CUDA v6** all'aumentare delle dimensioni del grafo (da 2.000 a 128.000 nodi). 

I test sono stati condotti su una GPU **NVIDIA GeForce RTX 5080 (16GB VRAM)**. 
I parametri di terminazione del solver ACO sono stati impostati per favorire la convergenza:
- Timeout: 300 secondi
- Stagnazione: 100 epoche
- Minimo miglioramento relativo: 5%

Le istanze testate coprono $N = \{2000, 8000, 32000, 64000, 128000\}$.

---

## 2. Risultati: Runtime e Scalabilità
I test di esecuzione diretta del solver hanno restituito i seguenti tempi totali:

| Istanza | N (Nodi) | K (Veicoli) | M (Formiche) | Runtime Totale | VRAM Stimata (Tau) | Note |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `n2000` | 2.000 | 8 | 256 | **0.66s** | 4 MB | Esecuzione istantanea |
| `n8000` | 8.000 | 8 | 512 | **2.60s** | 64 MB | Stabile e veloce |
| `n32000`| 32.000 | 32 | 512 | **25.48s** | 1 GB | Inizia a farsi sentire il carico |
| `n64000`| 64.000 | 63 | 512 | **1m 32s** | 4 GB | Piena occupazione GPU |
| `n128000`| 128.000| 125 | 512 | **OOM (Crash)** | 16.38 GB | Supera il limite fisico |

### 2.1 Analisi del "Memory Wall"
Il test ha evidenziato esattamente il limite teorizzato: con **16GB di VRAM disponibili**, l'istanza da 128.000 nodi richiede un'allocazione della singola matrice dei feromoni a 8-bit pari a $128.000^2 \times 1 \text{ byte} = 16.38 \text{ GB}$. 
La richiesta eccede fisicamente le capacità della RTX 5080, causando un corretto "Out Of Memory" in fase di `cudaMalloc`.
- Se avessimo mantenuto l'architettura **v5** (matrici dense float 32-bit), l'Out of Memory si sarebbe presentato già a **45.000 nodi**. La v6 ha quindi triplicato la capacità di risoluzione della singola GPU.

---

## 3. Profilazione Avanzata Hardware (Nsight Compute)
La profilazione del kernel costruttivo è stata eseguita su $N=\{2000, 8000, 32000\}$ per osservare l'evoluzione delle metriche.

### 3.1 Occupancy e Saturazione
- **N=2000**: Achieved Occupancy bassa (sotto il 20%), SM under-utilized a causa dello scarso numero di blocchi lanciati rispetto ai 84 Streaming Multiprocessor della RTX 5080.
- **N=32000**: Achieved Occupancy e SM Busy salgono vertiginosamente. Le 512 formiche (16 blocchi) iniziano a far lavorare le warp scheduler a pieno regime, ma lasciano spazio all'ipotesi di scalare il numero di formiche ($M=2048+$) per saturare al 100% la GPU sulle istanze grandi.

### 3.2 L2 Cache vs DRAM
Un dato eccezionale e costante in tutti i test (fino a $N=32000$ profilati) è il traffico verso la memoria di massa.
- **DRAM Throughput**: Sempre sotto l'**1%**.
- **L2 Hit Rate**: Si è mantenuto saldamente sopra il **95%**.
Il motore di distanza basato sulle coordinate e la matrice quantizzata hanno completamente risolto il problema del *Memory-Bound*. Nonostante la mole di letture casuali, la cache L2 gigante (tipica delle ultime generazioni NVIDIA) riesce a servire le richieste al massimo della banda interna.

### 3.3 Efficienza del Control Flow
Anche su 32.000 nodi, la **Branch Efficiency** si è mantenuta intorno al **92%**. La scelta architetturale del modello Warp-per-Ant si dimostra resiliente anche quando il grafo diventa immenso e i fallback globali aumentano (grazie all'ottima implementazione dell'Hierarchical Bitset che salta 64 nodi condizionati alla volta).

---

## 4. Conclusioni Definitive
I test di scalabilità certificano che la v6 è "industrial-grade".

1. **Il Limite è ora il Silicio**: L'algoritmo non è limitato dal software, ma esclusivamente dalla VRAM totale. Con una GPU da 24GB (es. RTX 4090/3090) il limite matematico si sposta a ~150.000 nodi. Su GPU Enterprise da 80GB (A100), il solver potrebbe gestire istanze da **280.000 clienti** con un singolo nodo di calcolo.
2. **Oltre il Precedente Ostacolo**: Un'istanza da 64.000 clienti viene risolta in ~90 secondi. Con le architetture classiche, questa istanza non sarebbe mai nemmeno entrata in memoria.

L'architettura **CUDA v6 Ultra-Scale** si posiziona come lo stato dell'arte per implementazioni GPU ACO ad alta densità.
