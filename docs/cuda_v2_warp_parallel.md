# CUDA Implementation V2: Ant-per-Warp (Optimized)

## 1. Evoluzione Architetturale
La seconda versione introduce un cambiamento radicale nel modello di esecuzione, passando da un modello "Ant-per-Thread" a un modello **"Ant-per-Warp"**.

### Innovazioni Tecniche
- **Collaborazione Warp**: Ogni formica è ora gestita da un **Warp (32 thread)** che lavorano in modo sincrono e coordinato.
- **Riduzione Parallela (Warp Reduction)**: Il calcolo del denominatore (somma dei punteggi) è stato parallelizzato usando le istruzioni intrinseche `__shfl_down_sync`. Questo riduce la complessità del calcolo del denominatore da $O(N)$ a $O(N/32)$.
- **Memoria Coalesced**: I thread del warp leggono i punteggi di attrazione in blocchi contigui di 32 elementi, massimizzando il throughput della memoria della GPU.
- **Inizializzazione Parallela**: Anche il reset delle strutture dati locali (come l'array `visited`) è stato distribuito sui 32 thread.

## 2. Benefici Ottenuti
- **Riduzione Latenza**: Una significativa riduzione del tempo speso in attesa della memoria globale.
- **Miglior Scaling**: La curva di crescita del tempo di esecuzione è meno ripida rispetto alla versione baseline, specialmente per $N > 500$.
- **Uso Efficiente dei Core**: Ogni core (thread) ha meno carico sequenziale, portando a una maggiore efficienza complessiva.

## 3. Risultati del Benchmark (Warp-Parallel)
Confronto con la Versione 1 (1024 formiche, 100 iterazioni).

| N (Problema) | Tempo V2 (ms) | Tempo V1 (ms) | Miglioramento (%) |
| :--- | :--- | :--- | :--- |
| 50 | 3.22 ms | 3.27 ms | ~2% |
| 100 | 4.53 ms | 4.70 ms | ~4% |
| 250 | 17.89 ms | 24.59 ms | **27%** |
| 500 | 57.46 ms | 93.38 ms | **38%** |
| 1000 | 220.74 ms | 366.23 ms | **40%** |

**Conclusione**: La versione V2 è fino al 40% più veloce della baseline. Lo scaling è migliorato, ma c'è ancora margine di ottimizzazione parallelizzando la scansione cumulativa per la scelta del nodo.
