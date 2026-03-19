# CUDA Implementation V1: Ant-per-Thread (Baseline)

## 1. Architettura Tecnica
La prima versione dell'algoritmo ACO per VRP su CUDA seguiva un modello di parallelismo a **grana grossa**.

### Strategia di Parallelismo
- **Mappatura**: 1 Formica = 1 Thread CUDA.
- **Kernel Principale**: `construct_solutions_kernel`.
- **Esecuzione**: Ogni thread/formica eseguiva autonomamente l'intero ciclo di costruzione del percorso (VRP con $K$ veicoli) in modo sequenziale.

### Dettagli del Calcolo della Probabilità
Per ogni passo del percorso, il thread eseguiva un ciclo `for` lungo $N$ (numero nodi) per:
1.  Verificare se il nodo era stato visitato (`if (!visited[j])`).
2.  Calcolare la somma dei punteggi di attrazione (denominatore).
3.  Eseguire una scansione cumulativa sequenziale per scegliere il prossimo nodo (Roulette Wheel Selection).

## 2. Limiti Identificati
- **Warp Divergence**: Poiché le formiche prendono decisioni diverse, i thread dello stesso warp finivano il lavoro in tempi molto differenti, lasciando i core GPU inattivi.
- **Latenza di Memoria**: Gli accessi alla memoria globale per leggere la matrice `scores` non erano coordinati (coalesced), causando molti cicli di attesa.
- **Complessità Algoritmica**: Il lavoro di una formica era $O(N^2)$, eseguito interamente da un singolo core.

## 3. Risultati del Benchmark (Baseline)
I test sono stati eseguiti con 1024 formiche e 100 iterazioni.

| N (Problema) | Tempo per Iterazione (ms) | Scaling Relativo |
| :--- | :--- | :--- |
| 50 | 3.27 ms | 1.0x |
| 100 | 4.70 ms | 1.4x |
| 250 | 24.59 ms | 7.5x |
| 500 | 93.38 ms | 28.5x |
| 1000 | 366.23 ms | 112.0x |

**Osservazione**: All'aumentare di $N$, il tempo cresce in modo quasi esponenziale, rendendo l'algoritmo inadatto a problemi di grandi dimensioni.
