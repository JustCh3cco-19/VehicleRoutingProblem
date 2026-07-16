# Technical Report: Analysis of Intra-Ant Parallelism Failures in Multicore ACO Solvers

---

## Abstract
Questo report analizza il tentativo di mitigare il collo di bottiglia della fase di "Construction" in un solutore Ant Colony Optimization (ACO) per istanze di grandi dimensioni ($N=16.000$). L'architettura proposta, denominata **V3 (Collaborative Ant Teams)**, mirava a ridurre la latenza della scansione della memoria (Fallback) dividendo il compito tra team di thread sincronizzati. Nonostante tre iterazioni di ottimizzazione (Rigida, Barrier-Free, e Wait-Signal), i risultati sperimentali hanno mostrato un degrado delle prestazioni di ordini di grandezza rispetto al modello a thread indipendenti (V2). L'indagine rivela che l'overhead di sincronizzazione gerarchica e la latenza del kernel superano sistematicamente i benefici del prefetching collaborativo su architetture CPU standard.

---

## 1. Introduction
Nel solutore ACO VRP V2, la fase di costruzione occupa circa il 75% del tempo di calcolo a $N=16.000$. Il colpevole è il **Fallback**, una scansione sequenziale $O(N)$ di righe della matrice distanze (64KB in precisione `float`). Sebbene teoricamente parallelizzabile, questa fase avviene circa 16.000 volte per ogni formica costruita. La V3 è stata progettata per trasformare questo calcolo da seriale a collaborativo, dividendo i 24 thread fisici in 6 team da 4 thread.

---

## 2. Methodology & Architectures

### 2.1 V3.1: Synchronous Team (The Barrier Wall)
L'implementazione iniziale utilizzava `#pragma omp barrier` per coordinare il passaggio tra la scelta probabilistica (Leader) e la scansione del fallback (Team).
*   **Punto di fallimento:** La frequenza di sincronizzazione. Con $1.536 \text{ formiche} \times 16.000 \text{ passi}$, il sistema invocava ~24 milioni di barriere globali per epoca.
*   **Risultato:** Tempo per epoca $\approx 8.4s$ (vs $1.4s$ della V2).

### 2.2 V3.2: Speculative Spinning (The Coherency Storm)
Per eliminare il costo delle barriere software, è stato implementato un sistema di "Generation-based Spinning" basato su variabili atomiche. I Worker monitoravano un contatore incrementato dal Leader.
*   **Punto di fallimento:** **Protocollo MESI**. Lo spinning continuo su variabili condivise ha generato un volume di traffico di coerenza cache tale da saturare il bus di sistema. Questo fenomeno, noto come *cache line bouncing*, ha rallentato il Leader nel caricare i dati reali dalle matrici.
*   **Risultato:** Esecuzione instabile, tempo per epoca $> 5 \text{ minuti}$.

### 2.3 V3.3: On-Demand Wakeup (The Kernel Barrier)
L'ultima iterazione utilizzava `pthread_cond_wait` per mettere i worker in uno stato di stop a consumo zero, svegliandoli solo per il fallback pesante.
*   **Punto di fallimento:** **Kernel Latency**. Il tempo necessario per una *syscall* di wakeup (~5-10 $\mu s$) è risultato comparabile o superiore al tempo necessario per eseguire la scansione seriale di 64KB. I core apparivano sotto-utilizzati in `htop` perché passavano il tempo ad aspettare lo scheduler del sistema operativo.

---

## 3. Technical Analysis of the Failure

### 3.1 Task Granularity Mismatch
La scansione di una riga da 64KB è un task di **"Micro-Granularità"**. Su CPU moderne (3.0 GHz+), il tempo di calcolo puro è estremamente ridotto. Qualsiasi forma di comunicazione inter-thread (barriere, atomici, segnali) ha un costo fisso che non scala con la dimensione del problema fino a valori di $N$ ben superiori a 16.000.

### 3.2 Coordination vs. Computation Ratio
Sia $T_{comp}$ il tempo di scansione seriale e $T_{coord}$ il tempo di coordinamento. Il tempo parallelo è $T_{par} = \frac{T_{comp}}{P} + T_{coord}$. 
Dai nostri test:
*   $T_{comp} \approx 10 \mu s$
*   $T_{coord} \approx 15-20 \mu s$
Poiché $T_{coord} > T_{comp}$, il sistema parallelo sarà sempre più lento del seriale, indipendentemente dal numero di processori $P$.

---

## 4. Conclusion
L'esperimento V3 dimostra che il **parallelismo intra-thread su CPU** per algoritmi stocastici a passo singolo (come l'ACO) è una strategia fallimentare a causa delle limitazioni fisiche della comunicazione tra core. Il modello **1 Formica = 1 Thread** è l'unico in grado di massimizzare il throughput ignorando i costi di coordinamento.

## 5. Future Direction: Intra-Core Collaboration (SIMD)
Per abbattere il muro del fallback senza incorrere in latenze di sincronizzazione, la ricerca deve spostarsi verso la **vettorizzazione SIMD**. 
*   **Vantaggio:** Le unità AVX2/AVX-512 permettono di processare 8-16 elementi contemporaneamente all'interno dello stesso ciclo di clock della stessa ALU.
*   **Risultato atteso:** Riduzione della latenza di fallback di 8x senza alcuna barriera, syscall o traffico di bus, mantenendo la saturazione dei core al 100%.

---
*End of Report*
