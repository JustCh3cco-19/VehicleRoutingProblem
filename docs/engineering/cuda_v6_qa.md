# CUDA v6 Architecture: Technical Q&A

### Q1: Quali valori allochiamo nei registri di ciascun thread? A che occupancy teorica ci portano? È un dato soddisfacente?

**Risposta Tecnico-Architetturale:**

Nella versione v6, ogni thread utilizza esattamente **48 registri**. Questa allocazione non è casuale ma deriva dalla necessità di mantenere l'intero stato operativo dell'algoritmo ACO "on-chip", evitando il ricorso alla memoria lenta.

**Valori allocati nei registri:**
*   **Puntatori a 64-bit**: Per gestire la matrice dei feromoni da 10GB, ogni puntatore occupa 2 registri (necessari per indirizzamento sopra i 4GB).
*   **Stato RNG (Random Number Generator)**: Fondamentale per la natura stocastica delle formiche.
*   **Coordinate Temporanee**: Caricamento dei dati `float2` per il calcolo della distanza Euclidea on-the-fly.
*   **Variabili di Stato Warp-per-Ant**: Score dei candidati, indici di bit-manipulation per l'Hierarchical Bitset e accumulatori per la riduzione parallela (`__shfl_sync`).

**Calcolo dell'Occupancy Teorica:**
Le moderne architetture NVIDIA (come Ampere/Ada) hanno un limite di **65.536 registri** per Streaming Multiprocessor (SM).
1.  **Thread per SM**: $65.536 / 48 = \mathbf{1.365 \text{ thread}}$.
2.  **Warp per SM**: $1.365 / 32 \approx \mathbf{42 \text{ warp}}$.
3.  **Occupancy Teorica**: Dato che il limite fisico è di 64 warp per SM, il nostro valore è $42 / 64 \approx \mathbf{66\%}$.

**Valutazione del dato:**
Sulla base dei principi descritti nel notebook **'Multicore'**, il dato è **altamente soddisfacente** per tre ragioni:
1.  **Latency Hiding**: 42 warp attivi sono ampiamente sufficienti per nascondere la latenza di accesso alla memoria globale (solitamente 400-600 cicli di clock).
2.  **Zero Register Spilling**: Rimanere sotto la soglia critica dei 64 registri evita che i dati finiscano nella memoria locale (DRAM), mantenendo le performance al massimo livello.
3.  **Natura Compute-Bound**: Essendo la v6 focalizzata sul calcolo delle distanze on-the-fly, l'algoritmo non beneficia significativamente di un'occupancy superiore al 60-70%. Incrementarla ulteriormente porterebbe a rendimenti decrescenti e potenziale contesa nelle cache L1/L2.

---

### Q2: Come abbiamo rimaneggiato il sistema di scrittura delle routes? Dove lo allochiamo, perché non allocarlo in maniera privata ad un warp e poi condividerlo?

**Risposta Tecnico-Architetturale:**

Nella versione v6, abbiamo abbandonato il layout classico per formica (**AoS**) in favore di un layout **Step-Interleaved SoA** (Structure of Arrays) in **Memoria Globale**.

**Organizzazione della memoria:**
Invece di `routes[ant_id][step]`, usiamo `routes[step][total_ants]`. Questo significa che tutti i nodi scelti dalle formiche al passo $t$ sono memorizzati in un blocco contiguo di memoria.

**Perché questa scelta? (Memory Coalescing):**
Nel modello Warp-per-Ant, al passo $t$, il Lane 0 di ogni Warp scrive il nodo scelto. Con il layout interleaved, 32 Lane 0 di 32 warp diversi scrivono in indirizzi **contigui**. La GPU accorpa queste 32 scritture in un'unica transazione da 128 byte. In v5, questo avrebbe richiesto 32 transazioni separate, saturando inutilmente la larghezza di banda.

**Perché allocarlo in Global Memory e non privatamente al Warp (Shared Memory)?**
Si potrebbe pensare di usare la **Shared Memory** (privata al blocco/warp) come buffer temporaneo e poi copiare tutto in Global Memory alla fine. Tuttavia, abbiamo scartato questa ipotesi per tre ragioni critiche descritte nel notebook **'Multicore'**:

1.  **Limiti di Capacità**: Per $N=100.000$, un percorso può essere molto lungo. La Shared Memory è una risorsa scarsissima (poche decine di KB per SM). Non potremmo mai farci stare i percorsi di migliaia di formiche senza limitare drasticamente il numero di warp attivi (abbattendo l'Occupancy).
2.  **Latenza di Sincronizzazione**: Scrivere in Shared Memory richiede spesso barriere di sincronizzazione (`__syncthreads()`) per evitare race conditions tra i thread che collaborano. Scrivere direttamente in Global Memory con layout SoA è "fire-and-forget" e non richiede pause nel flusso di esecuzione del warp.
3.  **Throughput vs Latency**: Grazie al calcolo delle distanze on-the-fly, abbiamo liberato quasi il 100% della banda della memoria globale. Usare la Shared Memory per "risparmiare" banda che è già ampiamente disponibile sarebbe un'ottimizzazione prematura e controproducente (spreco di cicli di clock per la gestione del buffer).

**In sintesi**: La Global Memory con layout SoA-Interleaved è la soluzione più scalabile. Sfrutta l'hardware di coalescing della GPU per rendere la scrittura massiva quasi istantanea, lasciando la preziosa Shared Memory libera per non ostacolare l'Occupancy dei registri.

---

### Q3: Cosa rappresenta esattamente lo "step" nel layout `routes[step][total_ants]`? Come scriviamo e leggiamo in questa struttura piatta senza mischiare le formiche?

**Risposta Tecnico-Architetturale:**

Lo **step** non è legato all'indice del veicolo, ma rappresenta la **progressione temporale sincronizzata** della simulazione. Immagina che tutte le formiche agiscano al battito di un metronomo globale: ad ogni "battito" (step), ogni formica compie un movimento.

**Logica di Scrittura (La "Corsia" riservata):**
Usiamo una struttura piatta in Memoria Globale organizzata come una matrice dove le **colonne sono le formiche** e le **righe sono i passi temporali**. Il segreto per non mischiare i dati è la formula dell'indice:
`index = (step * total_ants) + ant_id`

Se abbiamo 256 formiche, la **Formica 5** scriverà sempre e solo negli indici `5`, `261` (5 + 256), `517` (261 + 256), e così via. Ogni formica ha una "corsia verticale" virtuale nella memoria che nessun altro Warp può toccare.

**Gestione dei Veicoli:**
Non usiamo strutture separate per i veicoli. La rotta di una formica è una sequenza continua di nodi dove i veicoli sono separati dal **Nodo 0 (Deposito)**.
*Esempio*: Se la Formica 5 visita i clienti 10 e 20 con il primo veicolo, e poi il cliente 30 con il secondo, la sua corsia conterrà: `[10, 20, 0, 30, 0, ...]`.

**Logica di Lettura (Il "Carotaggio"):**
Per recuperare la soluzione completa di una specifica formica (es. la migliore dell'iterazione), l'Host deve eseguire un "carotaggio" verticale, leggendo i dati a salti:

```c
int ant_idx = 5; // Formica target
for (int t = 0; t < max_steps; t++) {
    int nodo = d_routes[t * total_ants + ant_idx];
    // Elabora il nodo...
    if (nodo == 0 && fine_simulazione) break;
}
```

**Perché questo layout è superiore?**
1.  **Scrittura Coalesced (GPU)**: Ad ogni step $T$, 32 warp scrivono i loro nodi in 32 celle contigue. La GPU invia un unico ordine di scrittura da 128 byte invece di 32 ordini sparsi.
2.  **Lettura Efficiente (Deposito)**: Durante il rinforzo dei feromoni, il warp legge i percorsi di 32 formiche contemporaneamente. Poiché i dati dello step $T$ sono vicini, la lettura è massimamente veloce.
3.  **Zero Frammentazione**: Non dobbiamo pre-allocare spazio massimo per ogni veicolo (che hanno lunghezze variabili), eliminando sprechi di VRAM.

**In sintesi**: Sacrifichiamo la contiguità del "diario" della singola formica per ottenere la contiguità degli accessi hardware della GPU. Il costo di lettura a "salti" per l'Host è irrilevante rispetto al guadagno di throughput del 97% durante la simulazione.

---

### Q4: Il layout SoA funziona se un Warp completa una formica e passa subito alla successiva per bilanciare il carico?

**Risposta Tecnico-Architetturale:**

Tecnicamente sì, ma con una **penalità prestazionale massiva** che renderebbe inutile l'uso del SoA. 

**Il legame tra Tempo e Spazio:**
Il layout SoA-Interleaved si basa sulla sincronizzazione temporale: "Allo step $T$, tutti i Warp scrivono nella riga $T$". Se il Warp 1 finisse la sua formica e ne iniziasse una nuova, il suo "nuovo Step 1" accadrebbe mentre gli altri Warp sono magari allo "Step 600".
In questo scenario, i Warp scriverebbero in indirizzi di memoria molto distanti tra loro. La GPU perderebbe la capacità di fare **Memory Coalescing**, trasformando ogni scrittura in una transazione lenta e separata.

**La scelta della v6 (Static Mapping):**
Nella v6 abbiamo optato per un mapping statico: **1 Warp = 1 Formica per l'intera iterazione**.
Sebbene questo possa creare dei piccoli tempi morti alla fine dell'iterazione (alcune formiche finiscono qualche istante prima di altre), i vantaggi superano di gran lunga gli svantaggi:
1.  **Memory Coalescing Garantito**: Finché i warp lavorano sulla stessa iterazione, le loro scritture sono sempre contigue.
2.  **Semplicità del Kernel**: Non serve gestire code di task o puntatori dinamici ai percorsi, risparmiando preziosi **registri**.
3.  **Basso Squilibrio (Skew)**: Nel CVRP, i percorsi hanno lunghezze simili. Il "Tail Effect" (warp che aspettano gli ultimi ritardatari) è minimo e non giustifica la perdita di efficienza della memoria.

**In sintesi**: Il layout SoA è un contratto di collaborazione tra i Warp. Per funzionare, richiede che tutti i partecipanti avanzino nello stesso "tempo algoritmico". Rompere questo avanzamento per un load-balancing dinamico distruggerebbe il throughput della memoria globale.

---

### Q5: Come abbiamo allineato le performance della v6 a PyVRP? Cos'è il "Global Best Reinforcement"?

**Risposta Tecnico-Architetturale:**

Inizialmente, la v6 soffriva di una convergenza lenta rispetto a PyVRP e al solver sequenziale. Abbiamo risolto il problema implementando due strategie chiave:

1.  **Global Best Reinforcement (Rank-Based Update)**: 
    Originariamente la v6 rinforzava solo la migliore formica dell'iterazione corrente. Abbiamo introdotto un secondo kernel di deposito che rinforza ad ogni epoca la **migliore soluzione di sempre** (Global Best).
    - **Peso algoritmico**: Usiamo un mix dove il 30% del feromone viene dalla *Iter-Best* e il 70% dalla *Global Best*. Questo spinge le formiche con molta più forza verso le zone "promettenti" del grafo scoperte finora.

2.  **Superamento dello "Stallo di Quantizzazione"**:
    Con matrici a 8-bit, i piccoli depositi di feromone rischiavano di essere arrotondati a zero (es. `quantize(old + 0.001)` restava identico). Abbiamo scalato l'intensità del feromone (`params.Q`) di un fattore **100x**.
    - **Risultato**: Gli incrementi sono ora sufficientemente grandi da superare la soglia di precisione del singolo byte, permettendo alla matrice di "apprendere" attivamente anche su archi lunghi.

---

### Q6: Come gestiamo la selezione probabilistica quando una formica finisce i 32 vicini (Fallback Globale)?

**Risposta Tecnico-Architetturale:**

Quando una formica esaurisce i 32 candidati più vicini ("Fallback Globale"), deve selezionare un nodo tra tutti quelli rimanenti sull'intero grafo. Inizialmente, avevamo implementato una classica "roulette wheel" a due passaggi: il primo per sommare tutti gli score $\tau^\alpha \cdot \eta^\beta$, e il secondo per selezionare il nodo. 

Tuttavia, su grafi da 100.000 nodi, questo doppio passaggio saturava le ALU della GPU calcolando due volte decine di migliaia di radici quadrate (`sqrtf`) e dequantizzazioni (`expf`) per ogni formica, causando un blocco infinito dell'algoritmo.

La soluzione adottata è l'implementazione del **Single-Pass Stochastic Fallback tramite Weighted Reservoir Sampling**.

1.  **Reservoir Sampling**: Questo algoritmo stocastico permette di effettuare una selezione pesata (roulette wheel) perfetta esaminando i dati in uno "stream", **in un solo passaggio**.
    *   Ogni thread mantiene una `local_sum` e un `local_selected_node`.
    *   Per ogni nuovo nodo valido trovato, calcola il peso $W$ e aggiorna la somma: `local_sum += W`.
    *   Tira un dado $R \in [0, 1]$. Se $R \le \frac{W}{\text{local\_sum}}$, il nuovo nodo sostituisce il `local_selected_node`.
    *   Matematicamente, alla fine del passaggio, ogni nodo ha esattamente la probabilità $W / \text{SommaTotale}$ di essere scelto, rispettando perfettamente la formula ACO senza un secondo loop.

2.  **Fast Math Hardware**: Per alleggerire ulteriormente il singolo loop, usiamo approssimazioni in hardware nativo:
    *   Invece di `1.0f / sqrtf(dist_sq)` per calcolare $\eta$, usiamo `rsqrtf(dist_sq + EPS)`, che esegue l'operazione in un singolo ciclo di clock.

3.  **Warp-Level Merge (Riduzione Stocastica)**: Alla fine del loop locale, i 32 thread del warp uniscono stocasticamente i loro risultati tramite 5 passaggi di `__shfl_down_sync`. In soli 5 cicli di clock, il warp elegge il nodo globale basandosi sui pesi aggregati e ricava la `total_fallback_sum`.

4.  **Fix dello Stallo al Deposito**: Passando questa `total_fallback_sum` fuori dalla funzione, il kernel principale può ora valutare correttamente la probabilità di ritornare al deposito *contro* la probabilità di andare nel nodo di fallback, risolvendo il fastidioso "stallo" matematico che faceva tornare prematuramente le formiche al deposito.

Questa riprogettazione ha abbattuto i tempi computazionali del fallback e ripristinato l'esplorazione logica corretta del solver originale.
