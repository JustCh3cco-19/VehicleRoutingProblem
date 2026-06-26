# General Implementation Improvements

Questo documento riassume miglioramenti possibili nell'implementazione generale
del solver VRP, analizzando `src/` e `include/`. Le priorita' sono orientate a
correttezza, scalabilita', manutenibilita' e benchmark riproducibili.

Regola di manutenzione: quando un miglioramento viene implementato, rimuoverlo
da questo documento. Il file deve contenere solo interventi ancora aperti.

Regola di implementazione: prima di implementare qualsiasi modifica, leggere e
seguire `docs/engineering/c_style_guide.md`. Per codice CUDA o host/device CUDA
leggere e seguire anche `docs/engineering/cuda_style_guide.md`.

## Miglioramenti Specifici Per Benchmark

### 1. Implementare Incrementi dei Contatori `CudaIterStats` nei Kernel CUDA

Dalle analisi del profiler emerge che i contatori diagnostici di `CudaIterStats` (es. `customer_moves`, `fallback_calls`, `fallback_moves`) sono attualmente azzerati ad ogni esecuzione. La struttura e la memoria device/host sono allocate e passate correttamente, ma il kernel `kernel_construct_solutions` in `src/cuda/aco_cuda_kernels.cu` non implementa le scritture atomiche per aggiornarne i valori.

Azioni richieste:
- Aggiungere gli incrementi (es. tramite `atomicAdd` o riduzioni a livello di warp/block) per i vari contatori diagnostici all'interno di `kernel_construct_solutions`.
- Verificare che il riepilogo stampato a fine esecuzione riporti valori congruenti con le iterazioni eseguite.

### 2. Ottimizzazione del Loop di Selezione in `aco_vrp_run_with_config` (CPU)

La profilazione tramite `gprof` evidenzia che oltre il **93% del tempo di calcolo CPU** è speso all'interno della funzione `aco_vrp_run_with_config`.

Azioni richieste:
- Profilare con micro-metriche (cache-misses, branch mispredictions) il ciclo interno in cui le formiche calcolano le probabilità di transizione per scegliere il prossimo nodo.
- Ottimizzare la pre-computazione o l'accesso a `eta^beta` e valutare la riduzione delle chiamate al generatore pseudo-casuale `aco_rand01_state`.

## Roadmap Suggerita

1. Implementare i contatori diagnostici CUDA in `aco_cuda_kernels.cu`.
2. Ottimizzare il collo di bottiglia principale CPU in `aco_vrp_run_with_config`.