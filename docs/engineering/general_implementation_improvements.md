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

### 1. Ottimizzazione del Loop di Selezione in `aco_vrp_run_with_config` (CPU)

La profilazione tramite `gprof` evidenzia che oltre il **93% del tempo di calcolo CPU** è speso all'interno della funzione `aco_vrp_run_with_config`.

Azioni richieste:
- Profilare con micro-metriche (cache-misses, branch mispredictions) il ciclo interno in cui le formiche calcolano le probabilità di transizione per scegliere il prossimo nodo.
- Ottimizzare la pre-computazione o l'accesso a `eta^beta` e valutare la riduzione delle chiamate al generatore pseudo-casuale `aco_rand01_state`.

## Roadmap Suggerita

1. Ottimizzare il collo di bottiglia principale CPU in `aco_vrp_run_with_config`.