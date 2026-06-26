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

### 1. Registrare Metadata Completi In Output CSV

Per confronti seri servono piu' informazioni nei risultati.

Campi consigliati:

- backend;
- commit hash;
- compiler e flags;
- host/GPU;
- `n`, `K`, capacity, `m`;
- candidate_k effettivo;
- seed;
- timeout/stagnation/min improvement;
- rank MPI, thread OpenMP;
- CUDA arch;
- runtime, best cost, validazione, exit status.

### 2. Aggiungere Test Di Regressione Su Correttezza

Gli smoke test sono utili, ma servono casi piu' mirati.

Test consigliati:

- istanza piccola con soluzione nota;
- capacita' impossibile;
- demand non unitaria quando supportata;
- route overflow;
- confronto costo stampato vs costo ricalcolato;
- backend che deve restituire `ACO_ERR_NO_SOLUTION`.

### 3. Profilazione Guidata

Prima di ottimizzare ancora, conviene produrre dati stabili.

Strumenti consigliati:

- `perf`/`gprof` per CPU;
- MPI profiling per tempo in comunicazione;
- Nsight Systems/Compute per CUDA;
- contatori interni gia' presenti in `CudaIterStats`;
- metriche cache per candidate list e fallback.

## Roadmap Suggerita

1. Rafforzare benchmark CSV e test di regressione.
