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

*Tutti gli interventi elencati sono stati implementati:*
- **Punto 1 (Registrazione Metadata)**: Automatizzato nei CSV dei risultati di seq, openmp_mpi e cuda.
- **Punto 2 (Test di Regressione)**: Aggiunta suite di test in `tools/python/regression_tests.py` ed esposta via `make regression`.
- **Punto 3 (Profilazione Guidata)**: Automatizzato lo script in `tools/bash/profile_guided.sh` ed esposto via `make profile`.

## Roadmap Suggerita

Nessun intervento aperto.


