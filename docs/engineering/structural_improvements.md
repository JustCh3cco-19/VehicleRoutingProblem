# Structural Improvements Roadmap

Questa roadmap contiene solo gli interventi strutturali ancora aperti. Gli step
gia' completati sono stati rimossi da questo documento e descritti nello stato
corrente dell'architettura:

```text
docs/engineering/architecture.md
```

## Completati

- Script Python consolidati in `tools/python/`.
- Script shell di smoke spostato in `tools/bash/`.
- Documentazione riorganizzata in `docs/usage`, `docs/engineering`,
  `docs/reports`.
- `instances/` documentata con `instances/README.md`.
- Dataset principale rinominato in `instances/generated_benchmark`.
- `results/` separata dagli artefatti versionati e documentata con
  `results/README.md`.
- Documento di architettura corrente creato.
- Layer comune per parsing CLI, output soluzione e validazione post-solver
  aggiunto in `include/cli_common.h` e `src/common/cli_common.c`.
- Smoke test ufficiali integrati nei target `make smoke`, `make smoke_seq`,
  `make smoke_mpi` e `make smoke_cuda`.
- Configurazione runtime consolidata in `include/aco_config.h` e
  `src/common/aco_config.c`.
- API backend uniformata con `AcoStatus` per CPU, MPI/OpenMP e CUDA.

## 1. Separare Meglio Core E Codice Sperimentale

Stato attuale:

```text
src/
experiments/openmp_v2/*/aco_v2.c
experiments/openmp_v2/*/aco_v2.h
```

Il codice in `src/` e' il backend stabile, mentre `experiments/openmp_v2/`
contiene molte copie complete del solver.

Da implementare:

- mantenere in `src/` solo le implementazioni ufficiali;
- usare `experiments/` per script, risultati, note e patch;
- evitare nuove copie complete dei solver;
- marcare chiaramente eventuali snapshot storici come legacy.

## Ordine Di Implementazione Consigliato

1. Ridurre o archiviare meglio copie complete dei solver in `experiments/`.
