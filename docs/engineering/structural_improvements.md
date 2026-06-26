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

## 2. Configurazione Runtime Unica

Timeout, stagnation, soglia di miglioramento, seed, numero formiche e candidate
list sono ancora letti in piu' punti.

Struttura consigliata:

```text
include/aco_config.h
src/common/aco_config.c
```

Esempio:

```c
typedef struct {
  double timeout_seconds;
  int stagnation_epochs;
  double min_rel_improvement;
  int fixed_epochs;
  int candidate_k;
  int ants;
  unsigned int seed;
} AcoRuntimeConfig;
```

## 3. API Uniforme Per I Backend

Stato attuale:

- backend CPU: funzioni `void`;
- backend CUDA: funzione con ritorno `int`;
- gestione errori non uniforme.

Struttura consigliata:

```c
typedef enum {
  ACO_OK = 0,
  ACO_ERR_INVALID_INPUT = 1,
  ACO_ERR_ALLOCATION = 2,
  ACO_ERR_NO_SOLUTION = 3,
  ACO_ERR_BACKEND = 4
} AcoStatus;
```

## 4. Layer Comune Per CLI, Output E Validazione

`src/main.c` e `src/cuda/main_vrp.cu` duplicano parsing argomenti, stampa route
e parte della gestione errori.

Struttura consigliata:

```text
include/cli_common.h
src/common/cli_common.c
```

## 5. Smoke Test Ufficiali

Esiste uno script smoke in `tools/bash/smoke_test.sh`, ma manca ancora una
integrazione Make completa.

Da implementare:

```bash
make smoke
make smoke_seq
make smoke_mpi
make smoke_cuda
```

I test dovrebbero verificare:

- generazione mini istanza;
- esecuzione solver;
- terminazione;
- soluzione valida;
- costo numerico presente;
- exit code corretto.

## Ordine Di Implementazione Consigliato

1. Introdurre `AcoRuntimeConfig`.
2. Uniformare API backend con `AcoStatus`.
3. Spostare parsing CLI/output in `cli_common`.
4. Aggiungere validazione CVRP comune.
5. Aggiungere target `make smoke`.
6. Ridurre o archiviare meglio copie complete dei solver in `experiments/`.
