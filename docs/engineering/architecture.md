# Current Architecture

Questo documento descrive lo stato corrente dell'architettura del progetto.

## Overview

Il progetto implementa solver ACO per CVRP con tre backend:

- `seq`: baseline CPU sequenziale;
- `openmp-mpi`: backend parallelo ibrido condiviso/distribuito;
- `cuda`: backend GPU single-device.

Il flusso operativo e':

```text
generate instances -> write manifests -> run solvers -> collect CSV/routes -> plot/report
```

## Repository Layout

```text
include/             Header pubblici C/CUDA.
src/                 Implementazioni ufficiali dei solver.
src/common/          Parser, matrici, soluzioni e helper condivisi.
src/seq/             Backend sequenziale.
src/openmp-mpi/      Backend OpenMP/MPI.
src/cuda/            Backend CUDA.
tools/makefile/      Moduli Makefile.
tools/bash/          Launcher shell e utility operative.
tools/python/        Generatori, benchmark, plotting e report Python.
tools/batch/         Script Slurm.
instances/           Dataset CVRP e manifest.
docs/usage/          Guide operative.
docs/engineering/    Note tecniche e roadmap ingegneristiche.
docs/reports/        Report e documenti finali.
experiments/         Varianti sperimentali e studi storici.
results/             Artefatti generati, ignorati da git salvo README.
```

## Instance Generation

Il dataset principale e':

```text
instances/generated_benchmark/
```

Si rigenera con:

```bash
make generate_problems
```

Il generatore produce:

- file `.vrp`;
- `manifest.csv` per PyVRP/sequenziale;
- `manifest_openmp_mpi.csv` per OpenMP/MPI;
- `manifest_cuda.csv` per CUDA.

La guida completa e':

```text
docs/usage/instance_generation.md
```

## Solver Execution

I target principali sono:

```bash
make solve_seq
make solve_mpi
make solve_cuda
make solve_pyvrp
```

I launcher shell vivono in:

```text
tools/bash/
```

I target Make passano manifest, budget runtime, stagnation, seed e soglie ai
solver tramite variabili d'ambiente.

## Result Pipeline

Gli output generati finiscono sotto:

```text
results/
```

Pattern principali:

```text
results/solve_manifest/csv/
results/solve_manifest/solutions/
results/practical_campaign/
results/analysis/
results/slurm/
```

`results/` e' ignorata da git, tranne `results/README.md`.

## Python Tooling

Tutti gli script Python mantenuti sono consolidati in:

```text
tools/python/
```

Contenuti tipici:

- generazione istanze;
- runner PyVRP;
- validazione;
- benchmark;
- plotting;
- summarizzazione campagne.

Gli script shell operativi restano in:

```text
tools/bash/
```

Esempi:

- `solve_seq.sh`, `solve_mpi.sh`, `solve_cuda.sh`: launcher usati dai target
  `make solve_*`;
- `smoke_test.sh`: controllo rapido locale;
- `run_scaling_tests.sh`: scaling/profiling CUDA manuale, eseguibile tramite
  `make run_scaling_tests`, con output in `results/scaling_tests/`.

## Documentation

La documentazione e' divisa per scopo:

```text
docs/usage/        Guide d'uso.
docs/engineering/  Architettura, analisi tecniche, roadmap.
docs/reports/      Report finali.
```

## Stable And Experimental Areas

Codice stabile:

```text
src/
include/
tools/
```

Codice e studi sperimentali:

```text
experiments/
```

Le varianti in `experiments/openmp_v2/` sono utili per ricostruire decisioni
storiche, ma non dovrebbero essere trattate come backend ufficiali.

## Extension Points

Per aggiungere un nuovo backend:

1. creare una directory sotto `src/<backend>/`;
2. esporre una funzione compatibile con `aco_vrp_with_capacity` o wrapper
   equivalente;
3. aggiungere target build in `tools/makefile/build.mk`;
4. aggiungere launcher in `tools/bash/` se serve;
5. aggiungere manifest o parametri backend-specifici se il backend richiede `m`
   diverso;
6. aggiungere smoke test e documentazione d'uso.

Per aggiungere una nuova campagna:

1. definire output sotto `results/<campaign>/`;
2. orchestrare da Makefile o `tools/python/`;
3. salvare configurazione effettiva;
4. produrre CSV e report riproducibili.
