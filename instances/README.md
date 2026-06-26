# Instances

Questa directory contiene dataset CVRP in formato `.vrp` usati dai solver e dai
benchmark.

## Dataset Principale

```text
generated_benchmark/
```

Contiene il benchmark sintetico generato dal target:

```bash
make generate_problems
```

File principali:

```text
generated_benchmark/*.vrp
generated_benchmark/manifest.csv
generated_benchmark/manifest_openmp_mpi.csv
generated_benchmark/manifest_cuda.csv
```

I manifest sono allineati per istanza ma contengono parametri `m` specifici per
backend.

## Dataset Temporanei

Per debug o prove locali usa una directory separata:

```bash
make generate_problems GEN_INST_DIR=instances/debug GEN_CLIENTS=50,100
```

I dataset temporanei non dovrebbero sostituire `generated_benchmark`.

## Documentazione

La procedura completa di generazione e' descritta in:

```text
docs/usage/instance_generation.md
```
