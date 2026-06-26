# Results

Questa directory e' riservata agli artefatti generati durante run, benchmark,
campagne batch e plotting.

Esempi di output:

```text
results/solve_manifest/
results/practical_campaign/
results/analysis/
results/slurm/
```

Regole:

- i contenuti generati sotto `results/` non sono versionati;
- questo README e' tracciato per documentare lo scopo della directory;
- report finali selezionati e stabili vanno spostati in `docs/reports/`;
- file temporanei o log locali non devono finire in `src/`, `docs/` o
  `instances/generated_benchmark`.
