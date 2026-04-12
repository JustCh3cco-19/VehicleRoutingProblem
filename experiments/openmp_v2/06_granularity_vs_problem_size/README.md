# Esperimento 06: Granularità vs Dimensione del Problema

Analisi bi-dimensionale dell'efficienza dello Strong Scaling variando sia la taglia del problema ($N$) che la densità di formiche per core ($M/T$).

## Obiettivo
Verificare se la "Regola del 60" (60 formiche per core) sia universale o se, all'aumentare di $N$, bastino meno formiche per core per ottenere un'efficienza elevata.

## Parametri del test
- **Dimensioni (N):** 2000, 8000, 16000
- **Densità (M/T):** 4, 16, 64 (corrispondenti a M=96, 384, 1536 su 24 thread)
- **Threads:** 1 (baseline), 24
- **Epoche:** 10 (FIXED)
- **Strategy:** `guided,1`

## Esecuzione
`./experiments/openmp_v2/06_granularity_vs_problem_size/run.sh`
