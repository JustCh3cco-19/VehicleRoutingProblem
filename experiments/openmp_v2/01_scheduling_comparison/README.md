# Esperimento 01: Confronto Scheduling OpenMP

Misura l'impatto della granularità dello scheduling (dynamic vs guided) sulle performance della V2.

## Esecuzione
Lancia `./run.sh` dalla root del progetto:
`./experiments/openmp_v2/01_scheduling_comparison/run.sh`

## Strategie testate
- `dynamic,1`
- `dynamic,4`
- `guided,1`

## Risultati attesi
Lo scheduling `guided` dovrebbe risultare significativamente più veloce (~7x) su molti thread a causa della riduzione della contesa sui lock interni di OpenMP.
