# Esperimento 02: Scientific Strong Scaling

Misura lo scaling puro dell'hardware eliminando la varianza algoritmica tramite epoche fisse.

## Esecuzione
Lancia `./run.sh` dalla root del progetto:
`./experimental/experiments/openmp_v2/02_scientific_strong_scaling/run.sh`

## Parametri del test
- **Epoche:** 100 (FIXED)
- **Threads:** 1, 2, 4, 8, 12, 16, 20, 24
- **Strategy:** `guided,1`

## Interpretazione dei risultati
Il valore di "Efficiency" indica quanto il sistema si avvicina allo scaling lineare (1.0 = 100%).
Lo speedup ideale a 24 thread è 24x. Nei nostri test abbiamo raggiunto circa 14.6x (60% efficienza).
