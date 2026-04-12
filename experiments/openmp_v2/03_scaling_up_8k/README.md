# Esperimento 03: Scaling Up a 8k Clienti

Valutazione dell'architettura V2 su un'istanza di dimensioni maggiori (N=8000) per testare l'impatto della granularità dello scheduling `guided`.

## Obiettivo
Determinare se, all'aumentare della taglia del problema, un `chunk` minimo superiore a 1 nello scheduling `guided` porti benefici riducendo ulteriormente l'overhead di OpenMP.

## Parametri del test
- **Istanza:** `n8000_k8_s19004.vrp` (~1GB RAM)
- **Formiche (M):** 768
- **Epoche:** 20 (FIXED)
- **Threads:** 24
- **Strategie:** `guided,1`, `guided,4`, `guided,8`, `guided,16`

## Esecuzione
`./experiments/openmp_v2/03_scaling_up_8k/run.sh`
