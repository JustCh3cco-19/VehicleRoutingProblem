# Esperimento 05: Granularità e Numero di Formiche (Ants per Core)

Questo esperimento analizza come il numero totale di formiche ($M$) influenzi l'efficienza dello Strong Scaling. L'obiettivo è trovare il rapporto minimo formiche/core necessario per giustificare l'uso di un nodo intero (24 core).

## Obiettivo
Verificare se con poche formiche per core l'overhead di OpenMP e lo sbilanciamento del carico distruggano l'efficienza, e identificare la soglia di saturazione.

## Parametri del test
- **Istanza:** `n2000_k8_s19002.vrp`
- **Formiche (M):** 24, 96, 384, 1536 (da 1 a 64 formiche per core)
- **Threads (T):** 1, 4, 12, 24
- **Epoche:** 20 (FIXED)
- **Strategy:** `guided,1`

## Esecuzione
`./experimental/experiments/openmp_v2/05_ants_granularity/run.sh`
