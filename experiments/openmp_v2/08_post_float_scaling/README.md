# Esperimento 08: Strong Scaling Post-Float (N=16000)

Valutazione dell'efficienza dello Strong Scaling della V2 dopo l'ottimizzazione `float` su istanze di grandi dimensioni.

## Obiettivo
Verificare se il dimezzamento del traffico bus (tramite `float`) ha migliorato lo scaling a 24 core su problemi grandi, identificando se e dove il sistema torna a mostrare segni di saturazione.

## Parametri del test
- **Istanza:** `n16000_k16_s19005.vrp`
- **Formiche (M):** 1536 (64 per core)
- **Threads (T):** 1, 4, 12, 24
- **Epoche:** 20 (FIXED)
- **Strategy:** `guided,4`

## Esecuzione
`./experiments/openmp_v2/08_post_float_scaling/run.sh`
