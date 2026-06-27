# Benchmark

Questo documento definisce il benchmark per il progetto Vehicle Routing Problem
con solver ACO sequenziale, MPI/OpenMP e CUDA.

## Obiettivi

Le implementazioni parallele devono scalare. La valutazione deve misurare tempi,
speedup ed efficienza al crescere delle risorse e della dimensione del problema.
Se una versione parallela non scala, il report deve motivare il motivo tecnico:
overhead di sincronizzazione, comunicazione MPI, saturazione memoria, granularita'
troppo piccola, trasferimenti CPU/GPU, sbilanciamento tra formiche o dimensione
del problema insufficiente.

La correttezza delle versioni parallele deve essere verificata confrontando i
risultati con quelli ottenuti dalla versione sequenziale sulla stessa istanza,
con lo stesso seed e con gli stessi criteri di arresto. Differenze numeriche
entro `1e-5` sono accettabili per calcoli in `float`; differenze superiori o
soluzioni invalide devono essere analizzate nel report.

Le definizioni di scaling seguono HPC Wiki:

- speedup: `S(N) = T(1) / T(N)`;
- strong scaling: il numero di processori/thread/rank aumenta mentre la
  dimensione del problema resta costante;
- weak scaling: risorse e dimensione del problema aumentano insieme, mantenendo
  circa costante il lavoro per risorsa;
- weak scaling efficiency: `E(N) = T(1) / T(N)` quando ogni risorsa mantiene lo
  stesso carico di lavoro.

La baseline deve essere la migliore versione sequenziale o il run a una singola
risorsa comparabile. I risultati devono usare wall-clock time, non stime o
estrapolazioni.

## Istanze Generabili

Le istanze sono generate in formato TSPLIB/CVRP da
`tools/python/generate_vrp_problem.py` e registrate nei manifest in
`instances/generated_benchmark/`.

Parametri di generazione correnti:

| Parametro | Valore |
| --- | ---: |
| Layout coordinate | griglia `100 x 100` |
| Domanda clienti | 1 per cliente |
| Seed base istanze | 19000 |
| Seed solver | 1234 |
| Target clienti per veicolo | 1024 |
| Veicoli minimi | 8 |
| Veicoli massimi | 512 |
| Slack capacita' | 20% |
| Formula capacita' | `ceil((1 + 20/100) * n / K)` |

Manifest attuale:

| Istanza | Clienti `n` | Veicoli `K` | Seed istanza |
| --- | ---: | ---: | ---: |
| `n500_k8_s19000` | 500 | 8 | 19000 |
| `n1000_k8_s19001` | 1000 | 8 | 19001 |
| `n2000_k8_s19002` | 2000 | 8 | 19002 |
| `n4000_k8_s19003` | 4000 | 8 | 19003 |
| `n8000_k8_s19004` | 8000 | 8 | 19004 |
| `n16000_k16_s19005` | 16000 | 16 | 19005 |
| `n32000_k32_s19006` | 32000 | 32 | 19006 |
| `n64000_k63_s19007` | 64000 | 63 | 19007 |
| `n100000_k98_s19008` | 100000 | 98 | 19008 |

Il generatore Make prevede anche una campagna piu' grande configurabile con:

```bash
make generate_problems GEN_CLIENTS=4000,8000,16000,32000,64000,128000,200000
```

Per il report conviene usare la scala gia' presente fino a `100000` clienti se
la macchina non ha memoria sufficiente per `128000` o `200000`.

## Backend Confrontati

| Backend | Target Make | Binario | Note |
| --- | --- | --- | --- |
| Sequenziale | `make seq` | `seq.out` | Baseline di correttezza e speedup. |
| MPI/OpenMP | `make openmp_mpi` | `openmp_mpi.out` | Backend ibrido CPU distribuito/condiviso. |
| CUDA | `make cuda` | `cuda.out` | Backend GPU; richiede GPU CUDA visibile. |
| PyVRP | `make solve_pyvrp` | runner Python | Riferimento esterno opzionale per qualita' soluzione. |

## Parametri Solver

I manifest usano valori diversi di `m` per backend e dimensione:

| Clienti `n` | `m` sequenziale | `m` MPI/OpenMP | `m` CUDA |
| ---: | ---: | ---: | ---: |
| 500 | 32 | 64 | 256 |
| 1000 | 32 | 64 | 256 |
| 2000 | 32 | 64 | 256 |
| 4000 | 16 | 32 | 256 |
| 8000 | 16 | 32 | 256 |
| 16000 | 8 | 16 | 256 |
| 32000 | 4 | 8 | 256 |
| 64000 | 3 | 4 | 256 |
| 100000 | 3 | 4 | 256 |

Per confronti di speedup stretti, usare gli stessi criteri di arresto per tutti
i backend:

```bash
ACO_SOLVER_TIMEOUT_SECONDS=<seconds>
ACO_SOLVER_STAGNATION_EPOCHS=<epochs>
ACO_SOLVER_MIN_REL_IMPROVEMENT=<percent>
ACO_SOLVER_PROGRESS_INTERVAL_SECONDS=10
```

Per misure di qualita' soluzione, riportare anche `best cost` finale e validita'
della soluzione.

## Campagne Consigliate

### 1. Correttezza Funzionale

Eseguire su istanze piccole e medie:

| Clienti | Scopo |
| ---: | --- |
| 500 | smoke e confronto rapido backend. |
| 1000 | verifica stabilita' su dimensione piccola. |
| 2000 | verifica correttezza con piu' lavoro. |
| 4000 | primo caso significativo per parallelismo. |

Per ogni backend:

- exit code deve essere 0;
- output deve contenere costo numerico;
- ogni cliente deve essere visitato una sola volta;
- ogni route deve iniziare e terminare al deposito;
- capacita' veicolo deve essere rispettata;
- costo ricalcolato deve coincidere con quello stampato entro tolleranza.

La validazione interna del binario copre la struttura della soluzione. Per una
validazione indipendente si puo' usare `tools/python/validate_pyvrp.py` quando
PyVRP e' disponibile.

### 2. Strong Scaling CPU

Fissare una dimensione e aumentare le risorse CPU. Questo misura quanto si
riduce il tempo per risolvere lo stesso problema quando aumentano thread/rank.
Secondo la definizione HPC, il workload per risorsa diminuisce; quindi e'
normale che oltre un certo punto lo speedup si appiattisca per overhead seriale,
sincronizzazione o comunicazione.

Dimensioni consigliate:

| Scenario | Clienti |
| --- | ---: |
| Piccolo | 4000 |
| Medio | 8000 |
| Grande | 16000 o 32000 |

Configurazioni:

| Backend | Risorse |
| --- | --- |
| OpenMP/MPI single rank | `SMOKE_MPI_RANKS=1`, variare `OMP_NUM_THREADS` o `SOLVE_MPI_OMP_THREADS`. |
| MPI puro/ibrido | variare `SOLVE_MPI_RANKS` e `SOLVE_MPI_OMP_THREADS`. |

Esempi di risorse: 1, 2, 4, 8, 16, 32, 64 core/thread, nei limiti della
macchina disponibile.

Metriche:

- tempo medio su almeno 3 run;
- speedup `T1 / Tp`;
- efficienza `speedup / p`;
- costo finale medio/minimo;
- motivazione se speedup si appiattisce o peggiora.

### 3. Weak Scaling CPU

Aumentare il problema insieme alle risorse. Il Makefile usa come default
`EXP_WEAK_BASE_N_PER_WORKER=2000`, quindi una campagna coerente e':

| Worker/core equivalenti | Clienti indicativi |
| ---: | ---: |
| 1 | 2000 |
| 2 | 4000 |
| 4 | 8000 |
| 8 | 16000 |
| 16 | 32000 |
| 32 | 64000 |

Se il tempo resta circa costante, la weak scaling e' buona. Se cresce molto,
discutere se il collo di bottiglia e' comunicazione, memoria o sincronizzazione.
La metrica principale e' `E(N) = T(1) / T(N)`: idealmente resta vicina a 1.

### 4. CUDA Scaling

CUDA va valutato sulle dimensioni supportate dal manifest CUDA:

| Clienti | `m` |
| ---: | ---: |
| 500 | 256 |
| 1000 | 256 |
| 2000 | 256 |
| 4000 | 256 |
| 8000 | 256 |
| 16000 | 256 |
| 32000 | 256 |
| 64000 | 256 |
| 100000 | 256 |

Campagne consigliate:

- scaling per dimensione con `m=256`;
- scaling per numero formiche su `500,1000,2000,4000,8000` con `m=64,128,256`;
- confronto contro sequenziale e MPI/OpenMP per le istanze che tutti i backend
  riescono a completare.

Per CUDA motivare eventuali non-scaling con trasferimenti host/device,
saturazione memoria GPU, occupancy insufficiente o overhead kernel dominante su
istanze piccole.

## Metriche Del Report

Ogni tabella benchmark dovrebbe contenere:

| Campo | Descrizione |
| --- | --- |
| backend | `seq`, `mpi`, `cuda`, `pyvrp` se usato. |
| instance | nome istanza dal manifest. |
| `n`, `K`, `m` | dimensione, veicoli, formiche. |
| seed | seed solver. |
| risorse | rank MPI, thread OpenMP, GPU, CUDA arch se rilevante. |
| runtime medio | media su run ripetuti. |
| runtime min/max/std | dispersione delle misure. |
| best cost | costo finale riportato. |
| valid | esito validazione soluzione. |
| speedup | rispetto al sequenziale comparabile. |
| efficiency | `speedup / risorse`. |
| weak_efficiency | `T(1) / T(N)` per campagne weak scaling. |

## Regole Di Correttezza

Il confronto con il sequenziale deve distinguere due aspetti:

1. Validita' CVRP: ogni cliente visitato una volta, route valide, capacita'
   rispettata, costo numerico finito.
2. Coerenza numerica: costo ricalcolato uguale al costo stampato entro `1e-5`
   quando si confrontano valori `float`.

Il costo finale tra backend ACO puo' differire anche con stesso seed se
l'ordine delle riduzioni o la politica di esplorazione cambia. In quel caso la
soluzione parallela puo' essere corretta anche con costo diverso, ma il report
deve dirlo esplicitamente e confrontare almeno:

- validita' della soluzione;
- costo finale;
- gap percentuale rispetto al sequenziale;
- gap percentuale rispetto a PyVRP, se disponibile.

## Quando Una Versione Non Scala

Il report deve motivare ogni caso in cui speedup o efficienza peggiorano.
Motivazioni tipiche per questo progetto:

- istanze piccole: lavoro per thread/rank insufficiente;
- MPI: comunicazione e sincronizzazione piu' costose del lavoro utile;
- OpenMP: contesa su aggiornamenti condivisi o memoria bandwidth-bound;
- CUDA: overhead kernel/trasferimenti dominante per `n` piccoli;
- memoria: matrice distanze o strutture pheromone troppo grandi;
- ACO: poche formiche su problemi grandi riducono parallelismo utile;
- early stopping: timeout/stagnation diversi rendono i tempi non comparabili.

Nel report non bisogna presentare speedup ottenuti con baseline non comparabili:
per strong scaling la baseline deve avere lo stesso problema; per weak scaling
la baseline deve avere lo stesso carico per risorsa. Se una dimensione non entra
in memoria su una sola risorsa, il report deve dichiararlo e usare la dimensione
piu' piccola confrontabile come riferimento.

## Comandi Utili

Generazione istanze:

```bash
make generate_problems
```

Smoke test:

```bash
make smoke
```

Solve per manifest:

```bash
make solve_seq SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300
make solve_mpi SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=8
make solve_cuda SOLVE_CUDA_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300
```

Campagne scaling gia' previste:

```bash
make exp_strong_openmp
make exp_strong_mpi
make exp_weak_mpi
make exp_cuda_all
```

## Note Su `zanoni_martinelli.pdf`

Il PDF analizzato usa K-means, non VRP. Da quel documento sono stati mantenuti
solo i principi metodologici applicabili anche qui:

- eseguire piu' run per configurazione;
- riportare media, min, max e deviazione standard;
- misurare execution time, speedup, efficiency e scaling quality;
- confrontare sempre con una baseline sequenziale;
- motivare i casi in cui la parallelizzazione non scala.

## Fonte Scaling

Le definizioni operative di strong scaling, weak scaling, speedup, efficienza e
linee guida di presentazione derivano da HPC Wiki, "Scaling":
https://hpc-wiki.info/hpc/Scaling
