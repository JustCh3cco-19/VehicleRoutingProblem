# Instance Generation Guide

Questa guida spiega come vengono generate le istanze CVRP usate dal progetto e
come crearne di nuove in modo riproducibile.

## Quick Start

Per rigenerare il set standard di istanze e manifest:

```bash
make generate_problems
```

Output principali:

```text
instances/generated_benchmark/*.vrp
instances/generated_benchmark/manifest.csv
instances/generated_benchmark/manifest_openmp_mpi.csv
instances/generated_benchmark/manifest_cuda.csv
```

Per generare solo alcune taglie:

```bash
make generate_problems GEN_CLIENTS=500,1000,2000
```

Per generare istanze in una directory separata:

```bash
make generate_problems GEN_INST_DIR=instances/my_test GEN_CLIENTS=100,500,1000
```

Per pulire le istanze generate nella directory configurata:

```bash
make generate_clean
```

## Dipendenze

Il generatore usa Python e le librerie dichiarate in `requirements.txt`.

Installazione tipica:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Il Makefile usa automaticamente `.venv/bin/python` se esiste, altrimenti
`python3`.

## Generazione Via Makefile

Il target principale e':

```bash
make generate_problems
```

La logica e' definita in `tools/makefile/generate.mk`.

Per ogni valore `n` in `GEN_CLIENTS`, il Makefile:

1. calcola il numero di veicoli `K`;
2. calcola il seed dell'istanza;
3. chiama `tools/python/generate_vrp_problem.py`;
4. scrive una riga nei tre manifest: sequenziale/PyVRP, OpenMP-MPI, CUDA;
5. ordina i manifest per numero clienti e seed.

## Parametri Principali

I parametri si passano come variabili Make.

```bash
make generate_problems \
  GEN_CLIENTS=500,1000,2000 \
  GEN_GRID=100 \
  GEN_SEED_BASE=19000 \
  GEN_TARGET_CUSTOMERS_PER_VEHICLE=1024 \
  GEN_MIN_VEHICLES=8 \
  GEN_MAX_VEHICLES=512 \
  GEN_CAPACITY_SLACK_PERCENT=20
```

| Variabile | Default | Significato |
| --- | ---: | --- |
| `GEN_INST_DIR` | `instances/generated_benchmark` | Directory di output per `.vrp` e manifest. |
| `GEN_CLIENTS` | `4000,8000,16000,32000,64000,128000,200000` | Lista CSV delle taglie `n` da generare. |
| `GEN_SEED_BASE` | `19000` | Seed della prima istanza; incrementa di 1 per ogni taglia. |
| `GEN_GRID` | `100` | Coordinate intere generate nel quadrato `[0, GEN_GRID)`. |
| `GEN_SOLVER_SEED` | `1234` | Seed scritto nei manifest per i solver. Non cambia la geometria dell'istanza. |
| `GEN_TARGET_CUSTOMERS_PER_VEHICLE` | `1024` | Obiettivo medio clienti/veicolo per calcolare `K`. |
| `GEN_MIN_VEHICLES` | `8` | Numero minimo di veicoli. |
| `GEN_MAX_VEHICLES` | `512` | Numero massimo di veicoli. |
| `GEN_CAPACITY_SLACK_PERCENT` | `20` | Slack percentuale sulla capacita' veicolo. |
| `GEN_CUDA_M` | `256` | Numero formiche scritto nel manifest CUDA. |

## Come Vengono Calcolati K E Capacita'

Il numero di veicoli e' calcolato cosi':

```text
K = ceil(n / GEN_TARGET_CUSTOMERS_PER_VEHICLE)
K = clamp(K, GEN_MIN_VEHICLES, GEN_MAX_VEHICLES)
```

La capacita' dei veicoli e':

```text
CAPACITY = ceil((1 + GEN_CAPACITY_SLACK_PERCENT / 100) * n / K)
```

Esempio con `n=16000`, `GEN_TARGET_CUSTOMERS_PER_VEHICLE=1024`,
`GEN_MIN_VEHICLES=8`, `GEN_MAX_VEHICLES=512`, `GEN_CAPACITY_SLACK_PERCENT=20`:

```text
K = ceil(16000 / 1024) = 16
CAPACITY = ceil(1.20 * 16000 / 16) = 1200
```

Tutti i clienti hanno domanda unitaria:

```text
depot demand = 0
customer demand = 1
```

Quindi la capacita' indica direttamente il massimo numero di clienti servibili da
un singolo veicolo.

## Layout Delle Istanze

Il file Python genera istanze TSPLIB-like con:

```text
TYPE: CVRP
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
DEMAND_SECTION
DEPOT_SECTION
```

La geometria e' sintetica:

- il deposito e' un punto casuale sulla griglia;
- ogni cliente e' un punto casuale sulla stessa griglia;
- le distanze sono euclidee 2D;
- il seed rende la generazione riproducibile.

Il nome standard e':

```text
n<N>_k<K>_s<SEED>.vrp
```

Esempio:

```text
n16000_k16_s19005.vrp
```

## Manifest Generati

Il target `generate_problems` produce tre manifest:

```text
manifest.csv
manifest_openmp_mpi.csv
manifest_cuda.csv
```

Le colonne sono:

```text
profile,name,instance_path,n,K,m,solver_seed,instance_seed,layout_id,capacity_formula
```

Significato:

| Colonna | Significato |
| --- | --- |
| `profile` | Profilo del backend: `generated`, `generated_mpi`, `generated_cuda`. |
| `name` | Nome logico dell'istanza. |
| `instance_path` | Percorso del file `.vrp`. |
| `n` | Numero clienti, escluso il deposito. |
| `K` | Numero veicoli. |
| `m` | Numero formiche consigliato per quel backend. |
| `solver_seed` | Seed usato dai solver durante i run. |
| `instance_seed` | Seed usato per generare coordinate e domanda. |
| `layout_id` | Identificatore del layout, ad esempio `grid100`. |
| `capacity_formula` | Formula usata per calcolare la capacita'. |

Il valore `m` cambia per backend:

- `manifest.csv`: usato da PyVRP e solver sequenziale;
- `manifest_openmp_mpi.csv`: usato dal solver OpenMP/MPI;
- `manifest_cuda.csv`: usa `GEN_CUDA_M`, default `256`.

## Generare Una Singola Istanza

Si puo' chiamare direttamente lo script Python:

```bash
python3 tools/python/generate_vrp_problem.py \
  --name n500_k8_s19000 \
  --clients 500 \
  --vehicles 8 \
  --grid 100 \
  --seed 19000 \
  --capacity-slack-percent 20 \
  --output instances/generated_benchmark/n500_k8_s19000.vrp
```

Questo genera solo il file `.vrp`, senza aggiornare i manifest.

Per generare anche una soluzione di riferimento con PyVRP:

```bash
python3 tools/python/generate_vrp_problem.py \
  --name small_test \
  --clients 100 \
  --vehicles 8 \
  --grid 100 \
  --seed 123 \
  --output instances/small_test.vrp \
  --solve \
  --runtime 10
```

Output aggiuntivo:

```text
instances/small_test_solution.txt
```

## Esempi Pratici

### Istanze piccole per debug

```bash
make generate_problems \
  GEN_INST_DIR=instances/debug \
  GEN_CLIENTS=20,50,100 \
  GEN_TARGET_CUSTOMERS_PER_VEHICLE=25 \
  GEN_MIN_VEHICLES=2 \
  GEN_MAX_VEHICLES=8 \
  GEN_SEED_BASE=1000
```

### Istanze medie per test locali

```bash
make generate_problems \
  GEN_CLIENTS=500,1000,2000,4000 \
  GEN_TARGET_CUSTOMERS_PER_VEHICLE=512 \
  GEN_MIN_VEHICLES=4 \
  GEN_MAX_VEHICLES=32
```

### Istanze grandi per esperimenti HPC

```bash
make generate_problems \
  GEN_CLIENTS=8000,16000,32000,64000,100000 \
  GEN_TARGET_CUSTOMERS_PER_VEHICLE=1024 \
  GEN_MIN_VEHICLES=8 \
  GEN_MAX_VEHICLES=512
```

### Profilo grande gia' pronto

```bash
make generate_problems_big
```

Questo usa:

```text
GEN_CLIENTS=4000,8000,12000,16000,20000,24000,28000,32000
```

## Usare Le Nuove Istanze Nei Solver

Se hai generato in `instances/my_test`:

```bash
make solve_seq SOLVE_MANIFEST=instances/my_test/manifest.csv
```

```bash
make solve_mpi SOLVE_MANIFEST_MPI=instances/my_test/manifest_openmp_mpi.csv
```

```bash
make solve_cuda SOLVE_MANIFEST_CUDA=instances/my_test/manifest_cuda.csv
```

Puoi filtrare per numero di clienti:

```bash
make solve_seq \
  SOLVE_MANIFEST=instances/my_test/manifest.csv \
  SOLVE_CLIENTS=500,1000
```

## Note Di Riproducibilita'

- A parita' di `GEN_CLIENTS`, `GEN_SEED_BASE`, `GEN_GRID` e parametri veicolo,
  gli stessi file `.vrp` vengono rigenerati.
- `generate_problems` rimuove prima i vecchi file `n*_k*_s*.vrp` nella directory
  target, poi rigenera istanze e manifest.
- Se vuoi conservare una campagna precedente, usa un nuovo `GEN_INST_DIR`.
- Il seed dell'istanza (`instance_seed`) controlla coordinate e layout.
- Il seed del solver (`solver_seed`) controlla i run euristici e viene solo scritto
  nei manifest.
