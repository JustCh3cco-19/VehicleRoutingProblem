# Comandi SLURM per i backend VRP

I tre backend del progetto sono:

- `seq`: CPU sequenziale;
- `openmp-mpi`: backend ibrido MPI + OpenMP;
- `cuda`: backend GPU CUDA.

Eseguire tutti i comandi dalla root della repository sul cluster:

```bash
cd /percorso/sul/cluster/VehicleRoutingProblem
```

Sostituire il percorso con la directory nella quale la repository è stata
copiata o clonata sul cluster.

## Configurazione iniziale

Controllare partizioni e risorse disponibili:

```bash
sinfo -o "%P %G %c %m"
srun --mpi=list
```

I comandi usano esplicitamente `CUDA_ARCH=sm_75`. È solo un esempio e deve
essere sostituito quando la GPU ha una compute capability differente. Per
controllare il modello della GPU tramite Slurm:

```bash
srun --nodes=1 --ntasks=1 --gres=gpu:1 nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

I comandi non impongono una partizione chiamata `gpu`, perché non esiste su
tutti i cluster. Se `sinfo -o "%P %G %c %m"` mostra le GPU in una partizione
specifica, aggiungere `--partition NOME_REALE` a ogni comando CUDA.

I risultati usano percorsi fissi sotto `results/manual_campaign/` e non
richiedono variabili come `TAG`. Prima di iniziare una nuova campagna completa,
archiviare la directory precedente se non si vogliono mescolare vecchi e nuovi
risultati.

## Avvio separato dei tre backend

Le istanze disponibili fino a 64.000 clienti sono:

```text
500,1000,2000,4000,8000,16000,32000,64000
```

Ogni job richiede al massimo 30 minuti. Per lasciare margine alla compilazione,
all'inizializzazione e alla scrittura dei risultati, ogni singolo run del solver
è limitato a 300 secondi. Con tre ripetizioni il calcolo può durare al massimo
15 minuti; con cinque ripetizioni può durare al massimo 25 minuti. Il limite
disponibile è di 32 GB di RAM e 32 CPU per task.

I backend sequenziale e OpenMP+MPI conservano matrici dense con memoria
quadratica. A 64.000 clienti un singolo processo richiede oltre 60 GiB, e il
sequenziale necessita di ulteriore memoria per le tabelle ausiliarie. Quel caso
non può quindi essere eseguito con il limite di 32 GB. Le campagne CPU si
fermano a 32.000 clienti; CUDA usa una sola GPU e arriva a 64.000.

### Backend sequenziale

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=500 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n500/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n500/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=1000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n1000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n1000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=2000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n2000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n2000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=4000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n4000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n4000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n8000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n8000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n16000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n16000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_seq --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/seq/n32000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/seq/n32000/solutions"
```

### Backend OpenMP + MPI

Questa configurazione usa quattro rank MPI su quattro nodi, ciascuno con 32
thread OpenMP:

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=500 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n500 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n500/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n500/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=1000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n1000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n1000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n1000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=2000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n2000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n2000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n2000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=4000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n4000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n4000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n4000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n8000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n8000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n8000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n16000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n16000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n16000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=backend_mpi_n32000 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/mpi/n32000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/mpi/n32000/solutions"
```

Il valore di `--ntasks` deve essere almeno pari a `SOLVE_MPI_RANKS`, mentre
`--cpus` deve essere pari a `SOLVE_MPI_OMP_THREADS`.

Se il cluster non supporta PMIx tramite `srun`, sostituire
`SOLVE_MPI_LAUNCHER=srun` con `SOLVE_MPI_LAUNCHER=mpirun`.

### Backend CUDA

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=500 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n500/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n500/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=1000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n1000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n1000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=2000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n2000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n2000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=4000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n4000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n4000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n8000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n8000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n16000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n16000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n32000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n32000/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=64000 SOLVE_CUDA_REPEATS=3 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75 SOLVE_CSV_DIR=results/manual_campaign/backend_sizes/cuda/n64000/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/backend_sizes/cuda/n64000/solutions"
```

La GPU e le altre risorse CUDA rimangono fisse per tutte le istanze: questo è
un benchmark per dimensione del problema, non un esperimento di scaling CUDA.

## Campagne di scaling CPU configurate manualmente

I comandi seguenti usano esclusivamente il target base `solve_mpi`. Ogni
configurazione viene costruita passando esplicitamente le variabili Make, senza
richiamare i target `exp_*` definiti nel Makefile.

Come indicato nella [guida Scaling di HPC Wiki](https://hpc-wiki.info/hpc/Scaling),
la dimensione del problema resta identica, i core crescono per potenze di due
e lo speedup della curva finale usa il punto a un core come baseline.

### Strong scaling OpenMP

Il problema rimane fisso a 32.000 clienti e i thread crescono per potenze di
due. Vengono usati un rank MPI e cinque run per configurazione:

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=1 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t1 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t1/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t1/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 2 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=2 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t2 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t2/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t2/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 4 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=4 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t4 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t4/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t4/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 8 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t8 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t8/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t8/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 16 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=16 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t16 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t16/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t16/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_openmp_t32 SOLVE_CSV_DIR=results/manual_campaign/strong_openmp/t32/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_openmp/t32/solutions"
```

### Strong scaling MPI

Il problema rimane fisso a 32.000 clienti. Ogni rank usa 32 thread OpenMP e
occupa un nodo; i punti sono quindi 32, 64 e 128 core totali:

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_mpi_r1_t32 SOLVE_CSV_DIR=results/manual_campaign/strong_mpi/r1_t32/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_mpi/r1_t32/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 2 --ntasks 2 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=2 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_mpi_r2_t32 SOLVE_CSV_DIR=results/manual_campaign/strong_mpi/r2_t32/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_mpi/r2_t32/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=32000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=strong_mpi_r4_t32 SOLVE_CSV_DIR=results/manual_campaign/strong_mpi/r4_t32/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/strong_mpi/r4_t32/solutions"
```

### Curva strong combinata finale

Non servono job aggiuntivi: la curva finale unisce i sei risultati OpenMP e
gli ultimi due risultati MPI già sottomessi sopra. La mappatura dell'asse x è:

| Core totali | Nodi × task/nodo × thread/task | Risultato da usare |
| ---: | --- | --- |
| 1 | `1 × 1 × 1` | OpenMP `t1` |
| 2 | `1 × 1 × 2` | OpenMP `t2` |
| 4 | `1 × 1 × 4` | OpenMP `t4` |
| 8 | `1 × 1 × 8` | OpenMP `t8` |
| 16 | `1 × 1 × 16` | OpenMP `t16` |
| 32 | `1 × 1 × 32` | OpenMP `t32` |
| 64 | `2 × 1 × 32` | MPI `r2_t32` |
| 128 | `4 × 1 × 32` | MPI `r4_t32` |

Il punto MPI `r1_t32` serve come controllo di coerenza con il punto OpenMP a
32 core; per la curva combinata si usa il punto OpenMP.

CUDA non fa parte delle campagne di scaling.

## Weak scaling CPU

La guida HPC richiede di aumentare il lavoro insieme alle risorse, mantenendo
costante il lavoro per processing element. Per questo solver si mantiene fissa
l'istanza a 8.000 clienti e si scala la popolazione ACO a **32 formiche per
core**. In questo modo non si confonde il weak scaling con il costo
superlineare delle matrici rispetto al numero di clienti.

Tutti i punti eseguono 10 epoche fisse; il limite di 300 secondi resta soltanto
una protezione. La configurazione debole è quindi:

| Core totali | Configurazione | Formiche totali |
| ---: | --- | ---: |
| 1 | 1 rank × 1 thread | 32 |
| 2 | 1 rank × 2 thread | 64 |
| 4 | 1 rank × 4 thread | 128 |
| 8 | 1 rank × 8 thread | 256 |
| 16 | 1 rank × 16 thread | 512 |
| 32 | 1 rank × 32 thread | 1024 |
| 64 | 2 rank × 32 thread | 2048 |
| 128 | 4 rank × 32 thread | 4096 |

### Weak scaling OpenMP, intra-nodo

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 1 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=1 SOLVE_MPI_M=32 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c1_m32 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c1_m32/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c1_m32/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 2 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=2 SOLVE_MPI_M=64 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c2_m64 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c2_m64/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c2_m64/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 4 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=4 SOLVE_MPI_M=128 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c4_m128 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c4_m128/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c4_m128/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 8 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=8 SOLVE_MPI_M=256 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c8_m256 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c8_m256/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c8_m256/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 16 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=16 SOLVE_MPI_M=512 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c16_m512 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c16_m512/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c16_m512/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_M=1024 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_openmp_c32_m1024 SOLVE_CSV_DIR=results/manual_campaign/weak_openmp/c32_m1024/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_openmp/c32_m1024/solutions"
```

### Weak scaling MPI, inter-nodo

Il punto a 32 core con un rank serve come controllo rispetto a OpenMP. Per la
curva weak combinata si usano poi i punti MPI a 64 e 128 core.

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 1 --ntasks 1 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=1 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_M=1024 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_mpi_c32_r1_t32_m1024 SOLVE_CSV_DIR=results/manual_campaign/weak_mpi/c32_r1_t32_m1024/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_mpi/c32_r1_t32_m1024/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 2 --ntasks 2 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=2 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_M=2048 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_mpi_c64_r2_t32_m2048 SOLVE_CSV_DIR=results/manual_campaign/weak_mpi/c64_r2_t32_m2048/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_mpi/c64_r2_t32_m2048/solutions"
```

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi --time 00:30:00 --nodes 4 --ntasks 4 --cpus 32 --mem 32G \
  --make-args "SOLVE_CLIENTS=8000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_M=4096 SOLVE_MPI_FIXED_EPOCHS=10 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=5 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=0 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001 SOLVE_BATCH_ID=weak_mpi_c128_r4_t32_m4096 SOLVE_CSV_DIR=results/manual_campaign/weak_mpi/c128_r4_t32_m4096/csv SOLVE_SOLUTIONS_DIR=results/manual_campaign/weak_mpi/c128_r4_t32_m4096/solutions"
```

Per il grafico weak finale usare i punti OpenMP da 1 a 32 core e i punti MPI
da 64 e 128 core. L'efficienza weak è `T(1)/T(N)`: idealmente il tempo resta
costante perché ogni core continua a gestire 32 formiche.


## Benchmark CUDA a configurazione fissa

Per un confronto aggiuntivo su una sola istanza, senza misurare lo scaling,
usare una singola GPU, mantenere fissa l'istanza e ripetere la misura cinque
volte:

```bash
tools/batch/submit_solve.sh \
  --target solve_cuda \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --gres gpu:1 \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_CUDA_REPEATS=5 SOLVE_CUDA_RUNTIME_S=300 SOLVE_CUDA_STAGNATION_EPOCHS=500 SOLVE_CUDA_MIN_REL_IMPROVEMENT=0.001 CUDA_ARCH=sm_75"
```

## Caricamento esplicito dei moduli

Il job prova automaticamente a caricare `/home/guest/init-hpc.sh`. Se questo
file non è disponibile o non configura tutti i compilatori, aggiungere ai
comandi di invio l'opzione seguente, adattando le versioni ai moduli installati:

```bash
--module-loads "gcc/13.2 openmpi/4.1 cuda/12.2"
```

Esempio:

```bash
tools/batch/submit_solve.sh \
  --target solve_mpi \
  --time 00:30:00 \
  --nodes 4 \
  --ntasks 4 \
  --cpus 32 \
  --mem 32G \
  --module-loads "gcc/13.2 openmpi/4.1" \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_MPI_RANKS=4 SOLVE_MPI_OMP_THREADS=32 SOLVE_MPI_LAUNCHER=srun SOLVE_MPI_REPEATS=3 SOLVE_MPI_RUNTIME_S=300 SOLVE_MPI_STAGNATION_EPOCHS=500 SOLVE_MPI_MIN_REL_IMPROVEMENT=0.001"
```

## Controllo preventivo

Per stampare il comando `sbatch` senza inviare il job, aggiungere `--dry-run`:

```bash
tools/batch/submit_solve.sh \
  --target solve_seq \
  --time 00:30:00 \
  --nodes 1 \
  --ntasks 1 \
  --cpus 1 \
  --mem 32G \
  --make-args "SOLVE_CLIENTS=16000 SOLVE_SEQ_REPEATS=3 SOLVE_SEQ_RUNTIME_S=300 SOLVE_SEQ_STAGNATION_EPOCHS=500 SOLVE_SEQ_MIN_REL_IMPROVEMENT=0.001" \
  --dry-run
```

## Monitoraggio

Durante ogni run, i launcher mostrano in tempo reale una riga come questa:

```text
[progress][mpi] epoca=120 | epoche_rimanenti_prima_stop=387 | tempo_trascorso=42.6s | tempo_rimanente=257.4s | best_cost=18432.771
```

`epoche_rimanenti_prima_stop` indica quante epoche senza miglioramenti
significativi mancano allo stop per stagnazione. Il contatore torna al valore
massimo quando viene trovato un nuovo best cost. `tempo_rimanente` indica il
tempo residuo rispetto a `SOLVE_*_RUNTIME_S`; diventa zero anche quando il run
termina prima per stagnazione.

Elencare i job dell'utente corrente:

```bash
squeue -u "$USER"
```

Controllare contabilità e stato di un job terminato:

```bash
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS,ExitCode
```

Seguire i log sostituendo `JOB_ID` con l'identificativo restituito da `sbatch`:

```bash
tail -f "results/slurm/vrp_solve_JOB_ID.out"
tail -f "results/slurm/vrp_solve_JOB_ID.err"
```

## Risultati

I run separati scrivono in:

```text
results/solve_manifest/csv/
results/solve_manifest/solutions/
```

Le campagne di scaling manuali scrivono in:

```text
results/manual_campaign/TAG/
```

Il benchmark CUDA a configurazione fissa scrive invece nelle normali directory
`results/solve_manifest/`.
