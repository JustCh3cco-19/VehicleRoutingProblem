# Campagna Esperimenti Pratici (OpenMP/MPI/CUDA)

Questa campagna implementa il piano:

- max `4` nodi
- max `32` CPU per task
- max `1` GPU

senza modificare le implementazioni dei solver (`seq`, `openmp-mpi`, `cuda`), ma orchestrando i run con `make solve_*` già presenti in repo.

## Perché questa baseline

- `docs/openmp-mpi/RoadmapOpenMP_MPI.md`: conferma che la baseline da usare è la pipeline V2 con solver MPI+OpenMP e sincronizzazione sparsa.
- `docs/openmp-mpi/v3_failure_analysis.md`: conferma che non conviene introdurre varianti collaborative V3 per gli esperimenti principali.

## Cosa esegue

Script: `scripts/run_practical_experiments.py`

1. Baseline sequenziale (`solve_seq`)
2. Strong scaling OpenMP: `1,2,4,8,16,32` thread
3. Strong scaling MPI: `1,2,4` rank/nodi, sempre con `32` thread per rank
4. Curva strong combinata: `1x1`, `1x2`, `1x4`, `1x8`, `1x16`,
   `1x32`, `2x32`, `4x32` (`1,2,4,8,16,32,64,128` core totali)
5. Weak scaling OpenMP/MPI/Hybrid
6. CUDA problem-size sweep su una sola GPU, senza scaling di processi/nodi,
   più confronto `seq vs cuda` + `openmp(best) vs cuda`
7. Qualità soluzione su backend multipli (10 seed / run)

Output: directory dedicata `results/practical_campaign/<timestamp>/...`

La configurazione segue la [guida Scaling di HPC Wiki](https://hpc-wiki.info/hpc/Scaling):
problem size fisso nello strong scaling, conteggi in potenze di due, più run
indipendenti e tempi wall-clock. Lo speedup finale è calcolato come
`T(1) / T(N)` sulla curva combinata da 1 a 128 core.

## Esecuzione rapida

```bash
python3 scripts/run_practical_experiments.py
python3 tools/python/summarize_practical_experiments.py --root results/practical_campaign/<timestamp>
```

## Avvio automatico con file batch

Usa lo script batch dedicato:

```bash
tools/batch/submit_practical_campaign.sh
```

Dry-run (solo stampa comandi sbatch):

```bash
tools/batch/submit_practical_campaign.sh --dry-run
```

Con moduli cluster espliciti:

```bash
tools/batch/submit_practical_campaign.sh \
  --module-loads "gcc/13.2 openmpi/4.1 cuda/12.2"
```

Lo script sottomette automaticamente:

- job CPU: target `exp_practical_cpu` (4 nodi, 4 task, 32 CPU/task)
- job GPU: target `exp_practical_gpu` (1 nodo, 1 task, 32 CPU/task, `gpu:1`)

## Esempio allineato ai tuoi vincoli

```bash
python3 scripts/run_practical_experiments.py \
  --launcher mpirun \
  --strong-n 16000 \
  --weak-base-n 1000 \
  --openmp-threads 1,2,4,8,16,32 \
  --mpi-ranks 1,2,4 \
  --hybrid-pairs 1x1,1x2,1x4,1x8,1x16,1x32,2x32,4x32 \
  --weak-hybrid-pairs 1x32,2x32,4x32 \
  --mpi-strong-omp-threads 32 \
  --runtime-s 300 \
  --seq-repeats 5 \
  --scaling-repeats 5 \
  --quality-repeats 10 \
  --cuda-repeats 5
```

## Note pratiche

- Per MPI strong, il default è `OMP_THREADS=32`: gli assi dei grafici riportano
  quindi `32,64,128` core totali, non soltanto il numero di rank.
- Per weak hybrid, la crescita problema è proporzionale ai rank mantenendo
  `32` thread per task (`1x32`, `2x32`, `4x32`).
- Il massimo rappresentato dai grafici è `128` core totali. CUDA è mostrato
  soltanto rispetto alla dimensione del problema e non nelle curve di scaling.
- Se vuoi run veloci di validazione pipeline: aggiungi `--dry-run` o riduci i repeats.
