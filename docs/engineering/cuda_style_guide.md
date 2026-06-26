# CUDA Style Guide

Questa guida definisce le regole locali per il codice CUDA del progetto
Vehicle Routing Problem. Si basa sulla NVIDIA CUDA C++ Best Practices Guide:

```text
https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
```

## Principi Generali

### Seguire Il Ciclo APOD

Ogni modifica CUDA deve seguire il ciclo Assess, Parallelize, Optimize, Deploy:

1. misurare prima di ottimizzare;
2. identificare hotspot reali;
3. parallelizzare solo lavoro sufficiente;
4. ottimizzare iterativamente;
5. verificare correttezza e performance dopo ogni modifica.

Non introdurre ottimizzazioni complesse senza un dato di profiling o benchmark
che le giustifichi.

### Correttezza Prima Della Performance

Ogni kernel o modifica host/device deve preservare:

- validita' CVRP della soluzione;
- costo numerico finito;
- assenza di clienti duplicati o mancanti;
- rispetto della capacita' veicolo;
- confronto con baseline sequenziale entro la tolleranza stabilita quando il
  confronto numerico e' applicabile.

Differenze floating-point tra CPU e GPU sono ammissibili solo se motivate e
misurate. Non assumere identita' bitwise tra CPU e CUDA.

## Error Handling

### Controllare Ogni Chiamata CUDA

Ogni chiamata runtime CUDA deve essere controllata:

```c
CHECK_CUDA(cudaMalloc(...));
CHECK_CUDA(cudaMemcpy(...));
CHECK_CUDA(cudaDeviceSynchronize());
```

Dopo ogni kernel launch chiamare `cudaGetLastError()` prima della sincronizzazione:

```c
kernel<<<grid, block>>>(...);
CHECK_CUDA_KERNEL();
CHECK_CUDA(cudaDeviceSynchronize());
```

La guida NVIDIA nota che gli esempi omettono spesso error checking per
concisione, ma il codice production deve controllare sistematicamente API CUDA
e kernel launch.

### Cleanup Unico

Le funzioni host CUDA devono avere un unico blocco `cleanup` per liberare tutte
le risorse host/device. Evitare return immediati dopo allocazioni parziali.

Schema richiesto:

```c
AcoStatus status = ACO_OK;
DeviceType *d_ptr = NULL;
HostType *h_ptr = NULL;

CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
h_ptr = malloc(bytes);
if (!h_ptr) {
  status = ACO_ERR_ALLOCATION;
  goto cleanup;
}

cleanup:
  cudaFree(d_ptr);
  free(h_ptr);
  return status;
```

## Timing E Benchmark

### Usare Wall-Clock Corretto

Le chiamate CUDA e i kernel launch sono asincroni. Se si misura con timer CPU,
sincronizzare prima di iniziare e prima di fermare il timer.

Per microbenchmark CUDA preferire CUDA events. Per misure end-to-end del solver
usare wall-clock includendo anche trasferimenti host/device, perche' e' il tempo
reale percepito dal programma.

### Misurare Bandwidth Ed Efficienza

Quando il kernel e' memory-bound, riportare:

- requested/global load throughput;
- requested/global store throughput;
- global memory load/store efficiency;
- occupancy;
- achieved occupancy;
- tempo kernel;
- tempo trasferimenti host/device.

Usare Nsight Systems per timeline end-to-end e Nsight Compute per metriche di
kernel.

## Memory Access

### Coalescing

Gli accessi global memory dei thread nello stesso warp devono essere coalesced
quando possibile. Per compute capability 6.0+, accessi contigui a word da 4
byte in un warp sono serviti da transazioni da 32 byte.

Regole locali:

- preferire layout struct-of-arrays per dati letti da lane consecutive;
- evitare stride grandi tra lane dello stesso warp;
- documentare i casi in cui il pattern non puo' essere coalesced;
- misurare global memory load/store efficiency prima e dopo modifiche.

### Shared Memory

Usare shared memory quando serve a:

- rendere coalesced accessi che sarebbero strided;
- eliminare letture ridondanti da global memory;
- ridurre traffico globale in dati riusati da piu' thread.

Non usare shared memory se riduce occupancy senza beneficio misurabile.

### Evitare Local Memory Accidentale

Array automatici grandi o indicizzati dinamicamente possono finire in local
memory, che e' off-chip e costosa. Nei kernel VRP evitare array per-thread
troppo grandi. Se servono buffer temporanei grandi, valutare:

- shared memory;
- compressione del dato;
- riduzione del candidate set;
- ristrutturazione warp-level.

## Kernel Design

### Funzioni Piccole E Testabili

Preferire kernel composti da funzioni `__device__` piccole. Quando possibile,
rendere funzioni matematiche pure `__host__ __device__` per testarle anche lato
CPU.

Esempi nel progetto:

- selezione candidata;
- calcolo distanza;
- quantizzazione/dequantizzazione pheromone;
- operazioni bitset visited.

### Warp-Level Programming

Quando si usa un modello warp-per-ant:

- documentare quali lane sono attive;
- usare primitive warp (`__shfl_sync`, ballot, prefix) con maschere esplicite;
- evitare divergenza non necessaria dentro il warp;
- garantire che le lane inattive non leggano memoria fuori range.

### Block Size

Il block size deve essere una costante documentata o un parametro motivato.
Ogni modifica a `CUDA_THREADS_PER_BLOCK` deve essere accompagnata da benchmark
su almeno:

- istanza piccola;
- istanza media;
- istanza grande;
- metriche occupancy e tempo kernel.

## Host/Device Transfers

Minimizzare trasferimenti tra host e device:

- non copiare route complete a ogni iterazione se basta copiare il best ant;
- evitare trasferimenti diagnostici nel percorso benchmark;
- usare pinned memory solo se misurata utile;
- valutare stream/async copy solo quando c'e' sovrapposizione reale con compute.

Per il solver VRP, distinguere sempre:

- tempo preprocess candidate list;
- tempo iterazioni ACO;
- tempo trasferimenti finali;
- tempo validazione host.

## Numerica

### Float Vs Double

CUDA usa spesso `float` per throughput. CPU usa spesso `double`. Le differenze
sono attese:

- confrontare risultati con tolleranza;
- non pretendere bitwise equality;
- indicare precisione effettiva nei benchmark;
- evitare conversioni implicite non documentate.

### FMA E Ordine Operazioni

Le operazioni floating-point non sono associative. Parallelizzazioni e FMA
possono cambiare leggermente il risultato. Ogni riduzione o confronto numerico
deve avere tolleranza esplicita.

## Naming E Organizzazione

### File

- `src/cuda/aco_cuda.cu`: host-side backend CUDA.
- `src/cuda/aco_cuda_kernels.cu`: kernel e device functions.
- `include/aco_cuda_kernels.h`: tipi e launcher CUDA interni.
- entrypoint CLI CUDA: preferire `src/main_cuda.cu`, non dentro `src/cuda/`.

### Nomi

- kernel: prefisso `kernel_`;
- launcher host: prefisso `launch_`;
- device helper: nome descrittivo con prefisso `cuda_` quando e' condiviso;
- tipi CUDA: prefisso `Cuda`;
- costanti CUDA: prefisso `CUDA_`.

## Regole Per Il Progetto VRP

### Candidate List

La candidate list CUDA deve dichiarare:

- `cand_k` effettivo;
- memoria usata;
- criterio di selezione;
- fallback quando i candidati sono esauriti.

### Pheromone

La rappresentazione quantizzata `uint8_t` deve documentare:

- range logaritmico;
- saturazione min/max;
- errore introdotto dalla quantizzazione;
- impatto sul confronto con CPU.

### Validazione

Il backend CUDA deve sempre produrre una soluzione validabile lato host con:

- `solution_validate_cvrp()`;
- costo numerico presente;
- exit status non-zero in caso di errore CUDA o soluzione mancante.

## Checklist Prima Di Commit

- `make cuda` compila.
- `make smoke_cuda` passa o viene skipped per assenza GPU.
- Ogni chiamata CUDA e kernel launch e' controllata.
- Non ci sono return anticipati che saltano cleanup.
- Le nuove allocazioni sono inizializzate a `NULL`.
- Le nuove metriche di performance sono documentate.
- La correttezza e' confrontata con sequenziale o validatore CVRP.
- Se il kernel non scala, il report spiega perche'.
