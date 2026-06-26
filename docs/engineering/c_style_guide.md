# C Style Guide

Questa guida definisce le regole locali per il codice C del progetto Vehicle
Routing Problem. Si applica a `src/seq`, `src/openmp-mpi`, `src/common` e agli
header in `include`.

## Obiettivi

- Rendere chiari ownership e cleanup delle risorse.
- Evitare simboli globali ambigui in un progetto senza namespace C.
- Mantenere il codice leggibile anche nelle parti ottimizzate.
- Ridurre macro e inizializzazioni fragili.

## Naming

### Funzioni, Struct, Enum E Typedef

Funzioni, `struct`, `enum` e `typedef` devono usare `lower_snake_case`.

```c
typedef struct aco_solution_view {
  int route_count;
} aco_solution_view_t;

static int choose_candidate_count(int n);
```

I tipi interni a un backend devono avere un prefisso coerente:

- codice comune/API: `aco_`;
- sequenziale interno: `seq_`;
- OpenMP-MPI interno: `aco_mpi_`;
- CUDA: seguire `docs/engineering/cuda_style_guide.md`.

Evitare nuovi typedef in stile `PascalCase` come `MatrixFloat` o
`HierarchicalWorkspace`. Per tipi interni usare nomi espliciti, ad esempio
`aco_mpi_matrix_float_t`.

### Simboli Pubblici

Ogni simbolo dichiarato in un header pubblico deve condividere un prefisso
univoco del modulo. In C non ci sono namespace, quindi il prefisso e' parte
dell'interfaccia.

```c
typedef enum aco_status {
  ACO_OK = 0,
  ACO_ERR_ALLOCATION,
} AcoStatus;

AcoStatus aco_vrp_run(...);
```

Non esportare helper interni dagli header pubblici. Se una struttura serve solo
al solver, tenerla in un header interno come `aco_internal.h` o nel `.c`.

### Costanti

Preferire `enum` per costanti intere locali invece di macro del preprocessore.

```c
enum {
  kAcoMpiAlignment = 64,
  kAcoMpiMaxCandidates = 512,
};
```

Le macro restano accettabili per:

- guardie di include;
- feature flag di compilazione;
- costanti floating-point dove un `enum` non e' applicabile;
- macro di error checking con semantica da singola istruzione.

## Preprocessore

Le macro funzione vanno evitate quando una funzione `static` e' sufficiente. Se
una macro e' necessaria, deve essere igienica:

```c
#define CHECK_CONDITION(condition_) \
  do {                              \
    if (!(condition_)) {            \
      goto cleanup;                 \
    }                               \
  } while (0)
```

Regole:

- racchiudere gli argomenti tra parentesi quando vengono espansi;
- usare variabili locali con suffisso `_`;
- usare `do { ... } while (0)` per macro multilinea;
- non terminare la definizione con `;`.

## Inizializzazione

Quando si inizializza una `struct` o `union`, usare sempre designated
initializers. Non usare inizializzazioni posizionali per struct.

```c
aco_mpi_sparse_delta_t delta = {
  .edge_idx = edge_idx,
  .increment = increment,
};
```

Per array, i designated initializers sono consigliati quando rendono esplicito
il significato degli indici.

```c
MPI_Datatype types[2] = {
  [0] = MPI_UINT32_T,
  [1] = MPI_FLOAT,
};
```

## Funzioni

Le dichiarazioni di funzioni senza parametri devono usare `void`.

```c
static double wall_time_seconds(void);
```

Evitare dichiarazioni in stile K&R come:

```c
static double wall_time_seconds();
```

Le funzioni `static` non devono essere marcate `inline`: il compilatore puo'
inlinarle comunque.

## Header

Gli header devono contenere solo interfacce necessarie ai chiamanti.

Regole:

- non dichiarare funzioni o variabili `static` in header pubblici;
- evitare definizioni di dati globali negli header;
- mantenere parametri e nomi coerenti tra dichiarazione e definizione;
- spostare dettagli interni in header interni o file `.c`.

## Allocazione E Cleanup

Ogni funzione che alloca piu' risorse deve avere cleanup robusto su allocazioni
parziali. Non dereferenziare mai il risultato di `malloc`, `calloc` o
`aligned_alloc` prima di averlo controllato.

```c
aco_mpi_matrix_float_t *matrix_create_float(int n) {
  aco_mpi_matrix_float_t *m = malloc(sizeof(*m));
  if (!m) {
    return NULL;
  }

  m->data = aligned_alloc(kAcoMpiAlignment, bytes);
  m->rows = malloc(row_count * sizeof(*m->rows));
  if (!m->data || !m->rows) {
    free(m->data);
    free(m->rows);
    free(m);
    return NULL;
  }

  return m;
}
```

Per funzioni lunghe con molte risorse, preferire un unico blocco `cleanup`.

## Builtin E Attributi Non Standard

Non usare direttamente builtin non standard in piu' punti del codice. Se serve
un builtin, creare un wrapper unico con fallback documentato.

Esempio:

```c
static uint32_t aco_bswap32(uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap32(value);
#else
  return ((value & 0x000000ffu) << 24) |
         ((value & 0x0000ff00u) << 8) |
         ((value & 0x00ff0000u) >> 8) |
         ((value & 0xff000000u) >> 24);
#endif
}
```

Gli attributi non standard devono essere supportati almeno da GCC e Clang o
protetti da `#if` espliciti.

## Formattazione

Usare indentazione a 2 spazi per il codice C nuovo o modificato. Quando si
tocca una funzione, evitare blocchi compressi su una sola riga se contengono
piu' istruzioni o cleanup.

Accettabile:

```c
if (!ptr) {
  return NULL;
}
```

Da evitare:

```c
if (!ptr) return NULL;
```

Il repository al momento non contiene una configurazione `.clang-format`
affidabile. Prima di introdurre formatting automatico, aggiungere una
configurazione condivisa e applicarla in modo separato dalle modifiche
funzionali.

## Regole Specifiche Del Progetto

- Il backend sequenziale deve tenere helper e workspace interni con prefisso
  `seq_` quando il simbolo non e' gia' chiaramente locale.
- Il backend OpenMP-MPI deve usare prefisso `aco_mpi_` per tipi e helper
  condivisi dentro il file.
- Le funzioni pubbliche devono restare coerenti con `include/aco.h`.
- Le validazioni CVRP devono passare da `solution_validate_cvrp()` quando si
  controlla una soluzione prodotta dal solver.
- Le modifiche che cambiano risultati numerici devono essere confrontate con il
  sequenziale o motivate nel report di benchmark.

## Checklist Prima Di Commit

- `make seq` compila.
- `make openmp_mpi` compila.
- `make smoke` passa o documenta gli skip ambientali.
- Le nuove struct literal usano designated initializers.
- Le nuove costanti intere usano `enum`, non macro.
- Le nuove allocazioni controllano il valore restituito.
- I dettagli interni non vengono aggiunti agli header pubblici.
