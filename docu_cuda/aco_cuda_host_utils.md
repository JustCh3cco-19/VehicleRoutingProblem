# Documentazione `aco_cuda_host_utils.cu` e `.h`

## Descrizione Generale
Questo modulo fornisce funzioni di utilità eseguite sull'host per facilitare la gestione dei dati tra le strutture dati del progetto (spesso matrici 2D allocate dinamicamente) e i requisiti della GPU (che preferisce array 1D contigui).

## Funzioni Implementate

### `flatten_matrix`
```c
double* flatten_matrix(double **matrix, int n);
```
- **Scopo**: Converte una matrice 2D (array di puntatori a array) in un array 1D piatto e contiguo in memoria.
- **Dettagli**: 
  - Alloca memoria sull'host per `(n+1)*(n+1)` elementi di tipo `double`.
  - Copia i valori dalla struttura `matrix[i][j]` alla posizione `flat[i * (n+1) + j]`.
  - È fondamentale per permettere un trasferimento efficiente verso la memoria device tramite `cudaMemcpy`.
- **Ritorno**: Puntatore all'array piatto allocato, o `NULL` in caso di fallimento dell'allocazione.

### `deposit_pheromones_host`
```c
void deposit_pheromones_host(double *tau_host, const Solution *best_solution, double deposit, int n);
```
- **Scopo**: Esegue l'aggiornamento (deposito) del feromone su una matrice host rappresentata come array piatto.
- **Utilizzo**: Sebbene il deposito principale avvenga sulla GPU tramite kernel, questa funzione può essere utilizzata per test di validazione o in fasi di sincronizzazione particolari dove l'aggiornamento deve essere riflesso anche su una struttura dati host.
- **Dettagli**: Itera su tutte le rotte della `best_solution` e aggiunge il valore `deposit` ad entrambi gli archi `(u, v)` e `(v, u)` (grafo non orientato).

## Considerazioni sulla Gestione Memoria
La funzione `flatten_matrix` alloca memoria utilizzando `malloc`. È responsabilità del chiamante liberare questa memoria tramite `free` una volta terminata la copia sul device o l'uso sull'host, per evitare memory leak.
