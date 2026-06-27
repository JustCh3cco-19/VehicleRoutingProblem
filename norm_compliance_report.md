# Norm Compliance Analysis & Refactoring Plan

This report analyzes the `VehicleRoutingProblem` codebase against the rules defined in [norm.md](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/norm.md) (the 42 Coding Norm) and provides a detailed, step-by-step plan to bring the code into full compliance.

---

## 1. Summary of Major Violations

An inspection of the codebase reveals that the code does not conform to the 42 Norm. The table below highlights the most critical systemic violations across the codebase:

| Category | Norm Rule | Code Status / Violation | Severity |
| :--- | :--- | :--- | :--- |
| **Indentation** | Real `\t` (tabulations) of width 4. | Code uses spaces (2 or 4 spaces) everywhere. | Critical |
| **Naming** | Structs start with `s_`, typedefs with `t_`, lowercase snake_case only. | Structs/types use PascalCase (e.g., `Solution`, `SeqShared`). Capitalized parameter names (e.g., `K`, `Q` in `aco_vrp`) are present. | High |
| **Formatting** | No declaration & initialization on the same line. | Commonly used: `int remaining = shared->n;` or `int stagnation_iters = 0;`. | High |
| **Formatting** | Declarations followed by one empty line; no other empty lines inside functions. | Almost all functions contain multiple empty lines to separate logic blocks. | High |
| **Functions** | Max 4 named parameters. | `aco_vrp` has 12 parameters; `aco_vrp_with_capacity` has 13 parameters. | Critical |
| **Functions** | Max 25 lines per function body (excluding braces). | Orchestrator and algorithm functions (e.g., `aco_vrp_run_with_config`, `build_ant_solution`) are significantly longer. | Critical |
| **Functions** | Max 5 variables declared per function. | Core algorithm loops declare 10+ variables. | High |
| **Syntax** | `return` must use parentheses: `return (value);`. | Code uses raw returns: `return value;`. | Medium |
| **Headers** | Every file must start with the standard 42 header comment. | No files contain the standard 42 header. | High |
| **Comments** | No comments are allowed inside function bodies. | Comments are frequently used inside function bodies to explain logic. | Medium |
| **File Structure** | Max 5 functions defined per `.c` file. | Files like `aco_seq_utils.c` and helper libraries have more than 5 functions. | Medium |

---

## 2. Refactoring Plan: Step-by-Step

Bringing a complex codebase like this under the 42 Norm requires a structured, phase-by-phase approach to avoid introducing compilation bugs or regression errors.

### Phase 1: Structure & Signature Refactoring (Function Signatures & Types)
The most challenging constraint is the **limit of 4 parameters** per function. We must bundle parameters into structs.

1. **Define Parameter Structs**:
   In [include/aco_internal.h](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/include/aco_internal.h), define structures starting with `s_` and typedefs with `t_` to group parameters:
   ```c
   typedef struct s_aco_params {
       int     n;
       int     k;
       int     m;
       double  alpha;
       double  beta;
       double  rho;
       double  tau0;
       double  q;
       unsigned int seed;
   } t_aco_params;
   ```
2. **Rename Existing Structures/Typedefs**:
   * Rename `SeqShared` to `t_seq_shared` and `struct s_seq_shared`.
   * Rename `SeqWorkspace` to `t_seq_workspace` and `struct s_seq_workspace`.
   * Rename `Solution` to `t_solution` and `struct s_solution`.
3. **Update Function Signatures**:
   Reduce parameters in `aco_vrp` and `aco_vrp_with_capacity` in [include/aco.h](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/include/aco.h):
   ```c
   t_aco_status  aco_vrp(t_aco_params *params, double **c, t_solution *best_sol, double *best_cost);
   ```

### Phase 2: Function Splitting & Variable Reduction
Every function must be split so it is **$\le$ 25 lines** long and uses **$\le$ 5 variables**.

1. **Split Orchestration Loops**:
   Break down `aco_vrp_run_with_config` (currently ~200 lines) in [src/seq/aco_sequential.c](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/src/seq/aco_sequential.c) into small helper functions:
   * `static void  init_pheromone_matrix(...)` (Initializes `tau`)
   * `static void  execute_ant_generation(...)` (Orchestrates the loop over `total_m` ants)
   * `static void  apply_pheromone_evaporation(...)` (Performs evaporation)
   * `static void  deposit_pheromones(...)` (Handles deposition for best and local paths)
2. **Split Tour Construction**:
   Break down `build_ant_solution` in [src/seq/aco_seq_tour.c](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/src/seq/aco_seq_tour.c):
   * Extract capacity constraint check logic.
   * Extract the fallback nearest customer search into a clean function.
3. **Split Utility Functions**:
   If a `.c` file contains more than 5 functions (e.g., [src/seq/aco_seq_utils.c](file:///home/justch3cco/Desktop/Universita/PSMC/VehicleRoutingProblem/src/seq/aco_seq_utils.c)), move excess functions to separate utility files or group them logically.

### Phase 3: Spacing, Comments, and Local Spacing Refactoring
1. **Move Comments**:
   Remove all inline comments inside function bodies. Move them to the header file or place them right above the function definitions.
2. **Refactor Local Variable Declarations**:
   * Move all declarations to the top of the function.
   * Do not initialize variables in the declaration line.
   * Align all variable names to the same column (using tabs).
   * Place exactly one empty line after declarations, and ensure no other empty lines exist inside the function body.
3. **Update Returns**:
   Enclose all return values in parentheses: `return (0);`, `return (true);`.

### Phase 4: Token Spacing & Indentation Conversion
1. **Token Spacing**:
   Add exactly one space around operators (`=`, `+`, `==`, `<`, etc.) and after commas/semicolons. Ensure no double spaces exist.
2. **Indentation Conversion**:
   Convert the entire workspace (all `.c` and `.h` files) from space-based indentation to **real tab characters (`\t`)** of width 4.
3. **Standard 42 Header**:
   Inject the standard 42 header comment at the top of each `.c` and `.h` file.

---

## 3. Before/After Code Example

To illustrate the transformation, here is how the utility function `aco_seq_choose_candidate_count` would be rewritten to follow the Norm.

### Before Refactoring (Non-compliant)
```c
int aco_seq_choose_candidate_count(int n, int requested_candidate_k) {
  if (requested_candidate_k > 0) {
    return aco_seq_clamp_int(requested_candidate_k, 1, n);
  }
  if (n <= kSeqDenseCandidateLimit) {
    return n;
  }
  return aco_seq_clamp_int(kSeqDefaultSparseCandidateCount, 1, n);
}
```

### After Refactoring (100% Compliant)
> [!NOTE]
> The code below uses real tabulations (`\t`) for indentation, separates return types and names with a tab, wraps return values in parentheses, and uses snake_case for all parameters.

```c
int	aco_seq_choose_candidate_count(int n, int req_k)
{
	if (req_k > 0)
		return (aco_seq_clamp_int(req_k, 1, n));
	if (n <= kSeqDenseCandidateLimit)
		return (n);
	return (aco_seq_clamp_int(kSeqDefaultSparseCandidateCount, 1, n));
}
```
