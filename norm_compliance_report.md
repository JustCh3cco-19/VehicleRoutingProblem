# Norm Compliance Analysis & Remaining Refactoring Plan

This report tracks the remaining work needed to align the current
`VehicleRoutingProblem` C codebase with [norm.md](norm.md). Completed refactors
have been removed from the action plan.

---

## 1. Current Status

Several previously listed violations have already been addressed:

* Public solver entrypoints now use `t_solver_params` instead of long argument
  lists.
* Sequential shared/workspace/solution types use `struct s_*` and `t_*` naming.
* `src/common/solution.c`, `src/common/instance_parser.c`,
  `src/common/cli_common.c`, `src/common/matrix.c`, and `src/main.c` have been
  substantially refactored toward the local norm style.
* Wrapper/helper split files introduced during refactoring were removed; the
  original compact file structure for `main.c` and `matrix.c` was restored.
* The project-specific norm now explicitly allows `for` loops when they improve
  readability, provided loop variables are declared before the loop header.

---

## 2. Remaining Major Violations

| Category | Remaining Issue | Priority |
| :--- | :--- | :--- |
| **Function signatures** | Some public/helper APIs still exceed 4 parameters, notably validation and CUDA-related functions. | High |
| **Function length** | Algorithm/orchestrator functions in sequential, OpenMP/MPI, CUDA, and parser/helper paths still need line-count review. | High |
| **Variable count** | Several functions still declare more than 5 local variables. | High |
| **Formatting** | Some files still contain declaration+initialization, inconsistent blank lines, or non-tab indentation. | Medium |
| **Return style** | Some C/CUDA files still use `return value;` instead of `return (value);`. | Medium |
| **Line width** | Lines over 80 columns remain, especially in CUDA headers/sources and selected OpenMP/MPI files. | Medium |
| **Naming** | CUDA types such as `CudaParams` remain PascalCase if CUDA is included in the norm scope. | Medium |

---

## 3. Remaining Refactoring Plan

### Phase 1: Finish Signature Cleanup

1. Bundle remaining high-arity validation parameters into small context structs.
2. Decide whether CUDA host/kernel launcher APIs are in scope for this norm pass.
3. If CUDA is in scope, rename PascalCase CUDA structs and typedefs to
   `struct s_*` / `t_*`.

### Phase 2: Audit Function Length and Variables

1. Re-scan `src/seq`, `src/openmp-mpi`, `src/cuda`, and `src/common` for
   functions over 25 lines.
2. Split only large or complex files/functions; do not split compact utility
   files solely because they exceed 5 functions.
3. Reduce local variable count where functions still exceed 5 declarations.

### Phase 3: Formatting Pass

1. Replace remaining declaration+initialization patterns where they violate
   `norm.md`.
2. Normalize return syntax to `return (...)`.
3. Fix remaining line-width issues.
4. Normalize indentation to real tab characters in C files under the chosen
   norm scope.

### Phase 4: Scope Decision

Explicitly decide whether the following paths must be norm-compliant:

* `src/cuda/` and `include/cuda/`
* `experiments/`
* generated/benchmark utilities

If they are out of scope, document that exclusion before doing further cleanup.
