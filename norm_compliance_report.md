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
* `experiments/` is out of scope for norm compliance cleanup.
* CUDA remains in scope, but only after the core C code has been refined.
* CUDA public structs were renamed from PascalCase to `struct s_*` / `t_*`.
* CUDA header guards, constants, prototypes, return syntax, and line-width
  issues have received an initial cleanup pass.
* Core C return syntax has been scanned; no `return value;` occurrences remain
  in `include/`, `src/common/`, `src/seq/`, or `src/openmp-mpi/`.
* Core C line-width scan is clean for `src/common`, `src/seq`,
  `src/openmp-mpi`, and non-CUDA `include` headers.

---

## 2. Remaining Major Violations

| Category | Remaining Issue | Priority |
| :--- | :--- | :--- |
| **Function signatures** | Some helper APIs still exceed 4 parameters, especially CUDA-related functions. | High |
| **Function length** | Algorithm/orchestrator functions in sequential, OpenMP/MPI, CUDA, and parser/helper paths still need line-count review. | High |
| **Variable count** | Several functions still declare more than 5 local variables. | High |
| **Formatting** | Some files still contain declaration+initialization, inconsistent blank lines, or non-tab indentation. | Medium |
| **Return style** | Core C is clean; CUDA comments still mention return values in prose. | Low |
| **Line width** | Core C is clean; CUDA sources need a final verification scan after each CUDA edit. | Medium |
| **Naming** | CUDA PascalCase typedefs have been renamed; continue watching new CUDA API additions. | Low |

---

## 3. Remaining Refactoring Plan

### Phase 1: Finish Signature Cleanup

1. Bundle remaining high-arity validation parameters into small context structs.
2. Finish remaining core C helper signatures that exceed 4 parameters.
3. Keep CUDA APIs on the new `struct s_*` / `t_*` naming convention.

### Phase 2: Audit Function Length and Variables

1. Re-scan `src/seq`, `src/openmp-mpi`, and `src/common` for functions over 25
   lines. Current core C hits include:
   * `src/common/instance_parser.c`: parser helpers require final verification
   * `src/seq`: primary runner/epoch/context/tour functions have been split
     and no longer appear in the >25-line scan
   * `src/openmp-mpi`: solver loop, init, epoch, tour, and math helpers have
     been split and no longer appear in the >25-line scan
   * Current core C >25-line scan is clean
2. Split only large or complex files/functions; do not split compact utility
   files solely because they exceed 5 functions.
3. Reduce local variable count where functions still exceed 5 declarations.

### Phase 3: Formatting Pass

1. Replace remaining declaration+initialization patterns where they violate
   `norm.md`.
2. Normalize return syntax to `return (...)`.
3. Keep CUDA line-width checks clean during the CUDA phase.
4. Normalize indentation to real tab characters in C files under the chosen
   norm scope.

### Phase 4: Scope Decision

Current scope decision:

* `experiments/` is excluded.
* Core C code is cleaned first: `include/`, `src/common/`, `src/seq/`, and
  `src/openmp-mpi/`.
* CUDA is handled after the core C pass: `src/cuda/` and `include/cuda/`.

Generated or benchmark utilities should be explicitly classified before any
norm cleanup is attempted there.
