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
* CUDA has been fully refactored, formatted, and validated against the local norm style.
* CUDA public structs were renamed to `struct s_*` / `t_*`.
* CUDA header guards, constants, prototypes, return syntax, and line-width have been cleaned and verified.
* All CUDA-related functions now accept at most 4 parameters, declare at most 5 local variables, and are strictly under 25 lines of length.
* Space-based indentation in CUDA files has been fully converted to real tab characters.

---

## 2. Remaining Major Violations

| Category | Remaining Issue | Priority |
| :--- | :--- | :--- |
| **Function signatures** | Core C helper signatures are clean; CUDA helper signatures are fully compliant. | - |
| **Function length** | Core C and CUDA functions are fully compliant (no functions exceed 25 lines). | - |
| **Variable count** | Core C and CUDA functions are fully compliant (no functions exceed 5 variables). | - |
| **Formatting** | Space indentation in CUDA files was converted to real tab characters. Inconsistent blank lines and combined declarations have been cleaned. | - |
| **Return style** | Core C and CUDA are clean (using parenthesized return statements). | - |
| **Line width** | Core C and CUDA are clean. | - |
| **Naming** | Naming conforms to snake_case and `s_` / `t_` prefix rules. | - |

---

## 3. Remaining Refactoring Plan

All phases of the refactoring plan have been successfully executed:
1. **Phase 1 (Signature Cleanup)**: Complete. Struct bundling was introduced for high-arity CUDA parameters, ensuring all functions have <= 4 parameters.
2. **Phase 2 (Function Length and Variables)**: Complete. All functions in `src/cuda` and `src/main_cuda.cu` have been decomposed into smaller helpers under 25 lines and with <= 5 local variables.
3. **Phase 3 (Formatting Pass)**: Complete. Tabulation formatting, Allman brace style, return statements, and variable declaration placement/alignment rules have been fully satisfied.
4. **Phase 4 (Scope Decision)**: Complete. Core C and CUDA scopes are fully compliant.

All targets in the chosen norm scope compile and run successfully.

