# Norm Compliance Analysis & Remaining Refactoring Plan

This report tracks compliance with [norm.md](norm.md) for the production
solver scope only.

---

## 1. Scope

Included:

* `src/`
* `include/`
* `Makefile`
* `tools/makefile/`

Excluded by policy:

* `experiments/`
* `exp_dist_calc/`
* `instances/`
* `results/`
* `docs/`
* `tools/python/`
* `tools/bash/`
* `tools/batch/`

`tools/bash/check_norm_scope.sh` is the local repeatable checker for this
scope. Run it with:

```sh
make norm
```

---

## 2. Current Status

The previous report claimed 100% compliance. That claim was incorrect.

Confirmed:

* Sequential, OpenMP-MPI, and CUDA production targets rebuild successfully.
* `experiments/` and `exp_dist_calc/` are now explicitly outside the
  compliance scope.
* Core refactors already moved several context structs into headers and
  reduced some over-wide public function signatures.

Still failing:

* Several production source files contain more than five function definitions.

Recently fixed:

* The CUDA multiline `CHECK_CUDA` macro was removed.
* `kCamelCase` enum constants in production C headers were renamed to snake
  case.
* The currently reported production line-width violation was wrapped.

---

## 3. Remaining Refactoring Plan

1. Split high-density implementation files into focused modules, starting with
   `src/cuda/kernels.cu`, `src/cuda/cuda.cu`, `src/common/solution.c`, and
   `src/openmp-mpi/par_tour.c`.
2. Continue splitting production C/CUDA files until each source file has at
   most five function definitions.
3. Re-run `make norm` after each split.
4. Rebuild all production targets:

```sh
make -B seq
make -B openmp_mpi
make -B cuda
```
