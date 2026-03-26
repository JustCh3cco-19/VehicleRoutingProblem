# GEMINI Context: Vehicle Routing Problem (VRP) Solver

This project implements a sequential Ant Colony Optimization (ACO) algorithm to solve the Vehicle Routing Problem (VRP) in C.

## Project Overview

- **Core Algorithm:** Ant Colony Optimization (ACO) applied to VRP.
- **Technology Stack:** C11, Standard C libraries, GCC, Make.
- **Key Components:**
  - `src/aco.c`: Main ACO logic (pheromone update, path selection).
  - `src/solution.c`: Logic for VRP solutions (routes, costs, validation).
  - `src/matrix.c`: Utility for 2D matrix allocation and management.
  - `tests/test_aco.c`: Comprehensive test suite including an exact VRP solver for small instances.

## Building and Running

The project uses a `Makefile` for all build-related tasks.

### Primary Commands
- **Build Sequential Executable:**
  ```sh
  make
  ```
  Generates the `./aco_vrp_seq` binary.

### Testing and Validation
- **Run Tests:**
  ```sh
  make test
  ```
  Compiles and executes the test suite in `tests/test_aco.c`.
- **Code Coverage:**
  ```sh
  make coverage
  ```
  Runs tests and generates `gcov` coverage data.
- **Debug Build:**
  ```sh
  make debug
  ```
  Compiles with `-DDEBUG` flag enabled.

### Cleaning Up
- **Clean Build Artifacts:**
  ```sh
  make clean
  ```

## Development Conventions

- **Standard:** C11 (`-std=c11`).
- **Includes:** Local headers are located in `include/`.
- **Memory Management:** Matrices and Solutions are dynamically allocated. Always use `matrix_free()` and `solution_free()` to prevent leaks.
- **Error Handling:** Check for allocation failures (e.g., in `aco_vrp_sequential`).
- **Naming:**
  - `aco_...`: Functions related to the ACO algorithm.
  - `solution_...`: Functions for managing VRP solutions.
  - `route_...`: Functions for managing individual vehicle routes.
- **Testing:** New features or bug fixes should be validated by adding test cases to `tests/test_aco.c`.

## Architecture Notes

- The problem instance is defined by a distance/cost matrix `c`.
- A `Solution` consists of `K` routes, each starting and ending at the depot (node 0).
- The `aco_vrp_sequential` function is the entry point for the solver.
- The `Makefile` contains placeholders for `MPICC`, `CUDACC`, and `OMPFLAG`, suggesting the project might serve as a scaffold for future parallel implementations.
