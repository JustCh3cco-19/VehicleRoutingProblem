VRP test files description (prof-style adaptation)
--------------------------------------------------

These files mirror the original professor test groups (`test_01..test_09`) in
the VRP context.

Why values are scaled
---------------------

The VRP solver uses a full `(n+1) x (n+1)` cost matrix, so memory is `O(n^2)`.
Using literal sizes like `30,000`, `1,000,000`, or `100,000,000` positions is
not feasible here. The workload classes are preserved with scaled values.

Mapping from original guide to this project
-------------------------------------------

- `test_01`:
  basic debug test with small instances, `n = 35`.
- `test_02`:
  small workload class mapped from `30k/20k` to `300/200` (`n = 300`, `m = 200`).
- `test_03` to `test_06`:
  border/partition stress on `n = 20` with `K = 2` and `K = 4`.
  Intended to run one file at a time with `2` and `4` ranks/threads in MPI+OpenMP mode.
- `test_07`:
  optimization class mapped from `1M/5k` to `1000/5` (`n = 1000`, `m = 5`).
- `test_08`:
  reduction-efficiency class with one ant (`m = 1`) and large matrix (`n = 4000`).
- `test_09`:
  two scenarios in one file (`n = 16` and `n = 17`) for parity/extreme checks.

File format
-----------

Line 1:
  `S`
  where `S` is the number of scenarios in the file.

Next `S` lines:
  `n K m T solver_seed instance_seed layout_id`

Fields:
- `n`: number of customers (depot is node `0`)
- `K`: number of vehicles/routes
- `m`: ants per iteration
- `T`: iterations
- `solver_seed`: deterministic seed for ACO
- `instance_seed`: deterministic seed for synthetic coordinates
- `layout_id`: instance shape selector used by the test runner
  - `0`: uniform random
  - `1`: clustered customers
  - `2`: border-heavy split (left/right stripes)
  - `3`: parity/line pattern (useful for `16` vs `17` checks)

Runner
------

- Final single-file runner:
  `tests/tests.c`
