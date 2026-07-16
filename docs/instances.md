# Instances

The maintained dataset is `instances/generated_benchmark/`. Each `.vrp` file
uses a TSPLIB-like CVRP representation with EUC_2D coordinates, unit customer
demand, a depot, vehicle capacity, and a vehicle count.

## Generate a dataset

```bash
make generate_problems
```

The target invokes `tools/python/generate_vrp_problem.py` and writes:

- one `n<N>_k<K>_s<seed>.vrp` file per requested size;
- `manifest.csv` for PyVRP and sequential runs;
- `manifest_openmp_mpi.csv` for the parallel CPU runner;
- `manifest_cuda.csv` for the CUDA runner.

Generation is deterministic for a fixed configuration. Common overrides are:

```bash
make generate_problems \
  GEN_INST_DIR=instances/debug \
  GEN_CLIENTS=500,1000,2000 \
  GEN_SEED_BASE=19000 \
  GEN_CUDA_M=256
```

`GEN_TARGET_CUSTOMERS_PER_VEHICLE`, `GEN_MIN_VEHICLES`, and
`GEN_MAX_VEHICLES` determine `K`. Capacity is computed from `n`, `K`, and
`GEN_CAPACITY_SLACK_PERCENT`. `make help` lists the exact defaults.

`make generate_problems_big` is the predefined non-CUDA profile for 4,000 to
32,000 customers. `make generate_clean GEN_INST_DIR=<dir>` deletes generated
VRP and manifest files in that directory.

## Use another manifest

```bash
make solve_seq SOLVE_MANIFEST=instances/debug/manifest.csv
make solve_mpi SOLVE_MANIFEST_MPI=instances/debug/manifest_openmp_mpi.csv
make solve_cuda SOLVE_MANIFEST_CUDA=instances/debug/manifest_cuda.csv
```

`SOLVE_CLIENTS` filters manifest rows by customer count and `SOLVE_LIMIT`
limits the number of selected rows. Current manifest columns are `profile`,
`name`, `instance_path`, `n`, `K`, `m`, `solver_seed`, `instance_seed`,
`layout_id`, and `capacity_formula`.

To generate one file directly, inspect the current CLI with:

```bash
python3 tools/python/generate_vrp_problem.py --help
```
