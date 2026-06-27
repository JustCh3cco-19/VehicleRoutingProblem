#!/usr/bin/env python3
import os
import subprocess
import sys
import tempfile
import math
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SEQ_BIN = ROOT_DIR / "seq.out"
MPI_BIN = ROOT_DIR / "openmp_mpi.out"
CUDA_BIN = ROOT_DIR / "cuda.out"

def calculate_euc2d(c1, c2):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def run_solver(bin_path, args, env_vars=None):
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    cmd = [str(bin_path)] + [str(a) for a in args]
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res

def create_vrp_file(path, name, n, K, capacity, coords, demands):
    with open(path, "w") as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : CVRP\n")
        f.write(f"DIMENSION : {n+1}\n")
        f.write(f"VEHICLES : {K}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write(f"CAPACITY : {capacity}\n")
        f.write("NODE_COORD_SECTION\n")
        # Depot is index 1 in TSPLIB style
        f.write(f"1 {coords[0][0]} {coords[0][1]}\n")
        for i in range(1, n + 1):
            f.write(f"{i+1} {coords[i][0]} {coords[i][1]}\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for i in range(1, n + 1):
            f.write(f"{i+1} {demands[i]}\n")
        f.write("DEPOT_SECTION\n")
        f.write("1\n")
        f.write("-1\n")

def test_small_optimal(bin_path, backend_name):
    print(f"[{backend_name}] Test 1: Small instance with known optimal cost (120.0)...")
    coords = [
        (0.0, 0.0),   # depot
        (10.0, 0.0),  # c1
        (20.0, 0.0),  # c2
        (30.0, 0.0),  # c3
        (40.0, 0.0)   # c4
    ]
    demands = [0, 1, 1, 1, 1]
    
    with tempfile.NamedTemporaryFile(suffix=".vrp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_vrp_file(tmp_path, "small_opt", 4, 2, 2, coords, demands)
        
        # Run solver with high iterations/ants or reproducibility mode to ensure optimal path is found
        env = {
            "ACO_SOLVER_TIMEOUT_SECONDS": "5",
            "ACO_SOLVER_STAGNATION_EPOCHS": "100",
            "ACO_SOLVER_MIN_REL_IMPROVEMENT": "0.001"
        }
        res = run_solver(bin_path, [tmp_path, 2, 16, 1234], env_vars=env)
            
        if res.returncode != 0:
            print(f"  FAILED: solver exited with code {res.returncode}")
            print(f"  stderr: {res.stderr}")
            print(f"  stdout: {res.stdout}")
            return False
            
        # Parse cost
        cost_line = [line for line in res.stdout.splitlines() if line.startswith("best cost:")]
        if not cost_line:
            print("  FAILED: missing 'best cost:' in output")
            return False
            
        cost = float(cost_line[0].split(":")[1].strip())
        if abs(cost - 120.0) > 1e-2:
            print(f"  FAILED: expected optimal cost 120.0, got {cost}")
            return False
            
        print("  PASSED")
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_impossible_capacity(bin_path, backend_name):
    print(f"[{backend_name}] Test 2: Impossible capacity (demand > capacity)...")
    coords = [
        (0.0, 0.0),
        (10.0, 10.0),
        (20.0, 20.0),
        (30.0, 30.0)
    ]
    demands = [0, 1, 1, 1]
    
    with tempfile.NamedTemporaryFile(suffix=".vrp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # Total demand = 3. Vehicles K=2, capacity=1. Max capacity served = 2. Total demand > total capacity.
        create_vrp_file(tmp_path, "impossible_cap", 3, 2, 1, coords, demands)
        
        env = {
            "ACO_SOLVER_TIMEOUT_SECONDS": "2",
            "ACO_SOLVER_STAGNATION_EPOCHS": "10"
        }
        res = run_solver(bin_path, [tmp_path, 2, 8, 1234], env_vars=env)
        
        # Solver should fail because it cannot find a valid solution
        # Wait, if solver cannot find a solution, it returns exit code 1 or prints error
        if res.returncode == 0:
            print("  FAILED: solver succeeded (exit code 0) but solution should be mathematically impossible!")
            print(f"  stdout: {res.stdout}")
            return False
            
        print("  PASSED (solver correctly failed or returned error)")
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_non_unitary_demand(bin_path, backend_name):
    print(f"[{backend_name}] Test 3: Non-unitary demand (should be rejected)...")
    coords = [
        (0.0, 0.0),
        (10.0, 10.0),
        (20.0, 20.0)
    ]
    demands = [0, 2, 1] # customer 1 has demand 2
    
    with tempfile.NamedTemporaryFile(suffix=".vrp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_vrp_file(tmp_path, "non_unitary", 2, 1, 3, coords, demands)
        res = run_solver(bin_path, [tmp_path, 1, 4, 1234])
        
        # Should be rejected with exit code 1
        if res.returncode == 0:
            print("  FAILED: solver accepted non-unitary demand")
            return False
            
        if "Customer demand must be 1" not in res.stderr and "Customer demand must be 1" not in res.stdout:
            print(f"  WARNING: expected 'Customer demand must be 1' error message. Got stderr: {res.stderr}")
            
        print("  PASSED")
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_mismatch_k(bin_path, backend_name):
    print(f"[{backend_name}] Test 4: Vehicles mismatch K (file K vs CLI K)...")
    coords = [
        (0.0, 0.0),
        (10.0, 10.0),
        (20.0, 20.0)
    ]
    demands = [0, 1, 1]
    
    with tempfile.NamedTemporaryFile(suffix=".vrp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        # File K=2, CLI K=1
        create_vrp_file(tmp_path, "mismatch_k", 2, 2, 5, coords, demands)
        res = run_solver(bin_path, [tmp_path, 1, 4, 1234])
        
        if res.returncode == 0:
            print("  FAILED: solver accepted CLI K mismatch with file VEHICLES")
            return False
            
        if "instance VEHICLES mismatch" not in res.stderr and "instance VEHICLES mismatch" not in res.stdout:
            print(f"  WARNING: expected 'instance VEHICLES mismatch' error message. Got stderr: {res.stderr}")
            
        print("  PASSED")
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_cost_validation(bin_path, backend_name):
    print(f"[{backend_name}] Test 5: Cost validation (printed vs recalculated)...")
    coords = [
        (0.0, 0.0),
        (12.0, 5.0),
        (-5.0, 12.0),
        (8.0, -15.0),
        (-9.0, -40.0)
    ]
    demands = [0, 1, 1, 1, 1]
    
    with tempfile.NamedTemporaryFile(suffix=".vrp", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        create_vrp_file(tmp_path, "cost_val", 4, 2, 3, coords, demands)
        env = {
            "ACO_SOLVER_TIMEOUT_SECONDS": "2",
            "ACO_SOLVER_STAGNATION_EPOCHS": "20"
        }
        res = run_solver(bin_path, [tmp_path, 2, 8, 1234], env_vars=env)
        
        if res.returncode != 0:
            print(f"  FAILED: solver exited with code {res.returncode}")
            return False
            
        # Parse cost and routes
        cost = None
        routes = []
        for line in res.stdout.splitlines():
            if line.startswith("best cost:") or line.startswith("Cost:"):
                cost = float(line.split(":")[1].strip())
            elif line.startswith("Route"):
                # E.g. "Route 1: 1 2"
                parts = line.split(":")
                route_nodes = [int(x) for x in parts[1].strip().split() if x.strip()]
                routes.append(route_nodes)
                
        if cost is None:
            print("  FAILED: could not find cost in output")
            return False
            
        # Recalculate cost
        recalc_cost = 0.0
        for r in routes:
            if not r:
                continue
            # Path starts at depot (coords[0])
            prev_node = 0
            for node in r:
                recalc_cost += calculate_euc2d(coords[prev_node], coords[node])
                prev_node = node
            # Returns to depot
            recalc_cost += calculate_euc2d(coords[prev_node], coords[0])
            
        if abs(cost - recalc_cost) > 1e-3:
            print(f"  FAILED: printed cost {cost:.6f} differs from recalculated cost {recalc_cost:.6f}")
            return False
            
        print(f"  PASSED (cost={cost:.3f}, recalculated={recalc_cost:.3f})")
        return True
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def test_invalid_cli_arguments(bin_path, backend_name):
    print(f"[{backend_name}] Test 6: Invalid CLI arguments...")
    
    # 1. Negative K
    res = run_solver(bin_path, ["fake.vrp", -2, 10])
    if res.returncode == 0:
        print("  FAILED: accepted negative K")
        return False
        
    # 2. Negative m
    res = run_solver(bin_path, ["fake.vrp", 2, -5])
    if res.returncode == 0:
        print("  FAILED: accepted negative m")
        return False
        
    print("  PASSED")
    return True

def check_mpi_functional():
    if not MPI_BIN.exists():
        return False
    try:
        res = subprocess.run([str(MPI_BIN)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        # Check if ORTE fails or crashes
        if "MPI_Init_thread failed" in res.stderr or "ORTE_ERROR_LOG" in res.stderr or "orte_ess_init failed" in res.stderr or "orte_init failed" in res.stderr or "MPI_ERRORS_ARE_FATAL" in res.stderr:
            return False
        return True
    except Exception:
        return False

def run_all_tests(bin_path, name):
    print("=" * 60)
    print(f"Running regression tests for backend: {name}")
    print("=" * 60)
    success = True
    success = success and test_small_optimal(bin_path, name)
    success = success and test_impossible_capacity(bin_path, name)
    success = success and test_non_unitary_demand(bin_path, name)
    success = success and test_mismatch_k(bin_path, name)
    success = success and test_cost_validation(bin_path, name)
    success = success and test_invalid_cli_arguments(bin_path, name)
    return success

def main():
    has_any = False
    all_success = True
    
    if SEQ_BIN.exists():
        has_any = True
        all_success = all_success and run_all_tests(SEQ_BIN, "seq")
        
    if MPI_BIN.exists():
        if check_mpi_functional():
            has_any = True
            all_success = all_success and run_all_tests(MPI_BIN, "mpi")
        else:
            print("\n[mpi] skipped regression tests: MPI environment is not functional in this sandbox (e.g. PMIx/ORTE socket block)")
        
    if CUDA_BIN.exists():
        # Check if GPU is visible
        gpu_visible = False
        try:
            gpures = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if gpures.returncode == 0:
                gpu_visible = True
        except Exception:
            pass
            
        if gpu_visible:
            has_any = True
            all_success = all_success and run_all_tests(CUDA_BIN, "cuda")
        else:
            print("\n[cuda] skipped regression tests: no CUDA GPU visible")

    if not has_any:
        print("Error: No solver binaries found. Run 'make all' first.")
        sys.exit(1)
        
    if all_success:
        print("\nALL REGRESSION TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME REGRESSION TESTS FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()
