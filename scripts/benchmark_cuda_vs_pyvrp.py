#!/usr/bin/env python3
import os
import subprocess
import time
import csv
import numpy as np
import vrplib
from pyvrp import Model, read, stop

def generate_instance(n, k, path, seed=42):
    np.random.seed(seed)
    grid_size = 1000
    depot_coord = np.random.randint(0, grid_size, size=2)
    client_coords = np.random.randint(0, grid_size, size=(n, 2))
    all_coords = np.vstack([depot_coord, client_coords])
    demands = np.ones(n, dtype=int)
    all_demands = np.concatenate(([0], demands))
    capacity = n - k + 3
    
    instance_data = {
        "NAME": f"prob_n{n}_k{k}",
        "TYPE": "CVRP",
        "DIMENSION": n + 1,
        "VEHICLES": k,
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        "CAPACITY": int(capacity),
        "NODE_COORD_SECTION": all_coords,
        "DEMAND_SECTION": all_demands,
        "DEPOT_SECTION": [1, -1]
    }
    vrplib.write_instance(path, instance_data)

def run_cuda(n, k, instance_path, m=1024, max_t=20000, conv=1000, eps=0.01):
    # Set seed to 42 for reproducibility
    cmd = ["./aco_vrp_cuda_vrp", instance_path, str(k), str(m), str(max_t), "42", str(conv), str(eps)]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    
    if result.returncode != 0:
        print(f"CUDA Error: {result.stderr}")
        return None, end - start, 0
    
    # Parse output
    lines = result.stdout.strip().split('\n')
    cost = 0.0
    iters = 0
    for line in lines:
        if line.startswith("Cost:"):
            cost = float(line.split(":")[1].strip())
        elif "at iteration" in line:
            # Example: "Converged at iteration 1234 (relative improvement < 1.000000e-04 for 500 iterations)"
            try:
                iters = int(line.split("iteration")[1].split("(")[0].strip())
            except:
                iters = 0
        elif "Reached maximum iterations" in line:
            try:
                iters = int(line.split("iterations")[1].strip())
            except:
                iters = 0
    
    return cost, end - start, iters

def run_pyvrp(instance_path, max_runtime=10.0):
    instance = read(instance_path, round_func="round")
    model = Model.from_data(instance)
    start = time.time()
    # PyVRP often reaches optimality very fast for small n
    res = model.solve(stop.MaxRuntime(max_runtime))
    end = time.time()
    
    if res.best.is_feasible():
        return res.best.distance(), end - start
    return None, end - start

def main():
    n_values = [10, 100, 500, 1000]
    
    def get_k(n):
        return max(1, n // 10)

    results = []
    fieldnames = ['n', 'k', 'cuda_cost', 'cuda_time', 'cuda_iters', 'pyvrp_cost', 'pyvrp_time', 'gap']
    
    with open('benchmark_results_v4.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for n in n_values:
            k = get_k(n)
            print(f"\n--- Testing n={n}, k={k} ---")
            
            instance_path = f"tmp_bench_n{n}.vrp"
            generate_instance(n, k, instance_path)
            
            # Using more ants to help convergence
            m = 1024
            max_t = max(10000, 20000 if n <= 100 else 10000)
            conv = 1000 # Very high threshold to let epsilon drive
            eps = 0.01 # 1% improvement threshold
            
            cuda_cost, cuda_time, cuda_iters = run_cuda(n, k, instance_path, m=m, max_t=max_t, conv=conv, eps=eps)
            
            # PyVRP runtime
            pyvrp_runtime = 5.0 if n < 500 else 15.0
            pyvrp_cost, pyvrp_time = run_pyvrp(instance_path, max_runtime=pyvrp_runtime)
            
            gap = ""
            if cuda_cost and pyvrp_cost:
                gap = (cuda_cost - pyvrp_cost) / pyvrp_cost * 100
                print(f"  CUDA Iterations: {cuda_iters}")
                print(f"  CUDA: {cuda_cost:.2f} ({cuda_time:.2f}s)")
                print(f"  PyVRP: {pyvrp_cost:.2f} ({pyvrp_time:.2f}s)")
                print(f"  Gap: {gap:.2f}%")
            
            writer.writerow({
                'n': n,
                'k': k,
                'cuda_cost': cuda_cost,
                'cuda_time': cuda_time,
                'cuda_iters': cuda_iters,
                'pyvrp_cost': pyvrp_cost,
                'pyvrp_time': pyvrp_time,
                'gap': gap
            })
            csvfile.flush()
            
            if os.path.exists(instance_path):
                os.remove(instance_path)

if __name__ == "__main__":
    main()
