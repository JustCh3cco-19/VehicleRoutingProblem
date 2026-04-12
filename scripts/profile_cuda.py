import os
import subprocess
import time
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione del benchmark
SIZES = [50, 100, 250, 500, 1000]
ANTS = 1024  # Numero di formiche fisso per vedere come scala con N
ITERATIONS = 100
K_VEHICLES = 10
BIN_PATH = "./aco_vrp_cuda_vrp"
REPORTS_DIR = Path("reports/cuda_profile")
INSTANCES_DIR = Path("instances/benchmark")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

def generate_instance(n):
    instance_path = INSTANCES_DIR / f"problem_{n}.vrp"
    # Semplice generazione di un file VRP in formato TSPLIB (EUC_2D)
    with open(instance_path, "w") as f:
        f.write(f"NAME : problem_{n}\n")
        f.write("TYPE : VRP\n")
        f.write(f"DIMENSION : {n+1}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        f.write(f"1 0.0 0.0\n") # Depot
        for i in range(2, n + 2):
            f.write(f"{i} {random.uniform(0, 100):.2f} {random.uniform(0, 100):.2f}\n")
        f.write("EOF\n")
    return instance_path

def run_benchmark():
    results = []
    
    # Assicurati che il binario sia compilato
    print("Building project...")
    subprocess.run(["make", "cuda-vrp"], check=True)
    
    for n in SIZES:
        print(f"\n--- Benchmarking N={n} ---")
        instance_path = generate_instance(n)
        
        # 1. Esecuzione standard per il tempo totale
        cmd = [BIN_PATH, str(instance_path), str(K_VEHICLES), str(ANTS), str(ITERATIONS)]
        start = time.perf_counter()
        res = subprocess.run(cmd, capture_output=True, text=True)
        end = time.perf_counter()
        
        total_time = end - start
        
        if res.returncode != 0:
            print(f"Error running N={n}: {res.stderr}")
            continue
            
        # 2. Profiling con nsys per metriche GPU (solo un'iterazione ridotta per velocità)
        profile_file = REPORTS_DIR / f"nsys_N{n}"
        nsys_cmd = [
            "nsys", "profile",
            "-o", str(profile_file),
            "--force-overwrite", "true",
            "--stats=true",
            BIN_PATH, str(instance_path), str(K_VEHICLES), str(ANTS), "10" # Pochi cicli per il profiling
        ]
        subprocess.run(nsys_cmd, capture_output=True)
        
        results.append({
            "N": n,
            "TotalTime_s": total_time,
            "TimePerIter_ms": (total_time / ITERATIONS) * 1000
        })
        print(f"Total Time: {total_time:.4f}s | Per Iter: {(total_time / ITERATIONS) * 1000:.2f}ms")

    df = pd.DataFrame(results)
    df.to_csv(REPORTS_DIR / "cuda_scaling_results.csv", index=False)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df["N"].to_numpy(), df["TimePerIter_ms"].to_numpy(), marker='o', linestyle='-', color='b')
    plt.title(f"CUDA Scaling: Time per Iteration (Ants={ANTS})")
    plt.xlabel("Problem Size (N)")
    plt.ylabel("Time per Iteration (ms)")
    plt.grid(True)
    plt.savefig(REPORTS_DIR / "scaling_plot.png")
    print(f"\nBenchmark complete. Results saved in {REPORTS_DIR}")


if __name__ == "__main__":
    run_benchmark()
