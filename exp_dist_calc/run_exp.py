import subprocess
import os

sizes = [100, 500, 1000, 5000, 10000, 20000, 30000, 50000, 100000]

def run_bench():
    # Compile
    print("Compiling CPU bench...")
    subprocess.run(["gcc", "-O3", "-fopenmp", "exp_dist_calc/bench_cpu.c", "-o", "exp_dist_calc/bench_cpu", "-lm"], check=True)
    print("Compiling GPU bench...")
    subprocess.run(["nvcc", "-O3", "exp_dist_calc/bench_gpu.cu", "-o", "exp_dist_calc/bench_gpu"], check=True)

    gpu_csv = "exp_dist_calc/results/gpu_results.csv"
    cpu_csv = "exp_dist_calc/results/cpu_results.csv"

    with open(gpu_csv, "w") as f:
        f.write("N,Memory_ms,Compute_ms,Matrix_MB,Coords_MB\n")
        for n in sizes:
            print(f"Running GPU bench for N={n}...")
            res = subprocess.run(["./exp_dist_calc/bench_gpu", str(n)], capture_output=True, text=True)
            if res.returncode == 0:
                f.write(res.stdout)
            else:
                print(f"GPU bench failed for N={n}")

    with open(cpu_csv, "w") as f:
        f.write("N,Memory_ms,Compute_ms\n")
        for n in sizes:
            print(f"Running CPU bench for N={n}...")
            res = subprocess.run(["./exp_dist_calc/bench_cpu", str(n)], capture_output=True, text=True)
            if res.returncode == 0:
                f.write(res.stdout)
            else:
                print(f"CPU bench failed for N={n}")

if __name__ == "__main__":
    run_bench()
    print("Experiment completed. Results in exp_dist_calc/results/")
