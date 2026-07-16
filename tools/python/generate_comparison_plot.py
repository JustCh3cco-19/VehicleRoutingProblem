import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Dati V1 (estratti dai log precedenti)
v1_data = {
    "N": [50, 100, 250, 500, 1000],
    "TimePerIter_ms": [3.27, 4.70, 24.59, 93.38, 366.23]
}

# Carica V2 e V3 dai CSV
v2_results = pd.read_csv("docs/cuda/performance/v2/results.csv")
v3_results = pd.read_csv("docs/cuda/performance/v3/results.csv")

plt.figure(figsize=(12, 7))

# Plot V1
plt.plot(v1_data["N"], v1_data["TimePerIter_ms"], marker='x', linestyle='--', label='V1: Baseline (Ant-per-Thread)', color='red', alpha=0.7)

# Plot V2
plt.plot(v2_results["N"].to_numpy(), v2_results["TimePerIter_ms"].to_numpy(), marker='s', linestyle='--', label='V2: Warp-Parallel', color='orange', alpha=0.7)

# Plot V3
plt.plot(v3_results["N"].to_numpy(), v3_results["TimePerIter_ms"].to_numpy(), marker='o', linestyle='-', label='V3: Warp-Scan + GPU-Deposit', color='green', linewidth=2)

plt.title("CUDA ACO Performance Comparison: Scaling Analysis", fontsize=14)
plt.xlabel("Problem Size (N)", fontsize=12)
plt.ylabel("Time per Iteration (ms)", fontsize=12)
plt.yscale('log')  # Usiamo scala logaritmica per vedere meglio la differenza
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(fontsize=11)

# Aggiungiamo etichette per N=1000
plt.annotate(f"{v1_data['TimePerIter_ms'][-1]:.1f}ms", (1000, v1_data["TimePerIter_ms"][-1]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
plt.annotate(f"{v2_results['TimePerIter_ms'].iloc[-1]:.1f}ms", (1000, v2_results["TimePerIter_ms"].iloc[-1]), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
plt.annotate(f"{v3_results['TimePerIter_ms'].iloc[-1]:.1f}ms", (1000, v3_results["TimePerIter_ms"].iloc[-1]), textcoords="offset points", xytext=(0,10), ha='center', color='green', weight='bold')

plt.tight_layout()
output_path = "docs/cuda/performance/comparison_scaling_plot.png"
plt.savefig(output_path)
print(f"Comparison plot saved to {output_path}")
