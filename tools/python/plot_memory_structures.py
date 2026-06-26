#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ALIGN = 64


def align_up(v: int, a: int = ALIGN) -> int:
    r = v % a
    return v if r == 0 else v + (a - r)


def gib(x: int) -> float:
    return x / (1024.0 ** 3)


def matrix_double_bytes(n: int) -> int:
    side = n + 1
    row = align_up(side * 8)
    data = side * row
    ptrs = side * 8
    return data + ptrs


def cand_k(n: int) -> int:
    if n <= 8:
        return n
    if n <= 256:
        return 16
    if n <= 4096:
        return 24
    return 32


def stride(cols: int, elem: int) -> int:
    return align_up(cols * elem) // elem


def shared_bytes(n: int) -> int:
    rows = n + 1
    k = cand_k(n)
    s = stride(k, 4)
    # int candidate + float eta + float score
    return rows * s * 4 + rows * s * 4 + rows * s * 4


def solution_bytes(k: int, n: int) -> int:
    cap = n + 2
    return 24 + k * 16 + k * cap * 4


def seq_other_bytes(n: int, k: int) -> int:
    visited = ((n // 64) + 1) * 8
    route_loads = k * 4
    # ws.sol + iter_best + best_solution
    sols = 3 * solution_bytes(k, n)
    return shared_bytes(n) + visited + route_loads + sols


def omp_other_bytes(n: int, k: int, threads: int) -> int:
    visited = ((n // 64) + 1) * 8
    route_loads = k * 4
    # per thread: sol + thread_best + visited + route_loads
    per_thread = 2 * solution_bytes(k, n) + visited + route_loads
    # rank-level iter_best + best_solution + shared tables
    rank_level = 2 * solution_bytes(k, n) + shared_bytes(n)
    return threads * per_thread + rank_level


def read_manifest(path: Path):
    out = []
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            out.append((r["name"], int(r["n"]), int(r["K"])))
    out.sort(key=lambda x: x[1])
    return out


def main():
    ap = argparse.ArgumentParser(description="Plot memory growth by data structure (seq/openmp).")
    ap.add_argument("--manifest-seq", default="instances/generated_benchmark/manifest.csv")
    ap.add_argument("--manifest-openmp", default="instances/generated_benchmark/manifest_openmp_mpi.csv")
    ap.add_argument("--openmp-threads", type=int, default=32)
    ap.add_argument("--out-csv", default="results/analysis/memory_structures_seq_openmp.csv")
    ap.add_argument("--out-png", default="results/analysis/memory_structures_seq_openmp_stacked.png")
    args = ap.parse_args()

    seq = read_manifest(Path(args.manifest_seq))
    omp = read_manifest(Path(args.manifest_openmp))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, n, k in seq:
        c = matrix_double_bytes(n)
        tau = matrix_double_bytes(n)
        other = seq_other_bytes(n, k)
        rows.append(("seq", name, n, k, gib(c), gib(tau), gib(other), gib(c + tau + other)))
    for name, n, k in omp:
        c = matrix_double_bytes(n)
        tau = matrix_double_bytes(n)
        other = omp_other_bytes(n, k, args.openmp_threads)
        rows.append(("openmp", name, n, k, gib(c), gib(tau), gib(other), gib(c + tau + other)))

    with out_csv.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["backend", "name", "n", "K", "c_gib", "tau_gib", "other_gib", "total_gib"])
        wr.writerows(rows)

    seq_rows = [r for r in rows if r[0] == "seq"]
    omp_rows = [r for r in rows if r[0] == "openmp"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, title, data in [
        (axes[0], "Sequential", seq_rows),
        (axes[1], f"OpenMP ({args.openmp_threads} threads)", omp_rows),
    ]:
        x = list(range(len(data)))
        c = [r[4] for r in data]
        tau = [r[5] for r in data]
        other = [r[6] for r in data]
        nvals = [r[2] for r in data]

        ax.bar(x, c, label="Cost matrix (c)")
        ax.bar(x, tau, bottom=c, label="Pheromone matrix (tau)")
        ax.bar(
            x,
            other,
            bottom=[a + b for a, b in zip(c, tau)],
            label="Auxiliary structures (candidate lists, score buffers, workspaces)",
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in nvals], rotation=45, ha="right")
        ax.set_xlabel("n clienti")
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Memoria stimata (GiB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=True,
    )
    fig.suptitle("Crescita memoria per struttura dati", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")

    print(f"wrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
