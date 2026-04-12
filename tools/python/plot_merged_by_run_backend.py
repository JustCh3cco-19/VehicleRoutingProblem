#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("merged_by_run_backend")
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
    p = ROOT / name
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"].copy()
    for c in ["n", "K", "m", "run_id", "elapsed_s", "best_cost", "mpi_ranks", "omp_threads"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160)
    plt.close()


def strong_openmp(df: pd.DataFrame) -> None:
    if df.empty:
        return
    g = df.groupby(["n", "omp_threads"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(9, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("omp_threads")
        plt.plot(sub["omp_threads"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    plt.xlabel("OpenMP threads")
    plt.ylabel("Elapsed time [s]")
    plt.title("OpenMP Strong Scaling")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("strong_openmp_elapsed.png")

    # Speedup vs thread=1 per n
    rows = []
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("omp_threads")
        if 1 not in set(sub["omp_threads"].dropna().astype(int)):
            continue
        t1 = float(sub[sub["omp_threads"] == 1]["elapsed_s"].iloc[0])
        tmp = sub.copy()
        tmp["speedup"] = t1 / tmp["elapsed_s"]
        rows.append(tmp)
    if rows:
        sp = pd.concat(rows, ignore_index=True)
        plt.figure(figsize=(9, 5))
        for n, sub in sp.groupby("n"):
            sub = sub.sort_values("omp_threads")
            plt.plot(sub["omp_threads"], sub["speedup"], marker="o", label=f"n={int(n)}")
        plt.plot(sorted(sp["omp_threads"].unique()), sorted(sp["omp_threads"].unique()), "k--", alpha=0.4, label="ideal")
        plt.xlabel("OpenMP threads")
        plt.ylabel("Speedup")
        plt.title("OpenMP Strong Scaling Speedup")
        plt.grid(alpha=0.3)
        plt.legend()
        savefig("strong_openmp_speedup.png")


def strong_mpi(df: pd.DataFrame) -> None:
    if df.empty:
        return
    g = df.groupby(["n", "mpi_ranks"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(9, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("mpi_ranks")
        plt.plot(sub["mpi_ranks"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    plt.xlabel("MPI ranks")
    plt.ylabel("Elapsed time [s]")
    plt.title("MPI Strong Scaling")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("strong_mpi_elapsed.png")


def strong_hybrid(df: pd.DataFrame) -> None:
    if df.empty:
        return
    d = df.copy()
    d["cfg"] = d["mpi_ranks"].astype("Int64").astype(str) + "x" + d["omp_threads"].astype("Int64").astype(str)
    g = d.groupby(["n", "cfg"], as_index=False)["elapsed_s"].mean()
    order = ["1x32", "2x16", "4x8", "1x1", "2x2", "4x4"]
    present = [c for c in order if c in set(g["cfg"])]
    if not present:
        present = sorted(g["cfg"].unique())
    plt.figure(figsize=(10, 5))
    for n, sub in g.groupby("n"):
        sub = sub.set_index("cfg").reindex(present).reset_index()
        plt.plot(sub["cfg"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    plt.xlabel("Hybrid config (ranks x threads)")
    plt.ylabel("Elapsed time [s]")
    plt.title("Hybrid Strong Scaling")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("strong_hybrid_elapsed.png")


def weak_generic(df: pd.DataFrame, x_col: str, title: str, outname: str) -> None:
    if df.empty or x_col not in df.columns:
        return
    g = df.groupby([x_col, "n"], as_index=False)["elapsed_s"].mean().sort_values(x_col)
    plt.figure(figsize=(9, 5))
    plt.plot(g[x_col], g["elapsed_s"], marker="o")
    for _, r in g.iterrows():
        plt.annotate(f"n={int(r['n'])}", (r[x_col], r["elapsed_s"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Elapsed time [s]")
    plt.title(title)
    plt.grid(alpha=0.3)
    savefig(outname)


def seq_vs_cuda(seq_df: pd.DataFrame, cuda_df: pd.DataFrame) -> None:
    if seq_df.empty and cuda_df.empty:
        return
    plt.figure(figsize=(9, 5))
    if not seq_df.empty:
        s = seq_df.groupby("n", as_index=False)["elapsed_s"].mean().sort_values("n")
        plt.plot(s["n"], s["elapsed_s"], marker="o", label="SEQ")
    if not cuda_df.empty:
        c = cuda_df.groupby("n", as_index=False)["elapsed_s"].mean().sort_values("n")
        plt.plot(c["n"], c["elapsed_s"], marker="o", label="CUDA")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Problem size n (log2)")
    plt.ylabel("Elapsed time [s] (log)")
    plt.title("SEQ vs CUDA Performance")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    savefig("seq_vs_cuda_elapsed.png")


def quality_plot(seq_q: pd.DataFrame, mpi_q: pd.DataFrame, cuda_q: pd.DataFrame) -> None:
    frames = []
    for name, df in [("SEQ", seq_q), ("MPI", mpi_q), ("CUDA", cuda_q)]:
        if not df.empty:
            tmp = df[["n", "best_cost", "elapsed_s"]].copy()
            tmp["backend"] = name
            frames.append(tmp)
    if not frames:
        return
    q = pd.concat(frames, ignore_index=True)

    # Best cost by n/backend
    g = q.groupby(["backend", "n"], as_index=False)["best_cost"].mean()
    plt.figure(figsize=(10, 5))
    for backend, sub in g.groupby("backend"):
        sub = sub.sort_values("n")
        plt.plot(sub["n"], sub["best_cost"], marker="o", label=backend)
    plt.xscale("log", base=2)
    plt.xlabel("Problem size n (log2)")
    plt.ylabel("Average best cost")
    plt.title("Quality Comparison (Average Best Cost)")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("quality_best_cost.png")

    # Elapsed by n/backend
    t = q.groupby(["backend", "n"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(10, 5))
    for backend, sub in t.groupby("backend"):
        sub = sub.sort_values("n")
        plt.plot(sub["n"], sub["elapsed_s"], marker="o", label=backend)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("Problem size n (log2)")
    plt.ylabel("Average elapsed [s] (log)")
    plt.title("Quality Campaign Runtime")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    savefig("quality_elapsed.png")


def write_summary(seq_p, cuda_p, omp_s, mpi_s, hyb_s, omp_w, mpi_w, hyb_w, seq_q, mpi_q, cuda_q):
    lines = []
    lines.append("# Plots generated from merged_by_run_backend")
    lines.append("")
    lines.append("Generated files:")
    for p in sorted(OUT.glob("*.png")):
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("Input rows (status=ok):")
    datasets = {
        "seq_performance": seq_p,
        "cuda_performance": cuda_p,
        "openmp_strong": omp_s,
        "mpi_strong": mpi_s,
        "hybrid_strong": hyb_s,
        "openmp_weak": omp_w,
        "mpi_weak": mpi_w,
        "hybrid_weak": hyb_w,
        "seq_quality": seq_q,
        "mpi_quality": mpi_q,
        "cuda_quality": cuda_q,
    }
    for k, df in datasets.items():
        lines.append(f"- {k}: {len(df)}")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main():
    seq_p = load_csv("seq_performance.csv")
    cuda_p = load_csv("cuda_performance.csv")
    omp_s = load_csv("openmp_strong.csv")
    mpi_s = load_csv("mpi_strong.csv")
    hyb_s = load_csv("hybrid_strong.csv")
    omp_w = load_csv("openmp_weak.csv")
    mpi_w = load_csv("mpi_weak.csv")
    hyb_w = load_csv("hybrid_weak.csv")
    seq_q = load_csv("seq_quality.csv")
    mpi_q = load_csv("mpi_quality.csv")
    cuda_q = load_csv("cuda_quality.csv")

    strong_openmp(omp_s)
    strong_mpi(mpi_s)
    strong_hybrid(hyb_s)

    weak_generic(omp_w, "omp_threads", "OpenMP Weak Scaling", "weak_openmp_elapsed.png")
    weak_generic(mpi_w, "mpi_ranks", "MPI Weak Scaling", "weak_mpi_elapsed.png")

    if not hyb_w.empty:
        h = hyb_w.copy()
        h["workers"] = h["mpi_ranks"] * h["omp_threads"]
        weak_generic(h, "workers", "Hybrid Weak Scaling", "weak_hybrid_elapsed.png")

    seq_vs_cuda(seq_p, cuda_p)
    quality_plot(seq_q, mpi_q, cuda_q)
    write_summary(seq_p, cuda_p, omp_s, mpi_s, hyb_s, omp_w, mpi_w, hyb_w, seq_q, mpi_q, cuda_q)


if __name__ == "__main__":
    main()
