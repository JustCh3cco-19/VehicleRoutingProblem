#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("merged_by_run_backend")
OUT = ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
MAX_NODES = 4
MAX_CPUS_PER_TASK = 32
MAX_TOTAL_CORES = 128
CORE_TICKS = [1, 2, 4, 8, 16, 32, 64, 128]
COMBINED_CONFIGS = {
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 16),
    (1, 32),
    (2, 32),
    (4, 32),
}


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
    if {"mpi_ranks", "omp_threads"}.issubset(df.columns):
        df = df[
            df["mpi_ranks"].between(1, MAX_NODES)
            & df["omp_threads"].between(1, MAX_CPUS_PER_TASK)
            & ((df["mpi_ranks"] * df["omp_threads"]) <= MAX_TOTAL_CORES)
        ].copy()
    return df


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=160)
    plt.close()


def configure_core_axis(
    label: str = "Core CPU totali", ticks: list[int] = CORE_TICKS
) -> None:
    ax = plt.gca()
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8 if ticks[0] == 1 else ticks[0] * 0.8, ticks[-1] * 1.25)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks])
    ax.set_xlabel(label)


def strong_openmp(df: pd.DataFrame) -> None:
    if df.empty:
        return
    g = df.groupby(["n", "omp_threads"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(9, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("omp_threads")
        plt.plot(sub["omp_threads"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    configure_core_axis(
        "Core CPU totali (1 rank MPI, thread OpenMP)", CORE_TICKS[:6]
    )
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
        configure_core_axis(
            "Core CPU totali (1 rank MPI, thread OpenMP)", CORE_TICKS[:6]
        )
        plt.ylabel("Speedup")
        plt.title("OpenMP Strong Scaling Speedup")
        plt.grid(alpha=0.3)
        plt.legend()
        savefig("strong_openmp_speedup.png")


def strong_mpi(df: pd.DataFrame) -> None:
    if df.empty:
        return
    d = df.copy()
    d["total_cores"] = d["mpi_ranks"] * d["omp_threads"]
    g = d.groupby(["n", "total_cores"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(9, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("total_cores")
        plt.plot(sub["total_cores"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    configure_core_axis(
        "Core CPU totali (rank/nodi MPI × 32 thread)", CORE_TICKS[-3:]
    )
    plt.ylabel("Elapsed time [s]")
    plt.title("MPI Strong Scaling")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("strong_mpi_elapsed.png")


def strong_hybrid(df: pd.DataFrame) -> None:
    if df.empty:
        return
    d = df.copy()
    d = d[
        d.apply(
            lambda row: (int(row["mpi_ranks"]), int(row["omp_threads"]))
            in COMBINED_CONFIGS,
            axis=1,
        )
    ].copy()
    if d.empty:
        return
    d["total_cores"] = d["mpi_ranks"] * d["omp_threads"]
    g = d.groupby(["n", "total_cores"], as_index=False)["elapsed_s"].mean()
    plt.figure(figsize=(10, 5))
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("total_cores")
        plt.plot(sub["total_cores"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    configure_core_axis()
    plt.ylabel("Elapsed time [s]")
    plt.title("Strong Scaling Combinato OpenMP + MPI")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("strong_hybrid_elapsed.png")

    rows = []
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("total_cores").copy()
        baseline = sub[sub["total_cores"] == 1]
        if baseline.empty:
            continue
        sub["speedup"] = float(baseline["elapsed_s"].iloc[0]) / sub["elapsed_s"]
        rows.append(sub)
    if rows:
        speedup = pd.concat(rows, ignore_index=True)
        plt.figure(figsize=(10, 5))
        for n, sub in speedup.groupby("n"):
            sub = sub.sort_values("total_cores")
            plt.plot(sub["total_cores"], sub["speedup"], marker="o", label=f"n={int(n)}")
        plt.plot(CORE_TICKS, CORE_TICKS, "k--", alpha=0.4, label="ideale")
        configure_core_axis()
        plt.ylabel("Speedup")
        plt.title("Strong Scaling Combinato: Speedup")
        plt.grid(alpha=0.3)
        plt.legend()
        savefig("strong_hybrid_speedup.png")


def weak_generic(df: pd.DataFrame, x_col: str, title: str, outname: str) -> None:
    if df.empty or x_col not in df.columns:
        return
    g = df.groupby([x_col, "n"], as_index=False)["elapsed_s"].mean().sort_values(x_col)
    plt.figure(figsize=(9, 5))
    plt.plot(g[x_col], g["elapsed_s"], marker="o")
    for _, r in g.iterrows():
        plt.annotate(f"n={int(r['n'])}", (r[x_col], r["elapsed_s"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    if x_col in {"omp_threads", "total_cores"}:
        configure_core_axis(ticks=CORE_TICKS[:6] if x_col == "omp_threads" else CORE_TICKS)
    else:
        plt.xlabel(x_col.replace("_", " ").title())
    plt.ylabel("Elapsed time [s]")
    plt.title(title)
    plt.grid(alpha=0.3)
    savefig(outname)


def weak_combined(openmp_df: pd.DataFrame, mpi_df: pd.DataFrame) -> None:
    frames = []
    for source, df in (("openmp", openmp_df), ("mpi", mpi_df)):
        if df.empty or "m" not in df.columns:
            continue
        d = df.copy()
        d["total_cores"] = d["mpi_ranks"] * d["omp_threads"]
        d = d[d["m"] == 32 * d["total_cores"]]
        d = d[d["total_cores"] <= 32] if source == "openmp" else d[d["total_cores"] > 32]
        frames.append(d)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    g = data.groupby(["n", "total_cores"], as_index=False)["elapsed_s"].mean()

    plt.figure(figsize=(10, 5))
    efficiency_rows = []
    for n, sub in g.groupby("n"):
        sub = sub.sort_values("total_cores").copy()
        if 1 not in set(sub["total_cores"].astype(int)):
            continue
        baseline = float(sub[sub["total_cores"] == 1]["elapsed_s"].iloc[0])
        sub["weak_efficiency"] = baseline / sub["elapsed_s"]
        efficiency_rows.append(sub)
        plt.plot(sub["total_cores"], sub["elapsed_s"], marker="o", label=f"n={int(n)}")
    if not efficiency_rows:
        plt.close()
        return
    configure_core_axis()
    plt.ylabel("Elapsed time [s]")
    plt.title("Weak Scaling Combinato (32 formiche/core)")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("weak_combined_elapsed.png")

    efficiency = pd.concat(efficiency_rows, ignore_index=True)
    plt.figure(figsize=(10, 5))
    for n, sub in efficiency.groupby("n"):
        sub = sub.sort_values("total_cores")
        plt.plot(
            sub["total_cores"],
            sub["weak_efficiency"],
            marker="o",
            label=f"n={int(n)}",
        )
    plt.axhline(1.0, color="k", linestyle="--", alpha=0.4, label="ideale")
    configure_core_axis()
    plt.ylabel("Efficienza weak T(1)/T(N)")
    plt.title("Weak Scaling Combinato: Efficienza")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig("weak_combined_efficiency.png")


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
    if not mpi_w.empty:
        m = mpi_w.copy()
        m["total_cores"] = m["mpi_ranks"] * m["omp_threads"]
        weak_generic(m, "total_cores", "MPI Weak Scaling", "weak_mpi_elapsed.png")

    if not hyb_w.empty:
        h = hyb_w.copy()
        h["total_cores"] = h["mpi_ranks"] * h["omp_threads"]
        weak_generic(h, "total_cores", "Hybrid Weak Scaling", "weak_hybrid_elapsed.png")

    weak_combined(omp_w, mpi_w)

    seq_vs_cuda(seq_p, cuda_p)
    quality_plot(seq_q, mpi_q, cuda_q)
    write_summary(seq_p, cuda_p, omp_s, mpi_s, hyb_s, omp_w, mpi_w, hyb_w, seq_q, mpi_q, cuda_q)


if __name__ == "__main__":
    main()
