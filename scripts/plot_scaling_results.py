#!/usr/bin/env python3
"""
Generate performance plots from reports/scaling_summary.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MAX_TOTAL_CORES = 128
MAX_NODES = 4
MAX_CPUS_PER_TASK = 32
CORE_TICKS = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass
class Row:
    experiment: str
    backend: str
    ranks: int
    threads_per_rank: int
    scale: int
    n: int
    k: int
    m: int
    t: int
    samples: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    speedup: float
    efficiency: float


@dataclass
class RawRow:
    experiment: str
    backend: str
    warmup: int
    status: str
    elapsed_s: float
    stderr: str


def parse_float(text: str) -> float:
    if text == "":
        return float("nan")
    return float(text)


def load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    experiment=r["experiment"],
                    backend=r["backend"],
                    ranks=int(r["ranks"]),
                    threads_per_rank=int(r["threads_per_rank"]),
                    scale=int(r["scale"]),
                    n=int(r["n"]),
                    k=int(r["K"]),
                    m=int(r["m"]),
                    t=int(r["T"]),
                    samples=int(r["samples"]),
                    mean_s=float(r["mean_s"]),
                    std_s=float(r["std_s"]),
                    min_s=float(r["min_s"]),
                    max_s=float(r["max_s"]),
                    speedup=parse_float(r["speedup"]),
                    efficiency=parse_float(r["efficiency"]),
                )
            )
    return rows


def load_raw_rows(path: Path) -> list[RawRow]:
    rows: list[RawRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                RawRow(
                    experiment=r["experiment"],
                    backend=r["backend"],
                    warmup=int(r["warmup"]),
                    status=r["status"],
                    elapsed_s=float(r["elapsed_s"]),
                    stderr=r["stderr"],
                )
            )
    return rows


def pick(rows: list[Row], experiment: str) -> list[Row]:
    subset = [r for r in rows if r.experiment == experiment]
    subset.sort(key=lambda r: r.scale)
    return subset


def as_arrays(rows: list[Row]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([r.scale for r in rows], dtype=float)
    y = np.array([r.mean_s for r in rows], dtype=float)
    e = np.array([r.std_s for r in rows], dtype=float)
    return x, y, e


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def configure_core_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8, 160)
    ax.set_xticks(CORE_TICKS)
    ax.set_xticklabels([str(x) for x in CORE_TICKS])
    ax.set_xlabel("Core CPU totali")


def plot_strong_runtime(out_dir: Path, omp: list[Row], mpi_s: list[Row], seq: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xo, yo, eo = as_arrays(omp)
    xm, ym, em = as_arrays(mpi_s)

    ax.errorbar(xo, yo, yerr=eo, marker="o", capsize=3, linewidth=2, label="OpenMP strong")
    ax.errorbar(xm, ym, yerr=em, marker="s", capsize=3, linewidth=2, label="MPI+OpenMP strong")

    if seq:
        seq_t = seq[0].mean_s
        ax.axhline(seq_t, linestyle="--", color="gray", label=f"Sequential baseline ({seq_t:.3f}s)")

    ax.set_title("Strong Scaling Runtime")
    configure_core_axis(ax)
    ax.set_ylabel("Mean Time (s)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "strong_runtime.png")


def plot_strong_speedup(out_dir: Path, omp: list[Row], mpi_s: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    xo = np.array([r.scale for r in omp], dtype=float)
    so = np.array([r.speedup for r in omp], dtype=float)
    xm = np.array([r.scale for r in mpi_s], dtype=float)
    sm = np.array([r.speedup for r in mpi_s], dtype=float)

    # Ideal lines use each experiment's own baseline scale.
    base_omp = float(min(r.scale for r in omp))
    base_mpi = float(min(r.scale for r in mpi_s))
    ax.plot(xo, xo / base_omp, linestyle="--", color="#1f77b4", alpha=0.6, label="Ideal OpenMP")
    ax.plot(
        xm,
        xm / base_mpi,
        linestyle="--",
        color="#ff7f0e",
        alpha=0.6,
        label="Ideal MPI+OpenMP (relativo a 32 core)",
    )

    ax.plot(xo, so, marker="o", linewidth=2, label="Measured OpenMP")
    ax.plot(
        xm,
        sm,
        marker="s",
        linewidth=2,
        label="Measured MPI+OpenMP (relativo a 32 core)",
    )

    ax.set_title("Strong Scaling Speedup")
    configure_core_axis(ax)
    ax.set_ylabel("Speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "strong_speedup.png")


def plot_strong_efficiency(out_dir: Path, omp: list[Row], mpi_s: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    xo = np.array([r.scale for r in omp], dtype=float)
    eo = np.array([r.efficiency for r in omp], dtype=float)
    xm = np.array([r.scale for r in mpi_s], dtype=float)
    em = np.array([r.efficiency for r in mpi_s], dtype=float)

    ax.plot(xo, eo, marker="o", linewidth=2, label="OpenMP efficiency")
    ax.plot(
        xm,
        em,
        marker="s",
        linewidth=2,
        label="MPI+OpenMP efficiency (relativa a 32 core)",
    )
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.7, label="Ideal efficiency")

    ax.set_title("Strong Scaling Efficiency")
    configure_core_axis(ax)
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0.0, 1.1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "strong_efficiency.png")


def plot_weak_runtime(out_dir: Path, mpi_w: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.array([r.scale for r in mpi_w], dtype=float)
    y = np.array([r.mean_s for r in mpi_w], dtype=float)
    e = np.array([r.std_s for r in mpi_w], dtype=float)
    ants = np.array([r.m for r in mpi_w], dtype=float)

    ax.errorbar(x, y, yerr=e, marker="o", capsize=3, linewidth=2, label="Weak scaling runtime")
    ax.axhline(y[0], linestyle="--", color="gray", label=f"Ideal constant time ({y[0]:.3f}s)")

    ax2 = ax.twinx()
    ax2.plot(x, ants, marker="s", linestyle=":", color="#2ca02c", linewidth=2, label="Ants (problem size)")
    ax2.set_ylabel("Ants (m)")

    ax.set_title("MPI+OpenMP Weak Scaling")
    configure_core_axis(ax)
    ax.set_ylabel("Mean Time (s)")
    ax.grid(True, alpha=0.25)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    save_plot(fig, out_dir / "weak_runtime.png")


def plot_weak_time_per_ant(out_dir: Path, mpi_w: list[Row]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.array([r.scale for r in mpi_w], dtype=float)
    y = np.array([r.mean_s / float(r.m) for r in mpi_w], dtype=float)

    ax.plot(x, y, marker="o", linewidth=2, label="Time per ant")
    ax.set_title("MPI+OpenMP Weak Scaling: Time per Ant")
    configure_core_axis(ax)
    ax.set_ylabel("Seconds / ant")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "weak_time_per_ant.png")


def first_cuda_error(raw_rows: list[RawRow]) -> str:
    for row in raw_rows:
        if row.warmup != 0:
            continue
        if row.backend != "cuda":
            continue
        if row.status == "error":
            msg = row.stderr.strip().replace("\n", " ")
            return msg[:140] + ("..." if len(msg) > 140 else "")
    return ""


def plot_backend_comparison(
    out_dir: Path,
    seq: list[Row],
    omp: list[Row],
    mpi_s: list[Row],
    cuda: list[Row],
    raw_rows: list[RawRow],
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    seq_time = seq[0].mean_s if seq else float("nan")
    omp_time = min((r.mean_s for r in omp), default=float("nan"))
    mpi_time = min((r.mean_s for r in mpi_s), default=float("nan"))
    cuda_time = min((r.mean_s for r in cuda), default=float("nan"))

    labels = ["SEQ", "OMP best", "MPI+OMP best", "CUDA"]
    values = [seq_time, omp_time, mpi_time, cuda_time]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

    x = np.arange(len(labels), dtype=float)
    max_valid = max(v for v in values if not np.isnan(v))
    ymax = max_valid * 1.22

    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        if np.isnan(value):
            ax.bar(x[idx], 0.0, color="#d3d3d3", edgecolor="#666666", hatch="//")
            ax.text(x[idx], ymax * 0.04, "N/A", ha="center", va="bottom",
                    fontsize=10, color="#333333")
        else:
            ax.bar(x[idx], value, color=color, alpha=0.9)
            ax.text(x[idx], value + ymax * 0.02, f"{value:.3f}s",
                    ha="center", va="bottom", fontsize=9)

    cuda_note = ""
    if np.isnan(cuda_time):
        cuda_note = first_cuda_error(raw_rows)
        if cuda_note:
            ax.text(
                0.02,
                0.96,
                f"CUDA unavailable: {cuda_note}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#7f0000",
            )

    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Time (s)")
    ax.set_title("Best Runtime Comparison Across Backends")
    ax.grid(True, axis="y", alpha=0.25)
    save_plot(fig, out_dir / "backend_comparison.png")


def plot_seq_vs_parallel_speedup(
    out_dir: Path,
    seq: list[Row],
    omp: list[Row],
    mpi_s: list[Row],
    cuda: list[Row],
) -> None:
    if not seq:
        return

    seq_t = seq[0].mean_s

    labels: list[str] = ["SEQ"]
    speedups: list[float] = [1.0]
    colors: list[str] = ["#4c72b0"]

    for r in omp:
        labels.append(f"OMP x{r.scale}")
        speedups.append(seq_t / r.mean_s)
        colors.append("#55a868")

    for r in mpi_s:
        labels.append(f"MPI+OMP x{r.scale}")
        speedups.append(seq_t / r.mean_s)
        colors.append("#c44e52")

    for r in cuda:
        labels.append("CUDA")
        speedups.append(seq_t / r.mean_s)
        colors.append("#8172b2")

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    x = np.arange(len(labels))
    bars = ax.bar(x, speedups, color=colors, alpha=0.9)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.2, label="SEQ baseline")

    for b, s in zip(bars, speedups):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            s + 0.04,
            f"{s:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Speedup vs SEQ")
    ax.set_title("Seq vs Parallel: Strong-Scaling Speedup")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "seq_vs_parallel_speedup.png")


def combined_strong_rows(omp: list[Row], mpi_s: list[Row]) -> list[Row]:
    """Build 1..32 from one-node OpenMP, then 64/128 from MPI+OpenMP."""
    by_scale: dict[int, Row] = {
        row.scale: row for row in omp if row.scale in CORE_TICKS and row.scale <= 32
    }
    for row in mpi_s:
        if row.scale in (64, 128):
            by_scale[row.scale] = row
    return [by_scale[x] for x in CORE_TICKS if x in by_scale]


def plot_combined_strong(out_dir: Path, omp: list[Row], mpi_s: list[Row]) -> None:
    rows = combined_strong_rows(omp, mpi_s)
    if not rows:
        return
    x = np.array([r.scale for r in rows], dtype=float)
    elapsed = np.array([r.mean_s for r in rows], dtype=float)
    errors = np.array([r.std_s for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(x, elapsed, yerr=errors, marker="o", capsize=3, linewidth=2)
    configure_core_axis(ax)
    ax.set_ylabel("Tempo medio (s)")
    ax.set_title("Strong Scaling Combinato OpenMP + MPI")
    ax.grid(True, alpha=0.25)
    save_plot(fig, out_dir / "strong_combined_runtime.png")

    baseline_time = elapsed[0]
    baseline_scale = x[0]
    speedup = baseline_time / elapsed
    ideal = x / baseline_scale
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(x, speedup, marker="o", linewidth=2, label="Misurato")
    ax.plot(x, ideal, linestyle="--", color="gray", label="Ideale")
    configure_core_axis(ax)
    ax.set_ylabel("Speedup")
    ax.set_title("Strong Scaling Combinato: Speedup 1–128 core")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(fig, out_dir / "strong_combined_speedup.png")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate scaling plots from CSV summary.")
    parser.add_argument("--summary", default="reports/scaling_summary.csv", help="Input summary CSV path")
    parser.add_argument("--raw", default="reports/scaling_results_raw.csv", help="Input raw CSV path")
    parser.add_argument("--out-dir", default="reports/plots", help="Output directory for PNG plots")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    raw_path = Path(args.raw)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    rows = [
        r
        for r in load_rows(summary_path)
        if r.scale <= MAX_TOTAL_CORES
        and r.ranks <= MAX_NODES
        and r.threads_per_rank <= MAX_CPUS_PER_TASK
    ]
    raw_rows = load_raw_rows(raw_path)
    omp = pick(rows, "omp_strong")
    mpi_s = pick(rows, "mpi_omp_strong")
    mpi_w = pick(rows, "mpi_omp_weak")
    seq = pick(rows, "seq_strong_baseline")
    cuda = pick(rows, "cuda_fixed")

    if not omp or not mpi_s or not mpi_w:
        raise RuntimeError("missing required experiments in summary CSV")

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(style_name)
            break
        except OSError:
            continue
    plot_strong_runtime(out_dir, omp, mpi_s, seq)
    plot_strong_speedup(out_dir, omp, mpi_s)
    plot_strong_efficiency(out_dir, omp, mpi_s)
    plot_combined_strong(out_dir, omp, mpi_s)
    plot_weak_runtime(out_dir, mpi_w)
    plot_weak_time_per_ant(out_dir, mpi_w)
    plot_backend_comparison(out_dir, seq, omp, mpi_s, cuda, raw_rows)
    plot_seq_vs_parallel_speedup(out_dir, seq, omp, mpi_s, cuda)

    print(f"Plots written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
