#!/usr/bin/env python3
"""Generate benchmark plots for OpenMP, MPI, Hybrid, and Seq vs CUDA."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

try:
    from adjustText import adjust_text  # type: ignore
except Exception:
    adjust_text = None

MPI_COLUMNS = [
    "name",
    "profile",
    "instance_path",
    "n",
    "K",
    "m",
    "solver_seed",
    "instance_seed",
    "layout_id",
    "run_id",
    "status",
    "elapsed_s",
    "max_rss_gb",
    "best_cost",
    "error",
    "mpi_ranks",
    "omp_threads",
    "batch_id",
]


@dataclass
class Row:
    data: Dict[str, str]

    def f(self, key: str) -> float:
        return float(self.data[key])

    def i(self, key: str) -> int:
        return int(float(self.data[key]))

    def s(self, key: str) -> str:
        return self.data.get(key, "")


def read_csv(path: Path, fallback_header: List[str] | None = None) -> List[Row]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        first = f.readline()
        f.seek(0)
        has_header = first.strip().startswith("name,") or first.strip().startswith("source,")
        rows: List[Row] = []
        if has_header:
            for r in csv.DictReader(f):
                if not r:
                    continue
                rows.append(Row(r))
            return rows
        if fallback_header is None:
            raise ValueError(f"{path} has no header and no fallback header provided")
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            if len(raw) < len(fallback_header):
                continue
            rows.append(Row({k: v for k, v in zip(fallback_header, raw)}))
        return rows


def ok_rows(rows: Iterable[Row]) -> List[Row]:
    out = []
    for r in rows:
        if r.s("status") and r.s("status") != "ok":
            continue
        try:
            _ = r.f("elapsed_s")
            _ = r.f("best_cost")
            _ = r.i("n")
            out.append(r)
        except Exception:
            continue
    return out


def group_best(rows: Iterable[Row], keys: Tuple[str, ...], metric: str = "elapsed_s") -> Dict[Tuple[int, ...], Row]:
    best: Dict[Tuple[int, ...], Row] = {}
    for r in rows:
        key = tuple(r.i(k) for k in keys)
        if key not in best or r.f(metric) < best[key].f(metric):
            best[key] = r
    return best


def _offset_for_series(series_idx: int) -> Tuple[int, int]:
    offsets = [(-16, 10), (14, 10), (-16, -12), (14, -12), (0, 14), (0, -14)]
    return offsets[series_idx % len(offsets)]


def _clamp_offset_for_axes(ax, x: float, y: float, dx: int, dy: int) -> Tuple[int, int]:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xr = (x - xmin) / (xmax - xmin) if xmax > xmin else 0.5
    yr = (y - ymin) / (ymax - ymin) if ymax > ymin else 0.5
    if xr > 0.93 and dx > 0:
        dx = -abs(dx)
    if xr < 0.07 and dx < 0:
        dx = abs(dx)
    if yr > 0.93 and dy > 0:
        dy = -abs(dy)
    if yr < 0.07 and dy < 0:
        dy = abs(dy)
    return dx, dy


def _plot_line(ax, xs: List[float], ys: List[float], label: str) -> None:
    ax.plot(xs, ys, marker="o", label=label)


def save(fig, path: Path, dpi: int) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_openmp(openmp_rows: List[Row], out_dir: Path, dpi: int, use_adjust_text: bool) -> None:
    target_threads = [1, 4, 8, 16, 32]
    best = group_best(ok_rows(openmp_rows), ("n", "omp_threads"))

    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    for sidx, t in enumerate(target_threads):
        points = sorted((k[0], v) for k, v in best.items() if k[1] == t)
        if not points:
            continue
        xs = [p[0] for p in points]
        ys_t = [p[1].f("elapsed_s") for p in points]
        ys_c = [p[1].f("best_cost") for p in points]
        _plot_line(ax_t, xs, ys_t, f"{t} threads")
        _plot_line(ax_c, xs, ys_c, f"{t} threads")

    ax_t.set_title("OpenMP: Execution Time vs Problem Size")
    ax_t.set_xlabel("Problem Size (n clients)")
    ax_t.set_ylabel("Execution Time (s)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(title="OpenMP Threads", fontsize=8)
    save(fig_t, out_dir / "openmp_time_vs_size.png", dpi)

    ax_c.set_title("OpenMP: Best Cost vs Problem Size")
    ax_c.set_xlabel("Problem Size (n clients)")
    ax_c.set_ylabel("Best Cost")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(title="OpenMP Threads", fontsize=8)
    save(fig_c, out_dir / "openmp_best_cost_vs_size.png", dpi)


def plot_mpi_strong(mpi_rows: List[Row], out_dir: Path, dpi: int, use_adjust_text: bool) -> None:
    best = group_best(ok_rows(mpi_rows), ("n", "mpi_ranks"))
    sizes = sorted({k[0] for k in best})

    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    fig_s, ax_s = plt.subplots(figsize=(10, 6))
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    for sidx, n in enumerate(sizes):
        points = sorted((k[1], v) for k, v in best.items() if k[0] == n)
        if not points:
            continue
        ranks = [p[0] for p in points]
        times = [p[1].f("elapsed_s") for p in points]
        costs = [p[1].f("best_cost") for p in points]
        _plot_line(ax_t, ranks, times, f"n={n}")
        _plot_line(ax_c, ranks, costs, f"n={n}")
        t1 = None
        for r, t in zip(ranks, times):
            if r == 1:
                t1 = t
                break
        if t1 is not None:
            speedup = [t1 / t for t in times]
            ax_s.plot(ranks, speedup, marker="o", label=f"n={n}")

    ax_t.set_title("MPI Strong Scaling: Time vs Ranks (32 threads/rank)")
    ax_t.set_xlabel("MPI Ranks")
    ax_t.set_ylabel("Execution Time (s)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)
    ax_s.set_title("MPI Strong Scaling: Speedup vs Ranks")
    ax_s.set_xlabel("MPI Ranks")
    ax_s.set_ylabel("Speedup (T1 / Tr)")
    all_ranks = sorted({k[1] for k in best})
    if all_ranks:
        ax_s.plot(all_ranks, all_ranks, linestyle="--", color="black", linewidth=1.2, label="Ideal speedup")
    ax_s.grid(True, alpha=0.3)
    ax_s.legend(fontsize=8)
    save(fig_t, out_dir / "mpi_strong_time_vs_ranks.png", dpi)
    save(fig_s, out_dir / "mpi_strong_speedup_vs_ranks.png", dpi)

    ax_c.set_title("MPI Strong Scaling: Best Cost vs Ranks")
    ax_c.set_xlabel("MPI Ranks")
    ax_c.set_ylabel("Best Cost")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)
    save(fig_c, out_dir / "mpi_strong_best_cost_vs_ranks.png", dpi)


def plot_hybrid(hybrid_rows: List[Row], out_dir: Path, dpi: int, use_adjust_text: bool) -> None:
    clean = ok_rows(hybrid_rows)
    for r in clean:
        total = r.i("mpi_ranks") * r.i("omp_threads")
        r.data["total_cores"] = str(total)

    best = group_best(clean, ("n", "total_cores"))
    sizes = sorted({k[0] for k in best})

    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    for sidx, n in enumerate(sizes):
        points = sorted((k[1], v) for k, v in best.items() if k[0] == n)
        cores = [p[0] for p in points]
        times = [p[1].f("elapsed_s") for p in points]
        costs = [p[1].f("best_cost") for p in points]
        _plot_line(ax_t, cores, times, f"n={n}")
        _plot_line(ax_c, cores, costs, f"n={n}")

    ax_t.set_title("Hybrid OpenMP+MPI: Time vs Total Cores")
    ax_t.set_xlabel("Total Cores (MPI ranks × OMP threads)")
    ax_t.set_ylabel("Execution Time (s)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)
    save(fig_t, out_dir / "hybrid_time_vs_total_cores.png", dpi)

    ax_c.set_title("Hybrid OpenMP+MPI: Best Cost vs Total Cores")
    ax_c.set_xlabel("Total Cores (MPI ranks × OMP threads)")
    ax_c.set_ylabel("Best Cost")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)
    save(fig_c, out_dir / "hybrid_best_cost_vs_total_cores.png", dpi)


def plot_seq_vs_cuda(seq_rows: List[Row], cuda_rows: List[Row], out_dir: Path, dpi: int, use_adjust_text: bool) -> None:
    seq_best = group_best(ok_rows(seq_rows), ("n",))
    cuda_best = group_best(ok_rows(cuda_rows), ("n",))

    fig_t, ax_t = plt.subplots(figsize=(10, 6))
    fig_c, ax_c = plt.subplots(figsize=(10, 6))
    if seq_best:
        xs = sorted(k[0] for k in seq_best)
        ys_t = [seq_best[(x,)].f("elapsed_s") for x in xs]
        ys_c = [seq_best[(x,)].f("best_cost") for x in xs]
        _plot_line(ax_t, xs, ys_t, "Sequential")
        _plot_line(ax_c, xs, ys_c, "Sequential")
    if cuda_best:
        xs = sorted(k[0] for k in cuda_best)
        ys_t = [cuda_best[(x,)].f("elapsed_s") for x in xs]
        ys_c = [cuda_best[(x,)].f("best_cost") for x in xs]
        _plot_line(ax_t, xs, ys_t, "CUDA")
        _plot_line(ax_c, xs, ys_c, "CUDA")

    ax_t.set_title("Sequential vs CUDA: Execution Time vs Problem Size")
    ax_t.set_xlabel("Problem Size (n clients)")
    ax_t.set_ylabel("Execution Time (s)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)
    save(fig_t, out_dir / "seq_vs_cuda_time.png", dpi)

    ax_c.set_title("Sequential vs CUDA: Best Cost vs Problem Size")
    ax_c.set_xlabel("Problem Size (n clients)")
    ax_c.set_ylabel("Best Cost")
    ax_c.grid(True, alpha=0.3)
    ax_c.legend(fontsize=8)
    save(fig_c, out_dir / "seq_vs_cuda_best_cost_vs_size.png", dpi)


def plot_convergence(
    openmp_rows: List[Row],
    mpi_rows: List[Row],
    hybrid_rows: List[Row],
    seq_rows: List[Row],
    cuda_rows: List[Row],
    out_dir: Path,
    dpi: int,
) -> None:
    impls: Dict[str, Dict[int, float]] = {}
    impls["OpenMP"] = {k[0]: v.f("best_cost") for k, v in group_best(ok_rows(openmp_rows), ("n",)).items()}
    impls["MPI strong"] = {k[0]: v.f("best_cost") for k, v in group_best(ok_rows(mpi_rows), ("n",)).items()}

    hy = ok_rows(hybrid_rows)
    for r in hy:
        r.data["total_cores"] = str(r.i("mpi_ranks") * r.i("omp_threads"))
    impls["Hybrid"] = {k[0]: v.f("best_cost") for k, v in group_best(hy, ("n",)).items()}
    impls["Sequential"] = {k[0]: v.f("best_cost") for k, v in group_best(ok_rows(seq_rows), ("n",)).items()}
    impls["CUDA"] = {k[0]: v.f("best_cost") for k, v in group_best(ok_rows(cuda_rows), ("n",)).items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in impls.items():
        if not values:
            continue
        xs = sorted(values)
        ys = [values[x] for x in xs]
        ax.plot(xs, ys, marker="o", label=name)

    ax.set_title("Convergence Value (Best Cost) vs Problem Size")
    ax.set_xlabel("Problem Size (n clients)")
    ax.set_ylabel("Best Cost (final convergence value)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save(fig, out_dir / "convergence_best_cost_vs_size.png", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plots from CSV files in nuovi_grafici.")
    parser.add_argument("--data-dir", default="nuovi_grafici", help="Directory containing CSV files.")
    parser.add_argument("--out-dir", default="nuovi_grafici/plots", help="Directory for PNG output.")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution (DPI).")
    parser.add_argument(
        "--use-adjust-text",
        action="store_true",
        help="Use adjustText (if installed) to reduce annotation overlaps automatically.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    openmp_rows = read_csv(data_dir / "openmp_strong.csv", fallback_header=MPI_COLUMNS)
    mpi_rows = read_csv(data_dir / "mpi_strong.csv", fallback_header=MPI_COLUMNS)
    hybrid_rows = read_csv(data_dir / "openmp_mpi_strong.csv", fallback_header=MPI_COLUMNS)
    seq_rows = read_csv(data_dir / "seq.csv")
    cuda_rows = read_csv(data_dir / "cuda_merged_with_v6.csv")

    plot_openmp(openmp_rows, out_dir, args.dpi, args.use_adjust_text)
    plot_mpi_strong(mpi_rows, out_dir, args.dpi, args.use_adjust_text)
    plot_hybrid(hybrid_rows, out_dir, args.dpi, args.use_adjust_text)
    plot_seq_vs_cuda(seq_rows, cuda_rows, out_dir, args.dpi, args.use_adjust_text)
    plot_convergence(openmp_rows, mpi_rows, hybrid_rows, seq_rows, cuda_rows, out_dir, args.dpi)

    print(f"Plots saved in: {out_dir}")


if __name__ == "__main__":
    main()
