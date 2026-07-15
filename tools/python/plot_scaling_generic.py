#!/usr/bin/env python3
"""Aggregate scaling CSV and optionally generate a speedup-only plot.

Usage:
  python3 tools/python/plot_scaling_generic.py \
    --csv results/solve_manifest/csv/exp_strong_openmp/manifest_openmp_mpi_per_instance_results.csv \
    --x omp_threads \
    --out-csv results/solve_manifest/csv/exp_strong_openmp/summary.csv \
    --out-plot results/solve_manifest/csv/exp_strong_openmp/plot_speedup.png \
    --title "Strong scaling OpenMP"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

MAX_TOTAL_CORES = 128
CORE_TICKS = [1, 2, 4, 8, 16, 32, 64, 128]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV with per-run rows.")
    p.add_argument(
        "--x",
        required=True,
        help=(
            "Column to group by (e.g. omp_threads), or total_cores to derive "
            "mpi_ranks * omp_threads."
        ),
    )
    p.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path for aggregated results.",
    )
    p.add_argument("--out-plot", help="Optional output PNG plot path.")
    p.add_argument("--title", default="", help="Optional plot title.")
    p.add_argument(
        "--filter",
        default="status=ok",
        help="Filter rows as key=value (default: status=ok).",
    )
    p.add_argument(
        "--filter-prefix",
        default="",
        help="Optional prefix filter: key=prefix (e.g., batch_id=strong_openmp_n2000_).",
    )
    p.add_argument(
        "--filter-contains",
        default="",
        help="Optional contains filter: key=substr (e.g., batch_id=_t8).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_plot = Path(args.out_plot) if args.out_plot else None

    if not in_path.exists():
        raise SystemExit(f"missing CSV: {in_path}")

    rows = read_rows(in_path)
    if not rows:
        raise SystemExit("empty CSV")

    key, _, val = args.filter.partition("=")
    if key:
        rows = [r for r in rows if r.get(key) == val]
    if args.filter_prefix:
        k, _, v = args.filter_prefix.partition("=")
        if k:
            rows = [r for r in rows if r.get(k, "").startswith(v)]
    if args.filter_contains:
        k, _, v = args.filter_contains.partition("=")
        if k:
            rows = [r for r in rows if v in r.get(k, "")]

    groups: dict[int, list[float]] = {}
    for r in rows:
        try:
            if args.x == "total_cores":
                g = int(r.get("mpi_ranks", "1")) * int(r.get("omp_threads", "1"))
            else:
                g = int(r.get(args.x, "0"))
            e = float(r.get("elapsed_s", ""))
        except ValueError:
            continue
        if g <= 0:
            continue
        if args.x == "omp_threads" and g > 32:
            continue
        if args.x == "total_cores" and g > MAX_TOTAL_CORES:
            continue
        groups.setdefault(g, []).append(e)

    if not groups:
        raise SystemExit("no valid rows after parsing")

    x_sorted = sorted(groups.keys())
    base_x = x_sorted[0]
    base_mean = mean(groups[base_x])

    summary_rows: list[dict[str, str]] = []
    for g in x_sorted:
        vals = groups[g]
        m = mean(vals)
        s = pstdev(vals) if len(vals) > 1 else 0.0
        speedup = base_mean / m if m > 0 else 0.0
        normalized_scale = float(g) / float(base_x)
        eff = speedup / normalized_scale if normalized_scale > 0 else 0.0
        summary_rows.append(
            {
                args.x: str(g),
                "runs": str(len(vals)),
                "mean_elapsed_s": f"{m:.4f}",
                "std_elapsed_s": f"{s:.4f}",
                "speedup": f"{speedup:.4f}",
                "efficiency": f"{eff:.4f}",
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                args.x,
                "runs",
                "mean_elapsed_s",
                "std_elapsed_s",
                "speedup",
                "efficiency",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if out_plot:
        import matplotlib.pyplot as plt

        x_vals = x_sorted
        real_speedup = [float(r["speedup"]) for r in summary_rows]
        ideal_speedup = [float(x) / float(x_vals[0]) for x in x_vals]

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(
            x_vals,
            real_speedup,
            color="blue",
            marker="+",
            markersize=8,
            linewidth=0.8,
            label="Real speedup",
        )
        ax.plot(
            x_vals,
            ideal_speedup,
            color="red",
            linewidth=0.8,
            label="Ideal speedup",
        )
        ax.set_xlabel(args.x)
        ax.set_ylabel("Speedup")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        if args.x in {"omp_threads", "total_cores"}:
            ticks = CORE_TICKS[:6] if args.x == "omp_threads" else CORE_TICKS
            ax.set_xlim(0.8, ticks[-1] * 1.25)
            ax.set_xticks(ticks)
        else:
            ax.set_xticks(x_vals)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(False)
        ax.legend(loc="upper left")

        if args.title:
            ax.set_title(args.title)

        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_plot, dpi=150)
        plt.close(fig)

    print(f"wrote summary CSV: {out_csv}")
    if out_plot:
        print(f"wrote plot: {out_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
