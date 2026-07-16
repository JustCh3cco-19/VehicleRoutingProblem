#!/usr/bin/env python3
"""
Summarize practical experiment outputs into CSV + Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, pstdev

MAX_NODES = 4
MAX_CPUS_PER_TASK = 32
MAX_TOTAL_CORES = 128


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fval(text: str) -> float | None:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def ival(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def summarize_scaling(
    rows: list[dict[str, str]], scale_mode: str, weak: bool = False
) -> list[dict[str, str]]:
    ok = [r for r in rows if r.get("status") == "ok"]
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = {}

    for r in ok:
        n = ival(r.get("n", "0"), 0)
        ranks = ival(r.get("mpi_ranks", "1"), 1)
        threads = ival(r.get("omp_threads", "1"), 1)
        if (
            ranks <= 0
            or ranks > MAX_NODES
            or threads <= 0
            or threads > MAX_CPUS_PER_TASK
            or ranks * threads > MAX_TOTAL_CORES
        ):
            continue
        groups.setdefault((n, ranks, threads), []).append(r)

    out: list[dict[str, str]] = []
    for (n, ranks, threads), grp in sorted(groups.items()):
        times = [fval(x.get("elapsed_s", "")) for x in grp]
        costs = [fval(x.get("best_cost", "")) for x in grp]
        times = [t for t in times if t is not None and t > 0]
        costs = [c for c in costs if c is not None]
        if not times:
            continue

        if scale_mode == "openmp":
            scale = threads
        else:
            # MPI and combined hybrid curves use total allocated CPU cores.
            scale = ranks * threads

        out.append(
            {
                "n": str(n),
                "ranks": str(ranks),
                "threads": str(threads),
                "scale": str(scale),
                "runs": str(len(times)),
                "mean_s": f"{mean(times):.6f}",
                "std_s": f"{pstdev(times) if len(times) > 1 else 0.0:.6f}",
                "mean_cost": f"{mean(costs):.6f}" if costs else "",
                "std_cost": f"{pstdev(costs) if len(costs) > 1 else 0.0:.6f}" if costs else "",
                "speedup": "",
                "efficiency": "",
            }
        )

    if not out:
        return out

    base = min(out, key=lambda x: int(x["scale"]))
    base_scale = float(base["scale"])
    base_time = float(base["mean_s"])
    for row in out:
        scale = float(row["scale"])
        t = float(row["mean_s"])
        if t <= 0.0 or base_time <= 0.0:
            continue
        time_ratio = base_time / t
        norm_scale = scale / base_scale if base_scale > 0.0 else math.nan
        efficiency = (
            time_ratio
            if weak
            else time_ratio / norm_scale if norm_scale > 0.0 else math.nan
        )
        row["speedup"] = "" if weak else f"{time_ratio:.6f}"
        row["efficiency"] = f"{efficiency:.6f}"
    return out


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def load_backend_rows(root: Path, exp_dir: str, filename: str) -> list[dict[str, str]]:
    return read_rows(root / "solve_manifest" / "csv" / exp_dir / filename)


def summarize_quality(root: Path) -> list[dict[str, str]]:
    backends: list[tuple[str, str, str]] = [
        ("seq", "exp_quality_seq", "manifest_seq_per_instance_results.csv"),
        ("openmp", "exp_quality_openmp", "manifest_openmp_mpi_per_instance_results.csv"),
        ("mpi", "exp_quality_mpi", "manifest_openmp_mpi_per_instance_results.csv"),
        ("hybrid", "exp_quality_hybrid", "manifest_openmp_mpi_per_instance_results.csv"),
        ("cuda", "exp_quality_cuda", "manifest_cuda_per_instance_results.csv"),
    ]
    out: list[dict[str, str]] = []
    for backend, exp_dir, file_name in backends:
        rows = [r for r in load_backend_rows(root, exp_dir, file_name) if r.get("status") == "ok"]
        if not rows:
            continue
        groups: dict[int, list[float]] = {}
        for r in rows:
            n = ival(r.get("n", "0"), 0)
            c = fval(r.get("best_cost", ""))
            if n <= 0 or c is None:
                continue
            groups.setdefault(n, []).append(c)
        for n, vals in sorted(groups.items()):
            out.append(
                {
                    "backend": backend,
                    "n": str(n),
                    "runs": str(len(vals)),
                    "best_cost": f"{min(vals):.6f}",
                    "avg_cost": f"{mean(vals):.6f}",
                    "std_cost": f"{pstdev(vals) if len(vals) > 1 else 0.0:.6f}",
                }
            )
    return out


def aggregate_by_n(rows: list[dict[str, str]]) -> dict[int, float]:
    groups: dict[int, list[float]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        n = ival(r.get("n", "0"), 0)
        t = fval(r.get("elapsed_s", ""))
        if n <= 0 or t is None or t <= 0:
            continue
        groups.setdefault(n, []).append(t)
    return {n: mean(vals) for n, vals in groups.items()}


def summarize_cuda_comparison(root: Path) -> list[dict[str, str]]:
    seq_rows = load_backend_rows(root, "exp_seq_vs_cuda_practical", "manifest_seq_per_instance_results.csv")
    omp_rows = load_backend_rows(root, "exp_openmp_vs_cuda_practical", "manifest_openmp_mpi_per_instance_results.csv")
    cuda_rows = load_backend_rows(
        root, "exp_cuda_size_sweep_practical", "manifest_cuda_per_instance_results.csv"
    )

    seq_n = aggregate_by_n(seq_rows)
    omp_n = aggregate_by_n(omp_rows)
    cuda_n = aggregate_by_n(cuda_rows)

    out: list[dict[str, str]] = []
    for n in sorted(cuda_n.keys()):
        cuda_t = cuda_n[n]
        seq_t = seq_n.get(n)
        omp_t = omp_n.get(n)
        out.append(
            {
                "n": str(n),
                "seq_mean_s": f"{seq_t:.6f}" if seq_t is not None else "",
                "openmp_mean_s": f"{omp_t:.6f}" if omp_t is not None else "",
                "cuda_mean_s": f"{cuda_t:.6f}",
                "seq_vs_cuda_speedup": f"{(seq_t / cuda_t):.6f}" if seq_t is not None and cuda_t > 0 else "",
                "openmp_vs_cuda_speedup": (
                    f"{(omp_t / cuda_t):.6f}" if omp_t is not None and cuda_t > 0 else ""
                ),
            }
        )
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_report(
    out_path: Path,
    scaling: dict[str, list[dict[str, str]]],
    quality_rows: list[dict[str, str]],
    cuda_cmp_rows: list[dict[str, str]],
) -> None:
    lines: list[str] = []
    lines.append("# Practical Experiment Summary")
    lines.append("")

    for title, rows in scaling.items():
        lines.append(f"## {title}")
        lines.append("")
        if not rows:
            lines.append("No rows found.")
            lines.append("")
            continue
        lines.append(
            md_table(
                ["n", "ranks", "threads", "scale", "runs", "mean_s", "std_s", "speedup", "efficiency"],
                [
                    [
                        r["n"],
                        r["ranks"],
                        r["threads"],
                        r["scale"],
                        r["runs"],
                        r["mean_s"],
                        r["std_s"],
                        r["speedup"] or "-",
                        r["efficiency"] or "-",
                    ]
                    for r in rows
                ],
            )
        )
        lines.append("")

    lines.append("## Quality")
    lines.append("")
    if quality_rows:
        lines.append(
            md_table(
                ["backend", "n", "runs", "best_cost", "avg_cost", "std_cost"],
                [[r["backend"], r["n"], r["runs"], r["best_cost"], r["avg_cost"], r["std_cost"]] for r in quality_rows],
            )
        )
    else:
        lines.append("No quality rows found.")
    lines.append("")

    lines.append("## CUDA Comparison")
    lines.append("")
    if cuda_cmp_rows:
        lines.append(
            md_table(
                ["n", "seq_mean_s", "openmp_mean_s", "cuda_mean_s", "seq_vs_cuda_speedup", "openmp_vs_cuda_speedup"],
                [
                    [
                        r["n"],
                        r["seq_mean_s"] or "-",
                        r["openmp_mean_s"] or "-",
                        r["cuda_mean_s"] or "-",
                        r["seq_vs_cuda_speedup"] or "-",
                        r["openmp_vs_cuda_speedup"] or "-",
                    ]
                    for r in cuda_cmp_rows
                ],
            )
        )
    else:
        lines.append("No CUDA comparison rows found.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Summarize practical experiment outputs.")
    p.add_argument("--root", required=True, help="Campaign root (results/practical_campaign/<tag>).")
    args = p.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"campaign root not found: {root}")
    out_dir = root / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_cfg: list[tuple[str, str, str, str, bool]] = [
        ("Strong OpenMP", "exp_strong_openmp_practical", "manifest_openmp_mpi_per_instance_results.csv", "openmp", False),
        ("Strong MPI (speedup relativo al punto a 32 core)", "exp_strong_mpi_practical", "manifest_openmp_mpi_per_instance_results.csv", "mpi", False),
        ("Strong Combined (OpenMP + MPI)", "exp_strong_hybrid_practical", "manifest_openmp_mpi_per_instance_results.csv", "hybrid", False),
        ("Weak OpenMP", "exp_weak_openmp_practical", "manifest_openmp_mpi_per_instance_results.csv", "openmp", True),
        ("Weak MPI", "exp_weak_mpi_practical", "manifest_openmp_mpi_per_instance_results.csv", "mpi", True),
        ("Weak Hybrid", "exp_weak_hybrid_practical", "manifest_openmp_mpi_per_instance_results.csv", "hybrid", True),
    ]

    scaling_summaries: dict[str, list[dict[str, str]]] = {}
    for title, exp_dir, file_name, mode, weak in scaling_cfg:
        rows = load_backend_rows(root, exp_dir, file_name)
        summary = summarize_scaling(rows, mode, weak=weak)
        scaling_summaries[title] = summary
        if summary:
            write_csv(
                out_dir / f"{exp_dir}_summary.csv",
                summary,
                [
                    "n",
                    "ranks",
                    "threads",
                    "scale",
                    "runs",
                    "mean_s",
                    "std_s",
                    "mean_cost",
                    "std_cost",
                    "speedup",
                    "efficiency",
                ],
            )

    quality_rows = summarize_quality(root)
    if quality_rows:
        write_csv(
            out_dir / "quality_summary.csv",
            quality_rows,
            ["backend", "n", "runs", "best_cost", "avg_cost", "std_cost"],
        )

    cuda_cmp_rows = summarize_cuda_comparison(root)
    if cuda_cmp_rows:
        write_csv(
            out_dir / "cuda_comparison_summary.csv",
            cuda_cmp_rows,
            ["n", "seq_mean_s", "openmp_mean_s", "cuda_mean_s", "seq_vs_cuda_speedup", "openmp_vs_cuda_speedup"],
        )

    write_report(out_dir / "practical_experiments_report.md", scaling_summaries, quality_rows, cuda_cmp_rows)
    print("Wrote summaries in", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
