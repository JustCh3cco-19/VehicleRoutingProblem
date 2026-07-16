#!/usr/bin/env python3
"""Validate the manual campaign, write derived summaries, and plot paper figures.

Run from the repository root:
    python3 docs/paper/scripts/analyze_results.py

Raw files below results/ are read-only inputs.  All derived artifacts are
written below docs/paper/generated/ and docs/paper/figures/.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_STRONG_OMP = [1, 2, 4, 8, 16, 32]
EXPECTED_STRONG_MPI = [32, 64, 128]
EXPECTED_WEAK = [1, 2, 4, 8, 16, 32, 64, 128]
PROGRESS_RE = re.compile(
    r"^\[progress\].*epoca=(\d+).*epoche_rimanenti_prima_stop=(\d+).*"
    r"tempo_trascorso=([0-9.]+)s"
)
ROUTE_RE = re.compile(r"^Route\s+(\d+):\s*(.*)$")
BEST_RE = re.compile(r"^best cost:\s*([0-9.eE+-]+)\s*$")


@dataclass
class Instance:
    n: int
    vehicles: int
    capacity: int
    coords: list[tuple[float, float]]
    demands: list[int]


@dataclass
class Run:
    csv_path: Path
    family: str
    backend: str
    name: str
    instance_path: Path
    n: int
    k: int
    m: int
    seed: int
    run_id: int
    status: str
    elapsed_s: float
    best_cost: float
    ranks: int
    threads: int
    batch_id: str
    solution_path: Path
    termination: str = "unknown"
    last_epoch: int | None = None
    solution_feasible: bool = False
    cost_consistent: bool = False
    recomputed_cost: float | None = None
    included_timing: bool = False
    notes: str = ""

    @property
    def workers(self) -> int:
        return self.ranks * self.threads


def finite_positive(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"expected finite positive value, got {value!r}")
    return parsed


def family_and_backend(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "backend_sizes" in parts:
        idx = parts.index("backend_sizes")
        return "backend_sizes", parts[idx + 1]
    for family in ("strong_openmp", "strong_mpi", "weak_openmp", "weak_mpi"):
        if family in parts:
            return family, "openmp-mpi"
    return "unknown", "unknown"


def solution_path_for(csv_path: Path, row: dict[str, str], backend: str) -> Path:
    # The weak-OpenMP download is a single family-level manifest, while its
    # solutions remain separated by configuration.  Route those rows through
    # batch_id instead of applying the per-configuration CSV layout used by
    # the other campaigns.
    if csv_path.parent.name == "weak_openmp":
        batch_id = row.get("batch_id", "")
        match = re.fullmatch(r"weak_openmp_(c(\d+)_m(\d+))", batch_id)
        if not match:
            raise ValueError(f"invalid aggregate weak-OpenMP batch_id {batch_id!r}")
        config, batch_threads, batch_m = match.groups()
        if int(batch_threads) != int(row.get("omp_threads") or 1) or int(batch_m) != int(row["m"]):
            raise ValueError(f"weak-OpenMP batch_id {batch_id!r} disagrees with threads/m columns")
        return (
            csv_path.parent
            / config
            / "solutions"
            / "mpi"
            / f"{row['name']}_mpi_run{row['run_id']}_solution.txt"
        )

    base = csv_path.parent.parent / "solutions"
    suffix = {"seq": "seq", "cuda": "cuda"}.get(backend, "mpi")
    return base / suffix / f"{row['name']}_{suffix}_run{row['run_id']}_solution.txt"


def discover_runs(root: Path) -> tuple[list[Run], list[str]]:
    campaign = root / "results" / "manual_campaign"
    csv_paths = sorted(campaign.rglob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"no CSV files found below {campaign}")

    runs: list[Run] = []
    discovery_notes: list[str] = []
    for path in csv_paths:
        family, backend = family_and_backend(path)
        if family == "unknown":
            discovery_notes.append(f"Unclassified CSV: {path.relative_to(root)}")
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    elapsed = finite_positive(row.get("elapsed_s", ""))
                    cost = finite_positive(row.get("best_cost", ""))
                    n = int(row["n"])
                    k = int(row["K"])
                    m = int(row["m"])
                    run_id = int(row["run_id"])
                    seed = int(row["solver_seed"])
                    ranks = int(row.get("mpi_ranks") or 1)
                    threads = int(row.get("omp_threads") or 1)
                except (KeyError, TypeError, ValueError) as exc:
                    discovery_notes.append(
                        f"Malformed row in {path.relative_to(root)}: {exc}"
                    )
                    continue
                try:
                    solution_path = solution_path_for(path, row, backend)
                except (KeyError, TypeError, ValueError) as exc:
                    discovery_notes.append(
                        f"Malformed row in {path.relative_to(root)}: {exc}"
                    )
                    continue
                runs.append(
                    Run(
                        csv_path=path,
                        family=family,
                        backend=backend,
                        name=row["name"],
                        instance_path=root / row["instance_path"],
                        n=n,
                        k=k,
                        m=m,
                        seed=seed,
                        run_id=run_id,
                        status=row.get("status", ""),
                        elapsed_s=elapsed,
                        best_cost=cost,
                        ranks=ranks,
                        threads=threads,
                        batch_id=row.get("batch_id", ""),
                        solution_path=solution_path,
                    )
                )
    return runs, discovery_notes


def load_instance(path: Path) -> Instance:
    if not path.is_file():
        raise ValueError(f"missing instance {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    dimension = vehicles = capacity = None
    for line in lines:
        if line.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1])
        elif line.startswith("VEHICLES"):
            vehicles = int(line.split(":", 1)[1])
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":", 1)[1])
    if dimension is None or vehicles is None or capacity is None:
        raise ValueError(f"incomplete metadata in {path}")
    n = dimension - 1
    try:
        coord_at = lines.index("NODE_COORD_SECTION") + 1
        demand_at = lines.index("DEMAND_SECTION") + 1
    except ValueError as exc:
        raise ValueError(f"missing coordinate/demand section in {path}") from exc
    coords: list[tuple[float, float]] = []
    demands: list[int] = []
    for line in lines[coord_at : coord_at + dimension]:
        _, x, y = line.split()
        coords.append((float(x), float(y)))
    for line in lines[demand_at : demand_at + dimension]:
        _, demand = line.split()
        demands.append(int(demand))
    if len(coords) != dimension or len(demands) != dimension:
        raise ValueError(f"truncated instance {path}")
    return Instance(n, vehicles, capacity, coords, demands)


def parse_solution(path: Path) -> tuple[list[list[int]], float, tuple[int, int, float] | None]:
    routes: dict[int, list[int]] = {}
    best_cost = None
    last_progress = None
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            match = ROUTE_RE.match(line)
            if match:
                text = match.group(2).strip()
                routes[int(match.group(1))] = [int(x) for x in text.split()] if text else []
                continue
            match = BEST_RE.match(line)
            if match:
                best_cost = float(match.group(1))
                continue
            match = PROGRESS_RE.match(line)
            if match:
                last_progress = (int(match.group(1)), int(match.group(2)), float(match.group(3)))
    if best_cost is None:
        raise ValueError("missing final best cost")
    ordered = [routes[i] for i in sorted(routes)]
    return ordered, best_cost, last_progress


def validate_solution(run: Run, instance: Instance) -> list[str]:
    notes: list[str] = []
    if not run.solution_path.is_file():
        return ["missing solution file"]
    try:
        routes, printed_cost, progress = parse_solution(run.solution_path)
    except (OSError, ValueError) as exc:
        return [f"unreadable solution: {exc}"]

    if progress:
        run.last_epoch = progress[0]
        if run.family.startswith("strong_") or run.family.startswith("weak_"):
            run.termination = "fixed_epochs" if progress[0] == 100 and progress[1] == 0 else "incomplete_fixed_epochs"
        else:
            run.termination = "time_limit" if progress[2] >= 299.0 else "stagnation"
    if len(routes) != run.k:
        notes.append(f"expected {run.k} routes, found {len(routes)}")

    customers = [node for route in routes for node in route]
    counts = Counter(customers)
    invalid_nodes = [node for node in counts if node < 1 or node > instance.n]
    duplicates = [node for node, count in counts.items() if count != 1]
    missing_count = instance.n - len({node for node in customers if 1 <= node <= instance.n})
    over_capacity = []
    for idx, route in enumerate(routes, start=1):
        demand = sum(instance.demands[node] for node in route if 0 <= node <= instance.n)
        if demand > instance.capacity:
            over_capacity.append((idx, demand))
    if invalid_nodes:
        notes.append(f"{len(invalid_nodes)} out-of-range customer ids")
    if duplicates:
        notes.append(f"{len(duplicates)} duplicated customer ids")
    if missing_count:
        notes.append(f"{missing_count} customers missing")
    if over_capacity:
        notes.append(f"{len(over_capacity)} routes exceed capacity")
    run.solution_feasible = not notes

    recomputed = 0.0
    for route in routes:
        previous = 0
        for node in route:
            x1, y1 = instance.coords[previous]
            x2, y2 = instance.coords[node]
            recomputed += math.hypot(x1 - x2, y1 - y2)
            previous = node
        x1, y1 = instance.coords[previous]
        x2, y2 = instance.coords[0]
        recomputed += math.hypot(x1 - x2, y1 - y2)
    run.recomputed_cost = recomputed
    tolerance = max(0.1, abs(printed_cost) * 1e-4)
    csv_matches_printed = abs(printed_cost - run.best_cost) <= tolerance
    route_matches_printed = abs(recomputed - printed_cost) <= tolerance
    run.cost_consistent = csv_matches_printed and route_matches_printed
    if not csv_matches_printed:
        notes.append("CSV and printed best cost differ")
    if not route_matches_printed:
        notes.append(
            f"route cost differs from printed best cost by {abs(recomputed - printed_cost):.3f}"
        )
    return notes


def validate_runs(root: Path, runs: list[Run]) -> None:
    cache: dict[Path, Instance] = {}
    for run in runs:
        notes: list[str] = []
        if run.status != "ok":
            notes.append(f"CSV status is {run.status!r}")
        try:
            if run.instance_path not in cache:
                cache[run.instance_path] = load_instance(run.instance_path)
            instance = cache[run.instance_path]
            if (run.n, run.k) != (instance.n, instance.vehicles):
                notes.append("CSV n/K do not match instance metadata")
            notes.extend(validate_solution(run, instance))
        except (OSError, ValueError) as exc:
            notes.append(str(exc))
        fixed_ok = run.termination == "fixed_epochs" if run.family.startswith(("strong_", "weak_")) else True
        run.included_timing = run.status == "ok" and run.solution_feasible and fixed_ok
        run.notes = "; ".join(notes)


def stats(values: Iterable[float]) -> dict[str, float | int]:
    vals = list(values)
    if not vals:
        raise ValueError("cannot summarize empty sample")
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else math.nan
    return {
        "samples": len(vals),
        "mean_s": mean,
        "median_s": statistics.median(vals),
        "min_s": min(vals),
        "max_s": max(vals),
        "std_s": std,
        "cv_pct": 100.0 * std / mean if len(vals) > 1 and mean else math.nan,
    }


def group_runs(runs: list[Run], key) -> dict[tuple, list[Run]]:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        if run.included_timing:
            groups[key(run)].append(run)
    return dict(groups)


def summary_row(group: list[Run]) -> dict[str, object]:
    result: dict[str, object] = dict(stats(r.elapsed_s for r in group))
    costs = [r.best_cost for r in group if r.cost_consistent]
    result.update(
        {
            "mean_cost": statistics.fmean(costs) if costs else math.nan,
            "std_cost": statistics.stdev(costs) if len(costs) > 1 else math.nan,
            "cost_samples": len(costs),
        }
    )
    return result


def build_summaries(runs: list[Run]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    backend_rows: list[dict[str, object]] = []
    backend_groups = group_runs(runs, lambda r: (r.backend, r.n, r.k, r.m, r.termination))
    for (backend, n, k, m, termination), group in sorted(backend_groups.items()):
        if group[0].family != "backend_sizes":
            continue
        row = {"backend": backend, "n": n, "K": k, "m": m, "termination": termination}
        row.update(summary_row(group))
        backend_rows.append(row)

    scaling_rows: list[dict[str, object]] = []
    scaling_groups = group_runs(runs, lambda r: (r.family, r.n, r.k, r.m, r.ranks, r.threads))
    for (family, n, k, m, ranks, threads), group in sorted(scaling_groups.items()):
        if not family.startswith("strong_"):
            continue
        row = {
            "family": family,
            "n": n,
            "K": k,
            "m": m,
            "ranks": ranks,
            "threads": threads,
            "workers": ranks * threads,
        }
        row.update(summary_row(group))
        scaling_rows.append(row)

    for family in ("strong_openmp", "strong_mpi"):
        subset = [r for r in scaling_rows if r["family"] == family]
        if not subset:
            continue
        baseline = min(subset, key=lambda r: int(r["workers"]))
        base_p = float(baseline["workers"])
        base_t = float(baseline["mean_s"])
        base_std = float(baseline["std_s"])
        for row in subset:
            p = float(row["workers"])
            t = float(row["mean_s"])
            std = float(row["std_s"])
            speedup = base_t / t
            rel_p = p / base_p
            # At the baseline the ratio is the same sample mean divided by
            # itself, hence it is exactly one rather than a ratio of two
            # independent estimates.
            rel_error = 0.0 if p == base_p else math.sqrt((base_std / base_t) ** 2 + (std / t) ** 2)
            row["baseline_workers"] = int(base_p)
            row["speedup"] = speedup
            row["speedup_std"] = speedup * rel_error
            row["efficiency"] = speedup / rel_p
            row["efficiency_std"] = speedup * rel_error / rel_p

    combined: list[dict[str, object]] = []
    omp = {int(r["workers"]): r for r in scaling_rows if r["family"] == "strong_openmp"}
    mpi = {int(r["workers"]): r for r in scaling_rows if r["family"] == "strong_mpi"}
    for p in sorted(set(omp) | {x for x in mpi if x > 32}):
        source = omp.get(p) or mpi[p]
        combined.append(dict(source, source_family=source["family"]))
    if combined and int(combined[0]["workers"]) == 1:
        base_t = float(combined[0]["mean_s"])
        base_std = float(combined[0]["std_s"])
        for row in combined:
            p = float(row["workers"])
            t = float(row["mean_s"])
            std = float(row["std_s"])
            speedup = base_t / t
            rel_error = 0.0 if p == 1 else math.sqrt((base_std / base_t) ** 2 + (std / t) ** 2)
            row["speedup"] = speedup
            row["speedup_std"] = speedup * rel_error
            row["efficiency"] = speedup / p
            row["efficiency_std"] = speedup * rel_error / p

    weak_rows: list[dict[str, object]] = []
    weak_groups = group_runs(runs, lambda r: (r.family, r.n, r.k, r.m, r.ranks, r.threads))
    for (family, n, k, m, ranks, threads), group in sorted(weak_groups.items()):
        if not family.startswith("weak_"):
            continue
        workers = ranks * threads
        row = {
            "family": family,
            "n": n,
            "K": k,
            "m": m,
            "ranks": ranks,
            "threads": threads,
            "workers": workers,
            "ants_per_worker": m / workers,
        }
        row.update(summary_row(group))
        weak_rows.append(row)
    if weak_rows:
        baseline = min(weak_rows, key=lambda r: int(r["workers"]))
        base_p = int(baseline["workers"])
        base_t = float(baseline["mean_s"])
        base_std = float(baseline["std_s"])
        for row in weak_rows:
            t = float(row["mean_s"])
            std = float(row["std_s"])
            efficiency = base_t / t
            rel_error = 0.0 if int(row["workers"]) == base_p else math.sqrt((base_std / base_t) ** 2 + (std / t) ** 2)
            row["baseline_workers"] = base_p
            row["weak_efficiency"] = efficiency
            row["weak_efficiency_std"] = efficiency * rel_error
    return backend_rows, scaling_rows, combined, weak_rows


def format_value(value: object) -> object:
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6f}"
    return value


def display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )


def core_axis(ax: plt.Axes, ticks: list[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(x) for x in ticks], rotation=35, ha="right")
    ax.set_xlabel("Total CPU workers p")


def select_weak_curve(weak: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build the documented combined curve without duplicating 32 workers."""
    selected: dict[int, dict[str, object]] = {}
    for row in weak:
        workers = int(row["workers"])
        family = str(row["family"])
        preferred = family == ("weak_openmp" if workers <= 32 else "weak_mpi")
        if workers not in selected or preferred:
            selected[workers] = row
    return [selected[workers] for workers in sorted(selected)]


def finish_metric_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    path: Path,
    title: str,
    legend_handles: list[object] | None = None,
    legend_labels: list[str] | None = None,
) -> None:
    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=len(legend_handles),
            frameon=False,
            columnspacing=1.0,
            handletextpad=0.45,
        )
        top = 0.72
    else:
        top = 0.84
    fig.tight_layout(rect=(0, 0, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_strong(figure_dir: Path, scaling: list[dict[str, object]], combined: list[dict[str, object]]) -> None:
    omp = sorted((r for r in scaling if r["family"] == "strong_openmp"), key=lambda r: int(r["workers"]))
    mpi = sorted((r for r in scaling if r["family"] == "strong_mpi"), key=lambda r: int(r["workers"]))
    if not omp:
        raise SystemExit("mandatory strong OpenMP baseline is missing")
    fig, ax = plt.subplots(figsize=(3.45, 2.85))
    for rows, label, marker, color in (
        (omp, "OpenMP", "o", "#1f77b4"),
        (mpi, "MPI", "s", "#d95f02"),
    ):
        if not rows:
            continue
        x = np.array([r["workers"] for r in rows], dtype=float)
        y = np.array([r["mean_s"] for r in rows], dtype=float)
        e = np.array([r["std_s"] for r in rows], dtype=float)
        ax.errorbar(x, y, yerr=e, fmt=marker, color=color, capsize=3, label=label)
    ax.set_ylabel("End-to-end runtime (s)")
    core_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128])
    handles, labels = ax.get_legend_handles_labels()
    finish_metric_plot(fig, ax, figure_dir / "strong_scaling_runtime.pdf", "Strong scaling", handles, labels)

    if combined:
        fig, ax = plt.subplots(figsize=(3.45, 2.85))
        x = np.array([r["workers"] for r in combined], dtype=float)
        speed = np.array([r["speedup"] for r in combined], dtype=float)
        speed_sd = np.array([r["speedup_std"] for r in combined], dtype=float)
        eff = np.array([r["efficiency"] for r in combined], dtype=float)
        eff_sd = np.array([r["efficiency_std"] for r in combined], dtype=float)
        ax.errorbar(x, speed, yerr=speed_sd, fmt="o", color="#2c7fb8", capsize=3, label="Measured")
        ax.plot([1, 128], [1, 128], "--", color="0.45", linewidth=1, label="Ideal")
        ax.set_yscale("log", base=2)
        ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
        ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
        ax.set_ylabel(r"Strong speedup $T_1/T_p$")
        core_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128])
        handles, labels = ax.get_legend_handles_labels()
        finish_metric_plot(fig, ax, figure_dir / "strong_scaling_speedup.pdf", "Strong scaling", handles, labels)

        fig, ax = plt.subplots(figsize=(3.45, 2.85))
        ax.errorbar(x, eff, yerr=eff_sd, fmt="o", color="#31a354", capsize=3, label="Measured")
        ax.axhline(1.0, linestyle="--", color="0.45", linewidth=1, label="Ideal")
        ax.set_ylim(bottom=0)
        ax.set_ylabel(r"Efficiency $(T_1/T_p)/p$")
        core_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128])
        handles, labels = ax.get_legend_handles_labels()
        finish_metric_plot(fig, ax, figure_dir / "strong_scaling_efficiency.pdf", "Strong scaling", handles, labels)


def plot_weak(figure_dir: Path, weak: list[dict[str, object]]) -> None:
    if not weak:
        raise SystemExit("mandatory weak-scaling data are missing")
    rows = select_weak_curve(weak)
    x = np.array([r["workers"] for r in rows], dtype=float)
    runtime = np.array([r["mean_s"] for r in rows], dtype=float)
    efficiency = np.array([r["weak_efficiency"] for r in rows], dtype=float)
    base = int(rows[0]["baseline_workers"])
    fig_runtime, ax_runtime = plt.subplots(figsize=(3.45, 2.85))
    fig_efficiency, ax_efficiency = plt.subplots(figsize=(3.45, 2.85))
    ax_runtime.plot(x, runtime, color="0.55", linewidth=0.9, zorder=1)
    ax_efficiency.plot(x, efficiency, color="0.55", linewidth=0.9, zorder=1)
    for family, label, marker, color in (
        ("weak_openmp", "OpenMP", "o", "#756bb1"),
        ("weak_mpi", "MPI", "s", "#d95f02"),
    ):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        family_x = np.array([row["workers"] for row in family_rows], dtype=float)
        ax_runtime.errorbar(
            family_x,
            [row["mean_s"] for row in family_rows],
            yerr=[row["std_s"] for row in family_rows],
            fmt=marker,
            capsize=3,
            color=color,
            label=label,
            zorder=2,
        )
        ax_efficiency.errorbar(
            family_x,
            [row["weak_efficiency"] for row in family_rows],
            yerr=[row["weak_efficiency_std"] for row in family_rows],
            fmt=marker,
            capsize=3,
            color=color,
            label=label,
            zorder=2,
        )
    ax_runtime.axhline(runtime[0], linestyle="--", color="0.45", linewidth=1, label="Ideal")
    ax_runtime.set_ylabel("End-to-end runtime (s)")
    ax_efficiency.axhline(1.0, linestyle="--", color="0.45", linewidth=1, label="Ideal")
    ax_efficiency.set_ylabel(rf"Relative efficiency $T_{{{base}}}/T_p$")
    ax_efficiency.set_ylim(bottom=0)
    ticks = [worker for worker in (1, 2, 4, 8, 16, 32, 64, 128) if worker in set(x.astype(int))]
    for ax in (ax_runtime, ax_efficiency):
        core_axis(ax, ticks)
    runtime_handles, runtime_labels = ax_runtime.get_legend_handles_labels()
    finish_metric_plot(
        fig_runtime,
        ax_runtime,
        figure_dir / "weak_scaling_runtime.pdf",
        "Weak scaling",
        runtime_handles,
        runtime_labels,
    )
    efficiency_handles, efficiency_labels = ax_efficiency.get_legend_handles_labels()
    finish_metric_plot(
        fig_efficiency,
        ax_efficiency,
        figure_dir / "weak_scaling_efficiency.pdf",
        "Weak scaling",
        efficiency_handles,
        efficiency_labels,
    )


def plot_backend(figure_dir: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit("mandatory sequential/CUDA size-sweep data are missing")
    fig_runtime, ax_runtime = plt.subplots(figsize=(3.45, 2.85))
    fig_cost, ax_cost = plt.subplots(figsize=(3.45, 2.85))
    time_limit_label = True
    for backend, label, marker, color in (
        ("seq", "Sequential", "o", "#1f77b4"),
        ("cuda", "CUDA", "s", "#d95f02"),
    ):
        subset = sorted((r for r in rows if r["backend"] == backend), key=lambda r: int(r["n"]))
        x = np.array([r["n"] for r in subset], dtype=float)
        runtime = np.array([r["mean_s"] for r in subset], dtype=float)
        runtime_sd = np.array([r["std_s"] for r in subset], dtype=float)
        cost = np.array([r["mean_cost"] for r in subset], dtype=float)
        cost_sd = np.array([r["std_cost"] for r in subset], dtype=float)
        ax_runtime.errorbar(x, runtime, yerr=runtime_sd, fmt=marker, capsize=3, label=label, color=color)
        ax_cost.errorbar(x, cost, yerr=cost_sd, fmt=marker, capsize=3, label=label, color=color)
        capped = [r for r in subset if r["termination"] == "time_limit"]
        if capped:
            ax_runtime.scatter(
                [r["n"] for r in capped],
                [r["mean_s"] for r in capped],
                marker="x",
                s=34,
                color="black",
                zorder=4,
                label="Time limit" if time_limit_label else None,
            )
            time_limit_label = False
    for ax in (ax_runtime, ax_cost):
        ax.set_xscale("log", base=2)
        ax.set_xticks([500, 1000, 2000, 4000, 8000, 16000, 32000, 64000])
        ax.set_xticklabels(["0.5k", "1k", "2k", "4k", "8k", "16k", "32k", "64k"], rotation=30)
        ax.set_xlabel("Customers n")
    ax_runtime.set_yscale("log")
    ax_runtime.set_ylabel("Runtime to stopping condition (s)")
    handles, labels = ax_runtime.get_legend_handles_labels()
    finish_metric_plot(
        fig_runtime,
        ax_runtime,
        figure_dir / "backend_runtime.pdf",
        "Sequential/CUDA comparison",
        handles,
        labels,
    )
    ax_cost.set_yscale("log")
    ax_cost.set_ylabel("Best cost (mean ± sample SD)")
    handles, labels = ax_cost.get_legend_handles_labels()
    finish_metric_plot(
        fig_cost,
        ax_cost,
        figure_dir / "backend_cost.pdf",
        "Sequential/CUDA comparison",
        handles,
        labels,
    )


def inspect_historic_logs(root: Path) -> list[str]:
    notes: list[str] = []
    for path in sorted((root / "results" / "slurm").glob("*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        job_match = re.search(r"^\[INFO\] job_id=(\d+)", text, re.MULTILINE)
        args_match = re.search(r"^\[INFO\] make_args=(.*)$", text, re.MULTILINE)
        job = job_match.group(1) if job_match else path.stem
        args = args_match.group(1) if args_match else ""
        err_path = path.with_suffix(".err")
        err = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
        if "SOLVE_MPI_FIXED_EPOCHS=500" in args and "strong_openmp" in args:
            reason = "cancelled/partial" if "CANCELLED" in err else "hit the 300 s solver limit before the requested 500 fixed epochs"
            notes.append(f"Slurm job {job}: obsolete strong-OpenMP 500-epoch configuration; {reason}; excluded.")
        if "SOLVE_CLIENTS=128000" in args and "no matching rows" in text:
            notes.append(f"Slurm job {job}: requested CUDA n=128000 but the manifest had no row; no measurement produced.")
    return notes


def write_validation(path: Path, root: Path, runs: list[Run]) -> None:
    fields = [
        "family", "backend", "name", "n", "K", "m", "seed", "run_id", "ranks", "threads",
        "workers", "status", "elapsed_s", "best_cost", "termination", "last_epoch",
        "solution_feasible", "cost_consistent", "recomputed_cost", "included_timing", "notes",
        "source_csv", "solution_file",
    ]
    rows = []
    for r in runs:
        rows.append(
            {
                **r.__dict__,
                "K": r.k,
                "workers": r.workers,
                "source_csv": r.csv_path.relative_to(root),
                "solution_file": r.solution_path.relative_to(root) if r.solution_path.is_relative_to(root) else r.solution_path,
            }
        )
    write_csv(path, rows, fields)


def write_report(
    path: Path,
    root: Path,
    runs: list[Run],
    discovery_notes: list[str],
    historic_notes: list[str],
    scaling: list[dict[str, object]],
    weak: list[dict[str, object]],
) -> None:
    included = [r for r in runs if r.included_timing]
    cost_issues = [r for r in runs if r.solution_feasible and not r.cost_consistent]
    invalid = [r for r in runs if not r.included_timing]
    omp_workers = sorted({r.workers for r in included if r.family == "strong_openmp"})
    mpi_workers = sorted({r.workers for r in included if r.family == "strong_mpi"})
    weak_workers = sorted({r.workers for r in included if r.family.startswith("weak_")})
    weak_omp_workers = sorted({r.workers for r in included if r.family == "weak_openmp"})
    weak_mpi_workers = sorted({r.workers for r in included if r.family == "weak_mpi"})
    missing_omp = [x for x in EXPECTED_STRONG_OMP if x not in omp_workers]
    missing_mpi = [x for x in EXPECTED_STRONG_MPI if x not in mpi_workers]
    missing_weak = [x for x in EXPECTED_WEAK if x not in weak_workers]
    lines = [
        "# Current result validation",
        "",
        "Generated by `python3 docs/paper/scripts/analyze_results.py`. Raw files under `results/` were not modified.",
        "",
        "## Coverage",
        "",
        f"- Discovered {len(runs)} current CSV rows; {len(included)} are valid for timing aggregation.",
        f"- Strong OpenMP workers present: {omp_workers}; missing planned points: {missing_omp}.",
        f"- Strong MPI workers present: {mpi_workers}; missing planned points: {missing_mpi}.",
        f"- Weak-scaling workers present: {weak_workers}; missing planned points: {missing_weak}.",
        f"- Weak OpenMP workers present: {weak_omp_workers}; weak MPI workers present: {weak_mpi_workers}.",
        "- The combined weak curve uses OpenMP through 32 workers and MPI at 64/128; the MPI 32-worker point is retained as a consistency check.",
        "- MPI `max_rss_gb` is the RSS of the `srun` launcher and is not used as rank memory.",
        "",
        "## Validation and exclusions",
        "",
    ]
    if invalid:
        for r in invalid:
            lines.append(f"- `{r.csv_path.relative_to(root)}` run {r.run_id}: excluded from timing ({r.notes or r.termination}).")
    else:
        lines.append("- No current CSV row was excluded from timing aggregation.")
    for note in discovery_notes + historic_notes:
        lines.append(f"- {note}")
    if cost_issues:
        lines.append("- The following runs have feasible routes but a printed cost inconsistent with the route cost; timing is retained, cost is excluded:")
        for r in cost_issues:
            lines.append(f"  - `{r.solution_path.relative_to(root)}`: {r.notes}")
    else:
        lines.append("- Every current solution is feasible and its route cost matches the CSV/printed cost within tolerance.")
    lines.extend(
        [
            "",
            "## Statistics and formulas",
            "",
            "- Runtime center: arithmetic mean over three independent seeds; median, min, max, sample SD (`n-1`), and CV are also emitted.",
            "- Strong speedup: `S_p = T_1 / T_p`; efficiency: `E_p = S_p / p`, with total workers `p = ranks * threads`.",
            "- MPI-only strong metrics use the 32-worker point as their local baseline; the combined curve uses the 1-thread OpenMP point.",
            "- Weak workload is 32 ants per worker at fixed `n=8000` and 100 epochs; weak efficiency is `E_p^weak = T_1 / T_p`.",
            "- Error bars are sample SD for runtime/cost and first-order relative-error propagation for ratios.",
            "",
            "## Stopping conditions and comparability",
            "",
            "- Strong/weak scaling rows completed 100 fixed epochs with stagnation disabled and a 300 s safety limit.",
            "- Sequential and CUDA size sweeps stop on stagnation or the 300 s solver limit; their ant populations differ. Their plot is descriptive and is not used to claim a like-for-like speedup.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--data-dir", type=Path, default=Path("docs/paper/generated"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/paper/figures"))
    args = parser.parse_args()
    root = args.root.resolve()
    data_dir = root / args.data_dir
    figure_dir = root / args.figure_dir
    summary_dir = data_dir / "summaries"
    scaling_figure_dir = figure_dir / "scaling"
    comparison_figure_dir = figure_dir / "comparison"

    runs, discovery_notes = discover_runs(root)
    validate_runs(root, runs)
    backend, scaling, combined, weak = build_summaries(runs)
    historic_notes = inspect_historic_logs(root)

    common_stats = ["samples", "mean_s", "median_s", "min_s", "max_s", "std_s", "cv_pct", "cost_samples", "mean_cost", "std_cost"]
    for legacy_name in (
        "run_validation.csv",
        "backend_size_summary.csv",
        "strong_scaling_summary.csv",
        "strong_combined_summary.csv",
        "weak_scaling_summary.csv",
    ):
        (data_dir / legacy_name).unlink(missing_ok=True)
    write_validation(summary_dir / "run_validation.csv", root, runs)
    write_csv(summary_dir / "backend_size_summary.csv", backend, ["backend", "n", "K", "m", "termination", *common_stats])
    write_csv(summary_dir / "strong_scaling_summary.csv", scaling, ["family", "n", "K", "m", "ranks", "threads", "workers", "baseline_workers", *common_stats, "speedup", "speedup_std", "efficiency", "efficiency_std"])
    write_csv(summary_dir / "strong_combined_summary.csv", combined, ["source_family", "n", "K", "m", "ranks", "threads", "workers", *common_stats, "speedup", "speedup_std", "efficiency", "efficiency_std"])
    write_csv(summary_dir / "weak_scaling_summary.csv", weak, ["family", "n", "K", "m", "ranks", "threads", "workers", "ants_per_worker", "baseline_workers", *common_stats, "weak_efficiency", "weak_efficiency_std"])
    write_report(data_dir / "analysis.md", root, runs, discovery_notes, historic_notes, scaling, weak)

    setup_plot_style()
    for legacy_name in (
        "strong_scaling.pdf",
        "weak_scaling.pdf",
        "backend_size_comparison.pdf",
        "strong_scaling_runtime.pdf",
        "strong_scaling_speedup.pdf",
        "strong_scaling_efficiency.pdf",
        "weak_scaling_runtime.pdf",
        "weak_scaling_efficiency.pdf",
        "backend_runtime.pdf",
        "backend_cost.pdf",
    ):
        (figure_dir / legacy_name).unlink(missing_ok=True)
    plot_strong(scaling_figure_dir, scaling, combined)
    plot_weak(scaling_figure_dir, weak)
    plot_backend(comparison_figure_dir, backend)

    print(f"Validated {len(runs)} current rows; {sum(r.included_timing for r in runs)} included for timing.")
    print(f"Wrote derived data to {display_path(data_dir, root)}")
    print(f"Wrote figures to {display_path(figure_dir, root)}")
    for note in discovery_notes + historic_notes:
        print(f"WARNING: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
