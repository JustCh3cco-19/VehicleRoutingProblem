#!/usr/bin/env python3
"""Compare C ACO (sequential and/or MPI) against PyVRP on aligned instances.

The script:
1) generates deterministic synthetic CVRP instances (same layout logic as C tests)
2) runs selected C solver mode(s): seq / mpi / both
3) runs PyVRP once per scenario
4) writes one CSV row per executed C run with objective/time/gap
5) computes strong/weak scaling metrics for MPI rows (when possible)
6) writes a compact markdown report.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Scenario:
    n: int
    K: int
    m: int
    T: int
    solver_seed: int
    instance_seed: int
    layout_id: int


@dataclass
class Point:
    x: float
    y: float


@dataclass
class RunConfig:
    solver_mode: str  # seq | mpi
    mpi_ranks: int
    omp_threads: int


def ensure_vrp_python() -> None:
    """Re-exec inside VRP/bin/python when available for stable PyVRP imports."""
    vrp_python = Path("VRP/bin/python")
    if not vrp_python.exists():
        return

    current = Path(sys.executable).absolute()
    target = vrp_python.absolute()
    if current != target:
        os.execv(str(target), [str(target)] + sys.argv)


def build_scenarios() -> list[Scenario]:
    levels = [
        500,
        1000,
        2000,
        4000,
        8000,
        12000,
        16000,
        24000,
        32000,
        40000,
        48000,
        56000,
        64000,
        72000,
        80000,
        90000,
        100000,
    ]
    out: list[Scenario] = []
    for idx, n in enumerate(levels):
        if n <= 2000:
            m, t = 32, 20
        elif n <= 8000:
            m, t = 16, 10
        elif n <= 16000:
            m, t = 8, 6
        elif n <= 32000:
            m, t = 4, 4
        else:
            m, t = 3, 3
        out.append(
            Scenario(
                n=n,
                K=max(8, min(128, n // 1000)),
                m=m,
                T=t,
                solver_seed=9000 + idx,
                instance_seed=19000 + idx,
                layout_id=idx % 4,
            )
        )
    return out


def parse_ranks_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty mpi ranks list")
    ranks: list[int] = []
    for p in parts:
        val = int(p)
        if val < 1:
            raise ValueError(f"invalid mpi rank value: {val}")
        ranks.append(val)
    # Keep stable order, remove duplicates.
    seen: set[int] = set()
    uniq: list[int] = []
    for r in ranks:
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def bytes_to_gib(x: int) -> float:
    return x / (1024.0**3)


def read_available_memory_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return 0
    out = 0
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
        if line.startswith("MemTotal:") and out == 0:
            out = int(line.split()[1]) * 1024
    return out


def estimate_memory_bytes(n: int) -> int:
    side = n + 1
    return int(side * side * 8 * 1.6)


def lcg_next(state: int) -> int:
    return ((state * 1664525) + 1013904223) & 0xFFFFFFFF


def rand01_lcg(state: int) -> tuple[float, int]:
    nxt = lcg_next(state)
    return nxt / 0xFFFFFFFF, nxt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_points(n: int, seed: int, layout: int) -> list[Point]:
    scale = 1000.0
    st = seed if seed != 0 else 1

    pts = [Point(0.0, 0.0) for _ in range(n + 1)]
    pts[0] = Point(0.5 * scale, 0.5 * scale)

    for i in range(1, n + 1):
        r1, st = rand01_lcg(st)
        r2, st = rand01_lcg(st)

        if layout == 1:
            cluster = i % 3
            cx = [0.2, 0.8, 0.5]
            cy = [0.2, 0.2, 0.8]
            x = clamp((cx[cluster] + 0.25 * (r1 - 0.5)) * scale, 0.0, scale)
            y = clamp((cy[cluster] + 0.25 * (r2 - 0.5)) * scale, 0.0, scale)
        elif layout == 2:
            left = (i % 2) == 0
            xbase = 0.04 if left else 0.84
            x = (xbase + 0.12 * r1) * scale
            y = r2 * scale
        elif layout == 3:
            x = (i / (n + 1)) * scale
            y = (0.35 if (i % 2) == 0 else 0.65) * scale + 0.03 * (r2 - 0.5) * scale
        else:
            x = r1 * scale
            y = r2 * scale

        pts[i] = Point(x, y)

    return pts


def write_vrplib_file(path: Path, sc: Scenario) -> None:
    pts = generate_points(sc.n, sc.instance_seed, sc.layout_id)
    capacity = (sc.n + sc.K - 1) // sc.K

    lines = [
        f"NAME : compare_n{sc.n}",
        "TYPE : CVRP",
        "COMMENT : synthetic deterministic instance aligned with C runner",
        f"DIMENSION : {sc.n + 1}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {capacity}",
        f"VEHICLES : {sc.K}",
        "NODE_COORD_SECTION",
    ]
    for i, p in enumerate(pts, start=1):
        lines.append(f"{i} {int(round(p.x))} {int(round(p.y))}")

    lines.append("DEMAND_SECTION")
    lines.append("1 0")
    for i in range(2, sc.n + 2):
        lines.append(f"{i} 1")

    lines.extend(["DEPOT_SECTION", "1", "-1", "EOF", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_helper(helper_bin: Path, auto_build: bool) -> None:
    if helper_bin.exists():
        return
    if not auto_build:
        raise RuntimeError(f"missing helper binary: {helper_bin}")
    cmd = ["make", str(helper_bin)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"failed to build {helper_bin}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def parse_key_values(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in stdout.splitlines():
        line = raw.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def run_c_solver_seq(
    helper_bin: Path,
    sc: Scenario,
    alpha: float,
    beta: float,
    rho: float,
    tau0: float,
    q_value: float,
    m_passed: int,
) -> tuple[float, float]:
    cmd = [
        str(helper_bin),
        str(sc.n),
        str(sc.K),
        str(m_passed),
        str(sc.T),
        str(sc.solver_seed),
        str(sc.instance_seed),
        str(sc.layout_id),
        str(alpha),
        str(beta),
        str(rho),
        str(tau0),
        str(q_value),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            f"seq helper failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    kv = parse_key_values(proc.stdout)
    if "best_cost" not in kv:
        raise RuntimeError(f"missing best_cost in seq output: {proc.stdout!r}")
    return float(kv["best_cost"]), elapsed


def run_c_solver_mpi(
    helper_bin: Path,
    mpi_ranks: int,
    omp_threads: int,
    sc: Scenario,
    alpha: float,
    beta: float,
    rho: float,
    tau0: float,
    q_value: float,
    m_passed: int,
) -> tuple[float, float]:
    cmd = [
        "mpirun",
        "-np",
        str(mpi_ranks),
        str(helper_bin),
        str(sc.n),
        str(sc.K),
        str(m_passed),
        str(sc.T),
        str(sc.solver_seed),
        str(sc.instance_seed),
        str(sc.layout_id),
        str(alpha),
        str(beta),
        str(rho),
        str(tau0),
        str(q_value),
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(max(1, omp_threads))
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, env=env
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mpi helper failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    kv = parse_key_values(proc.stdout)
    if "best_cost" not in kv:
        raise RuntimeError(f"missing best_cost in mpi output: {proc.stdout!r}")
    if "elapsed_s" not in kv:
        raise RuntimeError(f"missing elapsed_s in mpi output: {proc.stdout!r}")
    return float(kv["best_cost"]), float(kv["elapsed_s"])


def extract_pyvrp_cost(result: Any) -> float:
    for attr in ("cost", "best_cost", "objective"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            return float(val() if callable(val) else val)

    for meth in ("best", "best_solution", "solution"):
        if hasattr(result, meth):
            obj = getattr(result, meth)
            obj = obj() if callable(obj) else obj
            if hasattr(obj, "cost"):
                val = getattr(obj, "cost")
                return float(val() if callable(val) else val)

    raise RuntimeError("unable to extract PyVRP objective")


def run_pyvrp_solver(
    vrp_path: Path, seed: int, iterations: int, timeout_s: int
) -> tuple[float, float]:
    import pyvrp

    instance = pyvrp.read(str(vrp_path), round_func="none")
    kwargs = {"seed": seed}
    sig = inspect.signature(pyvrp.solve)
    if "stop" in sig.parameters:
        if timeout_s > 0:
            kwargs["stop"] = pyvrp.stop.MaxRuntime(timeout_s)
        else:
            kwargs["stop"] = pyvrp.stop.MaxIterations(max(1, iterations))

    start = time.perf_counter()
    result = pyvrp.solve(instance, **kwargs)
    elapsed = time.perf_counter() - start
    return extract_pyvrp_cost(result), elapsed


def compute_scaling_metrics(rows: list[dict[str, str]]) -> None:
    """Fill strong/weak scaling columns in-place for MPI rows with status=ok."""
    mpi_ok_rows = [
        r
        for r in rows
        if r["solver_mode"] == "mpi" and r["status"] == "ok" and r["c_elapsed_s"]
    ]
    if not mpi_ok_rows:
        return

    # Strong scaling: fixed n, varying mpi_ranks.
    by_n: dict[int, list[dict[str, str]]] = {}
    for r in mpi_ok_rows:
        n = int(r["n"])
        by_n.setdefault(n, []).append(r)

    for _, group in by_n.items():
        group.sort(key=lambda r: int(r["mpi_ranks"]))
        p0 = int(group[0]["mpi_ranks"])
        t0 = float(group[0]["c_elapsed_s"])
        for r in group:
            p = int(r["mpi_ranks"])
            tp = float(r["c_elapsed_s"])
            if tp <= 0.0:
                continue
            speedup = t0 / tp
            eff = speedup / (p / p0)
            r["strong_speedup"] = f"{speedup:.6f}"
            r["strong_efficiency"] = f"{eff:.6f}"

    # Weak scaling: approximately fixed n/p.
    by_npr: dict[int, list[dict[str, str]]] = {}
    for r in mpi_ok_rows:
        n = int(r["n"])
        p = int(r["mpi_ranks"])
        npr = int(round(n / p))
        r["n_per_rank"] = str(npr)
        by_npr.setdefault(npr, []).append(r)

    for _, group in by_npr.items():
        group.sort(key=lambda r: int(r["mpi_ranks"]))
        t_ref = float(group[0]["c_elapsed_s"])
        for r in group:
            tp = float(r["c_elapsed_s"])
            if tp <= 0.0:
                continue
            weak_eff = t_ref / tp
            r["weak_efficiency"] = f"{weak_eff:.6f}"


def write_scaling_report(rows: list[dict[str, str]], report_path: Path) -> None:
    mpi_rows = [r for r in rows if r["solver_mode"] == "mpi"]
    mpi_ok = [r for r in mpi_rows if r["status"] == "ok"]
    seq_ok = [r for r in rows if r["solver_mode"] == "seq" and r["status"] == "ok"]

    strong_rows = [r for r in mpi_ok if r["strong_speedup"]]
    weak_rows = [r for r in mpi_ok if r["weak_efficiency"]]

    lines: list[str] = []
    lines.append("# C vs PyVRP Compare Report")
    lines.append("")
    lines.append("## Overview")
    lines.append(f"- Total rows: {len(rows)}")
    lines.append(f"- Seq ok rows: {len(seq_ok)}")
    lines.append(f"- MPI ok rows: {len(mpi_ok)}")
    lines.append("")
    lines.append("## Scaling Availability")
    if strong_rows:
        lines.append(f"- Strong scaling rows: {len(strong_rows)}")
    else:
        lines.append("- Strong scaling: not computable (missing same n at multiple mpi_ranks)")
    if weak_rows:
        lines.append(f"- Weak scaling rows: {len(weak_rows)}")
    else:
        lines.append("- Weak scaling: not computable (missing repeated n_per_rank at multiple mpi_ranks)")
    lines.append("")

    if strong_rows:
        max_speedup = max(float(r["strong_speedup"]) for r in strong_rows)
        best = max(strong_rows, key=lambda r: float(r["strong_speedup"]))
        lines.append("## Best Strong Scaling Point")
        lines.append(
            f"- n={best['n']}, mpi_ranks={best['mpi_ranks']}, speedup={max_speedup:.3f}, efficiency={float(best['strong_efficiency']):.3f}"
        )
        lines.append("")

    if weak_rows:
        best_weak = max(weak_rows, key=lambda r: float(r["weak_efficiency"]))
        lines.append("## Best Weak Scaling Point")
        lines.append(
            f"- n={best_weak['n']}, mpi_ranks={best_weak['mpi_ranks']}, n_per_rank={best_weak['n_per_rank']}, weak_efficiency={float(best_weak['weak_efficiency']):.3f}"
        )
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare C ACO vs PyVRP")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/c_vs_pyvrp_compare.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--scaling-report",
        type=Path,
        default=Path("results/c_vs_pyvrp_scaling_report.md"),
        help="Output markdown report with scaling metrics",
    )
    parser.add_argument(
        "--mode",
        choices=("seq", "mpi", "both"),
        default="both",
        help="C solver mode to run",
    )
    parser.add_argument(
        "--helper-bin-seq",
        type=Path,
        default=Path("tests/c_compare_case.out"),
        help="Sequential helper binary",
    )
    parser.add_argument(
        "--helper-bin-mpi",
        type=Path,
        default=Path("tests/c_compare_case_mpi.out"),
        help="MPI helper binary",
    )
    parser.add_argument(
        "--no-build-helpers",
        action="store_true",
        help="Do not build helper binaries automatically",
    )
    parser.add_argument("--mpi-ranks-list", default="1,2,4")
    parser.add_argument("--omp-threads", type=int, default=1)
    parser.add_argument("--memory-utilization", type=float, default=0.70)
    parser.add_argument("--c-max-n", type=int, default=100000)
    parser.add_argument("--enforce-c-max-n", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--single-n", type=int, default=0)
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="PyVRP timeout in seconds; 0 uses MaxIterations(T)",
    )
    parser.add_argument("--fixed-ants", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--tau0", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    ensure_vrp_python()
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.scaling_report.parent.mkdir(parents=True, exist_ok=True)

    if args.memory_utilization <= 0.0 or args.memory_utilization > 1.0:
        raise SystemExit("--memory-utilization must be in (0, 1]")
    if args.c_max_n <= 0:
        raise SystemExit("--c-max-n must be > 0")
    if args.omp_threads < 1:
        raise SystemExit("--omp-threads must be >= 1")

    mpi_ranks_list = parse_ranks_list(args.mpi_ranks_list)
    run_configs: list[RunConfig] = []
    if args.mode in ("seq", "both"):
        run_configs.append(RunConfig(solver_mode="seq", mpi_ranks=1, omp_threads=1))
    if args.mode in ("mpi", "both"):
        for p in mpi_ranks_list:
            run_configs.append(
                RunConfig(solver_mode="mpi", mpi_ranks=p, omp_threads=args.omp_threads)
            )

    if args.mode in ("seq", "both"):
        ensure_helper(args.helper_bin_seq, auto_build=not args.no_build_helpers)
    if args.mode in ("mpi", "both"):
        ensure_helper(args.helper_bin_mpi, auto_build=not args.no_build_helpers)

    scenarios = build_scenarios()
    if args.single_n > 0:
        scenarios = [sc for sc in scenarios if sc.n == args.single_n]
        if not scenarios:
            raise SystemExit(f"--single-n={args.single_n} not found in scenario ladder")

    available = read_available_memory_bytes()
    threshold = int(available * args.memory_utilization) if available > 0 else 0

    print("========================================================================")
    print("C vs PyVRP Compare")
    print("========================================================================")
    print(f"[INFO] mode                 : {args.mode}")
    print(f"[INFO] run_configs          : {[(rc.solver_mode, rc.mpi_ranks, rc.omp_threads) for rc in run_configs]}")
    print(f"[INFO] csv_path             : {args.csv}")
    print(f"[INFO] scaling_report       : {args.scaling_report}")
    print(f"[INFO] memory_utilization   : {args.memory_utilization:.2f}")
    print(f"[INFO] c_max_n              : {args.c_max_n}")
    print(f"[INFO] enforce_c_max_n      : {'yes' if args.enforce_c_max_n else 'no'}")
    print(f"[INFO] force                : {'yes' if args.force else 'no'}")
    print(f"[INFO] available_mem_gib    : {bytes_to_gib(available):.2f}")
    print(f"[INFO] threshold_gib        : {bytes_to_gib(threshold):.2f}")
    print(f"[INFO] ants_mode            : {'fixed' if args.fixed_ants else 'auto(m=0)'}")
    print(
        f"[INFO] aco_params           : alpha={args.alpha} beta={args.beta} rho={args.rho} tau0={args.tau0} Q={args.q}"
    )
    print("")

    ok_count = 0
    skipped_count = 0
    failed_count = 0
    total_runs = len(scenarios) * len(run_configs)

    rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="c_vs_pyvrp_") as td:
        tmp_dir = Path(td)
        run_index = 0

        for sc in scenarios:
            m_passed = sc.m if args.fixed_ants else 0
            est_mem = estimate_memory_bytes(sc.n)
            est_gib = bytes_to_gib(est_mem)

            # Compute PyVRP once per scenario to reuse across run configs.
            py_cost = math.nan
            py_elapsed = math.nan
            py_error = ""

            if args.enforce_c_max_n and sc.n > args.c_max_n:
                py_error = f"n={sc.n} > c_max_n={args.c_max_n}"
            elif (not args.force) and threshold > 0 and est_mem > threshold:
                py_error = f"estimated memory {est_gib:.4f} GiB > threshold {bytes_to_gib(threshold):.4f} GiB"
            else:
                vrp_path = tmp_dir / f"compare_n{sc.n}.vrp"
                write_vrplib_file(vrp_path, sc)
                try:
                    py_cost, py_elapsed = run_pyvrp_solver(
                        vrp_path, sc.solver_seed, sc.T, args.timeout
                    )
                except Exception as exc:
                    py_error = f"{type(exc).__name__}: {exc}"

            for rc in run_configs:
                run_index += 1
                print(
                    f"[RUN  {run_index:02d}/{total_runs:02d}] mode={rc.solver_mode} p={rc.mpi_ranks} omp={rc.omp_threads} n={sc.n:<6} K={sc.K:<4} m={m_passed:<3} T={sc.T:<3} est_mem={est_gib:>6.2f} GiB"
                )
                print(
                    f"  [INPUT] solver/instance: n={sc.n} K={sc.K} m={m_passed} T={sc.T} solver_seed={sc.solver_seed} instance_seed={sc.instance_seed} layout={sc.layout_id}"
                )

                row: dict[str, str] = {
                    "solver_mode": rc.solver_mode,
                    "mpi_ranks": str(rc.mpi_ranks),
                    "omp_threads": str(rc.omp_threads),
                    "n": str(sc.n),
                    "K": str(sc.K),
                    "m_passed": str(m_passed),
                    "T": str(sc.T),
                    "solver_seed": str(sc.solver_seed),
                    "instance_seed": str(sc.instance_seed),
                    "layout_id": str(sc.layout_id),
                    "estimated_mem_gib": f"{est_gib:.4f}",
                    "status": "",
                    "c_elapsed_s": "",
                    "c_cost": "",
                    "pyvrp_elapsed_s": f"{py_elapsed:.6f}" if not math.isnan(py_elapsed) else "",
                    "pyvrp_cost": f"{py_cost:.6f}" if not math.isnan(py_cost) else "",
                    "gap_pct_vs_pyvrp": "",
                    "n_per_rank": "",
                    "strong_speedup": "",
                    "strong_efficiency": "",
                    "weak_efficiency": "",
                    "error": "",
                }

                # Scenario-level skips.
                if args.enforce_c_max_n and sc.n > args.c_max_n:
                    skipped_count += 1
                    row["status"] = "skipped_c_max_n"
                    row["error"] = f"n={sc.n} > c_max_n={args.c_max_n}"
                    print(f"  [SKIP] n={sc.n} supera c-max-n={args.c_max_n}")
                    rows.append(row)
                    continue

                if (not args.force) and threshold > 0 and est_mem > threshold:
                    skipped_count += 1
                    row["status"] = "skipped_memory"
                    row["error"] = (
                        f"estimated memory {est_gib:.4f} GiB > threshold {bytes_to_gib(threshold):.4f} GiB"
                    )
                    print(
                        f"  [SKIP] memoria stimata {est_gib:.2f} GiB > soglia {bytes_to_gib(threshold):.2f} GiB"
                    )
                    rows.append(row)
                    continue

                # Run selected C solver.
                try:
                    if rc.solver_mode == "seq":
                        c_cost, c_elapsed = run_c_solver_seq(
                            args.helper_bin_seq,
                            sc,
                            alpha=args.alpha,
                            beta=args.beta,
                            rho=args.rho,
                            tau0=args.tau0,
                            q_value=args.q,
                            m_passed=m_passed,
                        )
                    else:
                        c_cost, c_elapsed = run_c_solver_mpi(
                            args.helper_bin_mpi,
                            rc.mpi_ranks,
                            rc.omp_threads,
                            sc,
                            alpha=args.alpha,
                            beta=args.beta,
                            rho=args.rho,
                            tau0=args.tau0,
                            q_value=args.q,
                            m_passed=m_passed,
                        )
                except Exception as exc:
                    failed_count += 1
                    row["status"] = "failed_c"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  [FAIL] C solver failed: {type(exc).__name__}: {exc}")
                    rows.append(row)
                    continue

                row["c_cost"] = f"{c_cost:.6f}"
                row["c_elapsed_s"] = f"{c_elapsed:.6f}"

                if math.isnan(py_cost):
                    # C result is valid, PyVRP missing/failed for this scenario.
                    ok_count += 1
                    row["status"] = "ok_c_only"
                    row["error"] = f"pyvrp_unavailable: {py_error}"
                    print(
                        f"  [OK] c_cost={c_cost:.3f} ({c_elapsed:.2f}s)  pyvrp=unavailable ({py_error})"
                    )
                else:
                    gap_pct = 100.0 * (c_cost - py_cost) / py_cost if py_cost > 0 else math.nan
                    ok_count += 1
                    row["status"] = "ok"
                    row["gap_pct_vs_pyvrp"] = f"{gap_pct:.6f}"
                    print(
                        f"  [OK] c_cost={c_cost:.3f} ({c_elapsed:.2f}s)  pyvrp_cost={py_cost:.3f} ({py_elapsed:.2f}s)  gap={gap_pct:.2f}%"
                    )

                rows.append(row)

    compute_scaling_metrics(rows)

    with args.csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(
            fcsv,
            fieldnames=[
                "solver_mode",
                "mpi_ranks",
                "omp_threads",
                "n",
                "K",
                "m_passed",
                "T",
                "solver_seed",
                "instance_seed",
                "layout_id",
                "estimated_mem_gib",
                "status",
                "c_elapsed_s",
                "c_cost",
                "pyvrp_elapsed_s",
                "pyvrp_cost",
                "gap_pct_vs_pyvrp",
                "n_per_rank",
                "strong_speedup",
                "strong_efficiency",
                "weak_efficiency",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_scaling_report(rows, args.scaling_report)

    print("")
    print("========================================================================")
    print("Summary")
    print("========================================================================")
    print(f"[SUMMARY] runs_total        : {total_runs}")
    print(f"[SUMMARY] ok                : {ok_count}")
    print(f"[SUMMARY] failed            : {failed_count}")
    print(f"[SUMMARY] skipped           : {skipped_count}")
    print(f"[DONE] wrote {args.csv}")
    print(f"[DONE] wrote {args.scaling_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
