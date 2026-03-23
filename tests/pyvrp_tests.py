#!/usr/bin/env python3
#
# Progressive PyVRP scaling runner.
#
# This script executes a deterministic set of synthetic CVRP scenarios with
# growing customer counts (up to 40k), solves each scenario with PyVRP, and
# writes results to CSV. It is intentionally self-contained: it generates the
# instances on the fly in VRPLIB format, so no external instance files are
# required.
#
# Design goals:
# - reproducibility: deterministic pseudo-random instance generation via fixed
#   seeds and a simple LCG;
# - portability: automatic re-exec with the project venv interpreter
#   (VRP/bin/python);
# - safety: memory-aware skips to avoid unstable runs on large n;
# - observability: one CSV row per scenario with status/time/cost/error.
#
import argparse
import csv
import inspect
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Scenario:
    # Problem size parameters and deterministic seeds used for one run level.
    n: int
    K: int
    m: int
    T: int
    solver_seed: int
    instance_seed: int
    layout_id: int


@dataclass
class Point:
    # Euclidean coordinate of one node in the synthetic plane.
    x: float
    y: float


def bytes_to_gib(x: int) -> float:
    # Utility to print memory values in a human-readable unit.
    return x / (1024.0 ** 3)


def ensure_vrp_python():
    """
    Ensure execution happens inside the local venv at VRP/bin/python.

    Why re-exec:
    - users may launch this script with system Python from shell/IDE;
    - PyVRP is expected in the project venv;
    - re-exec guarantees imports and runtime behavior are consistent.
    """
    vrp_python = Path("VRP/bin/python")
    if not vrp_python.exists():
        raise RuntimeError("missing venv interpreter: VRP/bin/python")

    current = Path(sys.executable).absolute()
    target = vrp_python.absolute()

    if current != target:
        os.execv(str(target), [str(target)] + sys.argv)


def read_available_memory_bytes() -> int:
    """
    Read available memory from /proc/meminfo.

    Preference order:
    1) MemAvailable (best estimate for allocatable RAM),
    2) MemTotal fallback if MemAvailable is not present.
    Returns 0 when information is unavailable.
    """
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        with meminfo.open("r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    return 0


def estimate_pyvrp_memory_bytes(n: int) -> int:
    """
    Conservative memory estimate used only for skip decisions.

    We approximate at least one dense double matrix (n+1)^2 plus 60% overhead
    for internal structures, neighborhoods, population state, and runtime
    allocations. This is not an exact PyVRP memory model, but a practical
    guardrail for large-n runs.
    """
    # Conservative estimate for at least one dense (n+1)^2 matrix in doubles
    # plus overhead for neighborhoods/population/aux structures.
    side = n + 1
    one_dense = side * side * 8
    return int(one_dense * 1.6)


def build_scenarios() -> List[Scenario]:
    """
    Build the progressive scenario ladder.

    n increases gradually to stress scaling behavior. The (m, T) values are
    reduced as n grows to keep runtime bounded. K grows with n (capped) to
    avoid unrealistically overloaded routes in very large instances.
    """
    levels = [500, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000]
    out: List[Scenario] = []

    for idx, n in enumerate(levels):
        if n <= 2000:
            m, t = 64, 40
        elif n <= 8000:
            m, t = 32, 20
        elif n <= 16000:
            m, t = 16, 12
        elif n <= 32000:
            m, t = 8, 8
        else:
            m, t = 4, 6

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


def lcg_next(state: int) -> int:
    # 32-bit linear congruential generator step (deterministic).
    return ((state * 1664525) + 1013904223) & 0xFFFFFFFF


def rand01_lcg(state: int):
    # Map LCG output to [0, 1].
    nxt = lcg_next(state)
    return nxt / 0xFFFFFFFF, nxt


def clamp(x: float, lo: float, hi: float) -> float:
    # Clamp coordinate values to remain inside the synthetic square.
    return max(lo, min(hi, x))


def generate_points(n: int, seed: int, layout: int) -> List[Point]:
    """
    Generate depot + customer coordinates with deterministic geometry patterns.

    layout meanings (mirrors tests/tests.c):
    - 0: uniform random in a square;
    - 1: clustered customers around 3 centers;
    - 2: border-heavy left/right stripes;
    - 3: parity line pattern (alternating bands).

    Node 0 is always the depot at the center.
    """
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


def write_vrplib_file(path: Path, sc: Scenario):
    """
    Materialize one synthetic scenario as a VRPLIB CVRP instance.

    Conventions:
    - VRPLIB node ids are 1-based, while internal generation is 0-based.
      We therefore emit depot as node 1 and customers as 2..n+1.
    - Unit demand per customer and capacity=n ensure feasibility with K routes
      (total demand = n, each route can carry up to n, so capacity itself is
      not the bottleneck in these stress tests).
    - EDGE_WEIGHT_TYPE=EUC_2D lets PyVRP derive distances from coordinates.
    """
    pts = generate_points(sc.n, sc.instance_seed, sc.layout_id)

    lines = [
        f"NAME : scaling_n{sc.n}",
        "TYPE : CVRP",
        "COMMENT : synthetic deterministic instance generated from tests/tests.c layout",
        f"DIMENSION : {sc.n + 1}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        f"CAPACITY : {sc.n}",
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


def extract_pyvrp_cost(result) -> float:
    """
    Extract objective value from PyVRP result objects across API variants.

    Different PyVRP versions expose the cost through different attributes/methods
    (e.g. cost, best_cost, objective, best_solution().cost()).
    This helper keeps the runner robust to those minor API differences.
    """
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

    raise RuntimeError("unable to extract cost from pyvrp result object")


def solve_with_pyvrp(vrp_path: Path, seed: int, timeout_s: int, iterations: int):
    """
    Solve one VRPLIB instance with PyVRP.

    Stopping policy:
    - if timeout_s > 0: stop by wall-clock via MaxRuntime;
    - else: stop by iteration budget via MaxIterations(sc.T).
    """
    import pyvrp  # imported only after venv switch

    instance = pyvrp.read(str(vrp_path), round_func="round")
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
    cost = extract_pyvrp_cost(result)
    return cost, elapsed


def main():
    # CLI is intentionally compact: this script has one primary mode (PyVRP).
    parser = argparse.ArgumentParser(
        description="Progressive scaling tests (PyVRP-only) up to n=40000, using VRP uv venv."
    )
    parser.add_argument("--memory-utilization", type=float, default=0.70)
    parser.add_argument("--timeout", type=int, default=0, help="PyVRP per-scenario timeout in seconds")
    parser.add_argument("--csv", default="results/scaling_progressive_pyvrp.csv")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pyvrp-max-n", type=int, default=40000)
    args = parser.parse_args()

    # Force venv consistency before importing/using PyVRP internals.
    ensure_vrp_python()

    scenarios = build_scenarios()
    available = read_available_memory_bytes()
    threshold = int(available * args.memory_utilization) if available > 0 else 0

    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Runtime context summary helps debugging runs from IDE/CI.
    print(f"[info] python={sys.executable}")
    print(f"[info] available_mem_gib={bytes_to_gib(available):.2f}")
    print(f"[info] threshold_gib={bytes_to_gib(threshold):.2f} (utilization={args.memory_utilization:.2f})")

    with tempfile.TemporaryDirectory(prefix="vrp_scaling_pyvrp_") as td:
        case_dir = Path(td)

        with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(
                fcsv,
                fieldnames=[
                    "mode",
                    "n",
                    "K",
                    "m",
                    "T",
                    "estimated_mem_gib",
                    "status",
                    "elapsed_s",
                    "pyvrp_cost",
                    "error",
                ],
            )
            writer.writeheader()

            for i, sc in enumerate(scenarios, start=1):
                # Pre-flight estimate and logging for this scale level.
                est_mem = estimate_pyvrp_memory_bytes(sc.n)
                est_gib = bytes_to_gib(est_mem)

                print(f"[scenario {i:02d}/{len(scenarios)}] n={sc.n} K={sc.K} est={est_gib:.2f}GiB")

                if sc.n > args.pyvrp_max_n:
                    # User-enforced n cap: useful for smoke tests.
                    print(f"  -> skip (n > pyvrp-max-n={args.pyvrp_max_n})")
                    writer.writerow(
                        {
                            "mode": "pyvrp",
                            "n": sc.n,
                            "K": sc.K,
                            "m": sc.m,
                            "T": sc.T,
                            "estimated_mem_gib": f"{est_gib:.4f}",
                            "status": "skipped_pyvrp_max_n",
                            "elapsed_s": "",
                            "pyvrp_cost": "",
                            "error": "",
                        }
                    )
                    continue

                if not (args.force or threshold == 0 or est_mem <= threshold):
                    # Safety skip based on current machine memory budget.
                    print("  -> skip (estimated memory above threshold)")
                    writer.writerow(
                        {
                            "mode": "pyvrp",
                            "n": sc.n,
                            "K": sc.K,
                            "m": sc.m,
                            "T": sc.T,
                            "estimated_mem_gib": f"{est_gib:.4f}",
                            "status": "skipped_memory",
                            "elapsed_s": "",
                            "pyvrp_cost": "",
                            "error": "",
                        }
                    )
                    continue

                vrp_path = case_dir / f"scaling_n{sc.n}.vrp"
                write_vrplib_file(vrp_path, sc)

                try:
                    # Solve and record successful metrics.
                    cost, elapsed = solve_with_pyvrp(vrp_path, sc.solver_seed, args.timeout, sc.T)
                    print(f"  -> pyvrp: ok ({elapsed:.2f}s) cost={cost:.3f}")
                    writer.writerow(
                        {
                            "mode": "pyvrp",
                            "n": sc.n,
                            "K": sc.K,
                            "m": sc.m,
                            "T": sc.T,
                            "estimated_mem_gib": f"{est_gib:.4f}",
                            "status": "ok",
                            "elapsed_s": f"{elapsed:.6f}",
                            "pyvrp_cost": f"{cost:.6f}",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    # Keep the campaign running: log failure and continue.
                    print(f"  -> pyvrp: failed ({type(exc).__name__})")
                    writer.writerow(
                        {
                            "mode": "pyvrp",
                            "n": sc.n,
                            "K": sc.K,
                            "m": sc.m,
                            "T": sc.T,
                            "estimated_mem_gib": f"{est_gib:.4f}",
                            "status": "failed",
                            "elapsed_s": "",
                            "pyvrp_cost": "",
                            "error": f"{type(exc).__name__}:{exc}",
                        }
                    )

    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
