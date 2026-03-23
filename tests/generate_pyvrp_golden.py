#!/usr/bin/env python3
import argparse
import csv
import inspect
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scenario:
    n: int
    K: int
    m: int
    T: int
    solver_seed: int
    instance_seed: int
    layout_id: int
    max_gap_pct: float


@dataclass
class Point:
    x: float
    y: float


def ensure_vrp_python():
    vrp_python = Path("VRP/bin/python")
    if not vrp_python.exists():
        raise RuntimeError("missing venv interpreter: VRP/bin/python")

    if Path(sys.executable).absolute() != vrp_python.absolute():
        os.execv(str(vrp_python), [str(vrp_python)] + sys.argv)


def lcg_next(state: int) -> int:
    return ((state * 1664525) + 1013904223) & 0xFFFFFFFF


def rand01_lcg(state: int):
    nxt = lcg_next(state)
    return nxt / 0xFFFFFFFF, nxt


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def generate_points(n: int, seed: int, layout: int):
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
    pts = generate_points(sc.n, sc.instance_seed, sc.layout_id)
    lines = [
        f"NAME : golden_n{sc.n}",
        "TYPE : CVRP",
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
    for attr in ("cost", "best_cost", "objective"):
        if hasattr(result, attr):
            v = getattr(result, attr)
            return float(v() if callable(v) else v)
    for meth in ("best", "best_solution", "solution"):
        if hasattr(result, meth):
            obj = getattr(result, meth)
            obj = obj() if callable(obj) else obj
            if hasattr(obj, "cost"):
                v = getattr(obj, "cost")
                return float(v() if callable(v) else v)
    raise RuntimeError("unable to extract cost from pyvrp result")


def solve_pyvrp(vrp_path: Path, seed: int, iterations: int):
    import pyvrp

    data = pyvrp.read(str(vrp_path), round_func="round")
    kwargs = {"seed": seed}
    sig = inspect.signature(pyvrp.solve)
    if "stop" in sig.parameters:
        kwargs["stop"] = pyvrp.stop.MaxIterations(max(1, iterations))
    start = time.perf_counter()
    result = pyvrp.solve(data, **kwargs)
    elapsed = time.perf_counter() - start
    return extract_pyvrp_cost(result), elapsed


def default_scenarios():
    return [
        Scenario(500, 8, 64, 40, 9000, 19000, 0, 15.0),
        Scenario(1000, 8, 64, 40, 9001, 19001, 1, 20.0),
        Scenario(2000, 8, 64, 40, 9002, 19002, 2, 25.0),
        Scenario(4000, 8, 32, 20, 9003, 19003, 3, 30.0),
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate golden PyVRP baseline CSV.")
    parser.add_argument("--out", default="tests/files/golden_pyvrp.csv")
    args = parser.parse_args()

    ensure_vrp_python()
    scenarios = default_scenarios()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="golden_pyvrp_") as td:
        tmp = Path(td)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "n",
                    "K",
                    "m",
                    "T",
                    "solver_seed",
                    "instance_seed",
                    "layout_id",
                    "pyvrp_cost",
                    "max_gap_pct",
                ]
            )
            for sc in scenarios:
                vrp = tmp / f"case_n{sc.n}.vrp"
                write_vrplib_file(vrp, sc)
                cost, elapsed = solve_pyvrp(vrp, sc.solver_seed, sc.T)
                print(f"[OK] n={sc.n} cost={cost:.3f} elapsed={elapsed:.2f}s")
                w.writerow(
                    [
                        sc.n,
                        sc.K,
                        sc.m,
                        sc.T,
                        sc.solver_seed,
                        sc.instance_seed,
                        sc.layout_id,
                        f"{cost:.6f}",
                        f"{sc.max_gap_pct:.2f}",
                    ]
                )
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()

