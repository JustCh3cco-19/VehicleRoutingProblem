#!/usr/bin/env python3
"""Run a single PyVRP solve for make-based experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyvrp import Model, read, stop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PyVRP instance solve")
    parser.add_argument("instance_path")
    parser.add_argument("runtime_s", type=float)
    parser.add_argument("seed", type=int)
    parser.add_argument("solution_out")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    inst = read(args.instance_path, round_func="none")
    model = Model.from_data(inst)
    res = model.solve(stop.MaxRuntime(args.runtime_s), seed=args.seed)
    best = res.best

    out_path = Path(args.solution_out)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, route in enumerate(best.routes(), start=1):
            visits = " ".join(map(str, route.visits()))
            f.write(f"Route {idx}: {visits}\n")
        f.write(f"Cost: {best.distance():.6f}\n")

    if best.is_feasible():
        print(f"best_cost={best.distance():.6f}")
    else:
        print("best_cost=")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
