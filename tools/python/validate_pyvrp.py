#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import sys

def parse_solution_file(filepath: Path) -> tuple[list[list[int]], float | None]:
    routes = []
    cost = None
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Route'):
                # Format: "Route X: node1 node2 node3 ..."
                parts = line.split(':', maxsplit=1)
                if len(parts) == 2:
                    nodes_str = parts[1].strip()
                    if nodes_str:
                        route_nodes = [int(n) for n in nodes_str.split()]
                        routes.append(route_nodes)
            elif line.startswith('Cost:'):
                # Format: "Cost: 123.45"
                cost_str = line.split(':', maxsplit=1)[1].strip()
                cost = float(cost_str)
    
    return routes, cost

def validate_solution(
    instance_path: Path,
    solution_path: Path,
    round_func: str = "none",
    reference_path: Path | None = None,
) -> bool:
    import pyvrp
    from pyvrp import read, Solution

    if not instance_path.is_file():
        raise FileNotFoundError(f"instance not found: {instance_path}")
    if not solution_path.is_file():
        raise FileNotFoundError(f"solution not found: {solution_path}")
    print(f"Loading instance: {instance_path}")
    instance = read(instance_path, round_func=round_func)

    print(f"Loading solution: {solution_path}")
    routes_data, reported_cost = parse_solution_file(solution_path)
    
    if reported_cost is None:
        raise ValueError("solution does not contain a 'Cost: <value>' line")
    if not routes_data:
        raise ValueError("solution does not contain any routes")

    pyvrp_routes = []
    for r in routes_data:
        try:
            route = pyvrp.Route(instance, r, vehicle_type=0)
            pyvrp_routes.append(route)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"cannot create route {r}: {exc}") from exc

    solution = Solution(instance, pyvrp_routes)

    print(f"\n--- EVALUATION ---")
    print(f"Reported Cost:        {reported_cost:.3f}")

    is_feasible = solution.is_feasible()
    raw_distance = solution.distance()
    
    print(f"PyVRP Recalculated: {raw_distance:.3f}")
    print(f"Is Feasible:         {is_feasible}")
    
    if not is_feasible:
        raise ValueError("solution is infeasible according to PyVRP constraints")

    tolerance = 1e-3
    if not math.isclose(raw_distance, reported_cost, abs_tol=tolerance):
        raise ValueError(
            f"cost mismatch: reported={reported_cost}, recalculated={raw_distance}"
        )

    if reference_path:
        print(f"\n--- REFERENCE COMPARISON ---")
        if not reference_path.is_file():
            raise FileNotFoundError(f"reference solution not found: {reference_path}")
        ref_routes, ref_cost = parse_solution_file(reference_path)
        if ref_cost is None or ref_cost <= 0:
            raise ValueError("reference solution has no positive Cost value")
        print(f"Reference Solver (PyVRP) Cost: {ref_cost:.3f}")
        
        diff = reported_cost - ref_cost
        gap = (diff / ref_cost) * 100
        print(f"Relative Gap to Reference: {gap:+.2f}%")

        if abs(gap) > 0.01:
            print(f"Note: CUDA solution is {gap:.2f}% {'worse' if gap > 0 else 'better'} than Reference.")

        print("\n--- Route Comparison (Solution vs Reference) ---")
        max_cmp = min(len(routes_data), len(ref_routes), 3)
        for i in range(max_cmp):
            c_r = routes_data[i]
            r_r = ref_routes[i]
            print(f"Route {i+1} Solution:  {' '.join(map(str, c_r[:10]))}{'...' if len(c_r) > 10 else ''}")
            print(f"Route {i+1} Reference: {' '.join(map(str, r_r[:10]))}{'...' if len(r_r) > 10 else ''}")
    
    print("\nVALIDATION SUCCESSFUL.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate VRP solutions using PyVRP.")
    parser.add_argument("instance", help="Path to the VRPLIB instance file (.vrp)")
    parser.add_argument("solution", help="Path to the solution text file")
    parser.add_argument("--round-func", default="none", choices=["none", "round", "trunc", "dimacs"], 
                        help="Rounding function used for distance matrix (default: none)")
    parser.add_argument("--reference", help="Path to a reference solution for comparison")
    
    args = parser.parse_args()
    validate_solution(
        Path(args.instance),
        Path(args.solution),
        args.round_func,
        Path(args.reference) if args.reference else None,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
