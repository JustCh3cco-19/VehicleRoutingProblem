#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

def generate_instance(name, num_clients, num_vehicles, grid_size=100, seed=None,
                      capacity_slack_percent=20):
    import numpy as np

    if seed is not None:
        np.random.seed(seed)
    
    # Depot at center or random? Let's go with random for variety
    depot_coord = np.random.randint(0, grid_size, size=2)
    client_coords = np.random.randint(0, grid_size, size=(num_clients, 2))
    all_coords = np.vstack([depot_coord, client_coords])
    
    # Unit-demand CVRP: one unit per customer, depot demand 0.
    demands = np.ones(num_clients, dtype=int)
    all_demands = np.concatenate(([0], demands))

    numerator = (100 + capacity_slack_percent) * num_clients
    denominator = 100 * num_vehicles
    capacity = (numerator + denominator - 1) // denominator
    if capacity < 1:
        capacity = 1
    
    instance_data = {
        "NAME": name,
        "TYPE": "CVRP",
        "DIMENSION": num_clients + 1,
        "VEHICLES": num_vehicles,
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        "CAPACITY": int(capacity),
        "NODE_COORD_SECTION": all_coords,
        "DEMAND_SECTION": all_demands,
        "DEPOT_SECTION": [1, -1]
    }
    
    return instance_data

def save_reference_solution(instance_path, output_path, runtime=2.0):
    print(f"Solving instance for {runtime}s to create a reference solution...")

    # Load from file to ensure exactly the same distance interpretation.
    from pyvrp import read, Model
    from pyvrp import stop
    pyvrp_instance = read(instance_path, round_func="none")
    
    model = Model.from_data(pyvrp_instance)
    
    # Solve
    res = model.solve(stop.MaxRuntime(runtime))
    best = res.best
    
    if not best.is_feasible():
        print("Warning: Could not find a feasible reference solution in the given time.")
        return False
    
    # Save in our validation format
    with open(output_path, 'w', encoding="utf-8") as f:
        for i, route in enumerate(best.routes()):
            nodes_str = " ".join(map(str, route.visits()))
            f.write(f"Route {i+1}: {nodes_str}\n")
        f.write(f"Cost: {best.distance()}\n")
    
    print(f"Reference solution saved to {output_path}")
    print(f"Reference Cost: {best.distance()}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VRP instances and reference solutions.")
    parser.add_argument("--name", default="synthetic_vrp", help="Name of the instance")
    parser.add_argument("--clients", "--nodes", type=int, default=20, help="Number of customer nodes")
    parser.add_argument("--vehicles", type=int, default=5, help="Number of vehicles available")
    parser.add_argument("--grid", type=int, default=100, help="Coordinate grid size (0 to grid)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", default="problem.vrp", help="Output .vrp file path")
    parser.add_argument("--solve", action="store_true", help="Also generate a reference solution (.txt)")
    parser.add_argument("--runtime", type=float, default=2.0, help="Max solve time for reference solution")
    parser.add_argument("--capacity-slack-percent", type=int, default=20,
                        help="Capacity slack percentage over n / K (default: 20)")

    args = parser.parse_args()

    if args.clients < 1:
        parser.error("--clients must be positive")
    if args.vehicles < 1:
        parser.error("--vehicles must be positive")
    if args.grid < 1:
        parser.error("--grid must be positive")
    if args.capacity_slack_percent < 0:
        parser.error("--capacity-slack-percent must be non-negative")
    if args.runtime <= 0:
        parser.error("--runtime must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")

    instance = generate_instance(args.name, args.clients, args.vehicles,
                                 args.grid, args.seed,
                                 args.capacity_slack_percent)

    # Save instance
    import vrplib

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    vrplib.write_instance(args.output, instance)
    print(f"Problem instance saved to {args.output}")

    if args.solve:
        sol_path = os.path.splitext(args.output)[0] + "_solution.txt"
        if not save_reference_solution(args.output, sol_path, args.runtime):
            return 1
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ImportError as exc:
        print(f"Error: missing Python dependency: {exc.name}", file=sys.stderr)
        raise SystemExit(1)
