#!/usr/bin/env python3
import argparse
import sys
import math

try:
    import pyvrp
    from pyvrp import read, Solution, CostEvaluator
except ImportError:
    print("Error: pyvrp is not installed. Please install it using 'pip install pyvrp'")
    sys.exit(1)

def parse_solution_file(filepath):
    routes = []
    cost = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('Route'):
                # Format: "Route X: node1 node2 node3 ..."
                parts = line.split(':')
                if len(parts) == 2:
                    nodes_str = parts[1].strip()
                    if nodes_str:
                        route_nodes = [int(n) for n in nodes_str.split()]
                        routes.append(route_nodes)
            elif line.startswith('Cost:'):
                # Format: "Cost: 123.45"
                cost_str = line.split(':')[1].strip()
                cost = float(cost_str)
    
    return routes, cost

def validate_solution(instance_path, solution_path, round_func="none"):
    print(f"Loading instance: {instance_path}")
    try:
        # round_func="none" ensures distances are exact floats, 
        # or use "round" for TSPLIB standard if your C code rounds.
        instance = read(instance_path, round_func=round_func)
    except Exception as e:
        print(f"Failed to load instance: {e}")
        sys.exit(1)

    print(f"Loading solution: {solution_path}")
    routes_data, reported_cost = parse_solution_file(solution_path)
    
    if reported_cost is None:
        print("Error: Solution file does not contain a 'Cost: <value>' line.")
        sys.exit(1)

    if not routes_data:
        print("Warning: No routes found in the solution file.")
    
    # Construct PyVRP Routes
    # Note: PyVRP Route expects the problem data instance and a list of client indices.
    # In PyVRP, clients are 1-indexed (0 is depot). Assuming the C code output uses 
    # 1-indexed customers and omits the depot in the route string.
    pyvrp_routes = []
    for r in routes_data:
        try:
            route = pyvrp.Route(instance, r, vehicle_type=0)
            pyvrp_routes.append(route)
        except Exception as e:
            print(f"Error creating route {r}: {e}")
            sys.exit(1)

    # Construct the PyVRP Solution
    try:
        solution = Solution(instance, pyvrp_routes)
    except Exception as e:
        print(f"Error creating solution: {e}")
        sys.exit(1)

    print(f"\nEvaluating Solution...")
    print(f"Reported Cost: {reported_cost}")
    
    # Evaluate feasibility and cost
    # PyVRP requires explicit penalties: [load_penalty], tw_penalty, dist_penalty
    # Defaulting to high penalties for validation purposes.
    cost_evaluator = CostEvaluator(load_penalties=[1000], tw_penalty=1000, dist_penalty=1000)
    penalised_cost = cost_evaluator.penalised_cost(solution)
    is_feasible = solution.is_feasible()
    
    # Also calculate the raw distance (unpenalized cost)
    # For a feasible solution, this should match the penalised_cost.
    raw_distance = solution.distance()
    
    print(f"PyVRP Penalised Cost: {penalised_cost}")
    print(f"PyVRP Raw Distance:   {raw_distance}")
    print(f"Is Feasible:          {is_feasible}")
    
    if not is_feasible:
        print("\nValidation Failed: The solution is infeasible according to PyVRP constraints.")
        # We check which cost matches the reported one to provide better feedback
        if math.isclose(raw_distance, reported_cost, abs_tol=tolerance):
            print("Note: The reported cost matches the raw distance, but the solution violates constraints (e.g. capacity).")
        sys.exit(1)

    # Allow a small tolerance for floating point comparisons
    tolerance = 1e-3
    if not math.isclose(penalised_cost, reported_cost, abs_tol=tolerance):
        print(f"\nValidation Failed: Cost mismatch! Reported: {reported_cost}, PyVRP calculated: {penalised_cost}")
        sys.exit(1)

    print("\nValidation Successful: Solution is feasible and costs match.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VRP solutions using PyVRP.")
    parser.add_argument("instance", help="Path to the VRPLIB instance file (.vrp)")
    parser.add_argument("solution", help="Path to the solution text file")
    parser.add_argument("--round-func", default="none", choices=["none", "round", "trunc", "dimacs"], 
                        help="Rounding function used for distance matrix (default: none)")
    
    args = parser.parse_args()
    
    validate_solution(args.instance, args.solution, args.round_func)
