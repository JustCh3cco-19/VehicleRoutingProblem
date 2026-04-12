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

def validate_solution(instance_path, solution_path, round_func="none", reference_path=None):
    print(f"Loading instance: {instance_path}")
    try:
        instance = read(instance_path, round_func=round_func)
    except Exception as e:
        print(f"Failed to load instance: {e}")
        sys.exit(1)

    print(f"Loading CUDA solution: {solution_path}")
    routes_data, reported_cost = parse_solution_file(solution_path)
    
    if reported_cost is None:
        print("Error: Solution file does not contain a 'Cost: <value>' line.")
        sys.exit(1)

    pyvrp_routes = []
    for r in routes_data:
        try:
            route = pyvrp.Route(instance, r, vehicle_type=0)
            pyvrp_routes.append(route)
        except Exception as e:
            print(f"Error creating route {r}: {e}")
            sys.exit(1)

    try:
        solution = Solution(instance, pyvrp_routes)
    except Exception as e:
        print(f"Error creating solution: {e}")
        sys.exit(1)

    print(f"\n--- EVALUATION ---")
    print(f"CUDA Reported Cost:  {reported_cost:.3f}")
    
    cost_evaluator = CostEvaluator(load_penalties=[1000], tw_penalty=1000, dist_penalty=1000)
    penalised_cost = cost_evaluator.penalised_cost(solution)
    is_feasible = solution.is_feasible()
    raw_distance = solution.distance()
    
    print(f"PyVRP Recalculated: {raw_distance:.3f}")
    print(f"Is Feasible:         {is_feasible}")
    
    if not is_feasible:
        print("\nERROR: CUDA Solution is infeasible according to PyVRP constraints.")
        sys.exit(1)

    tolerance = 1e-3
    if not math.isclose(raw_distance, reported_cost, abs_tol=tolerance):
        print(f"\nERROR: Cost mismatch! CUDA: {reported_cost}, PyVRP: {raw_distance}")
        sys.exit(1)

    if reference_path:
        print(f"\n--- REFERENCE COMPARISON ---")
        ref_routes, ref_cost = parse_solution_file(reference_path)
        print(f"Reference Solver (PyVRP) Cost: {ref_cost:.3f}")
        
        diff = reported_cost - ref_cost
        gap = (diff / ref_cost) * 100
        print(f"Relative Gap to Reference: {gap:+.2f}%")

        if abs(gap) > 0.01:
            print(f"Note: CUDA solution is {gap:.2f}% {'worse' if gap > 0 else 'better'} than Reference.")

        print("\n--- Route Comparison (CUDA vs Reference) ---")
        max_cmp = min(len(routes_data), len(ref_routes), 3)
        for i in range(max_cmp):
            c_r = routes_data[i]
            r_r = ref_routes[i]
            print(f"Route {i+1} CUDA:      {' '.join(map(str, c_r[:10]))}{'...' if len(c_r) > 10 else ''}")
            print(f"Route {i+1} Reference: {' '.join(map(str, r_r[:10]))}{'...' if len(r_r) > 10 else ''}")
    
    print("\nVALIDATION SUCCESSFUL.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate VRP solutions using PyVRP.")
    parser.add_argument("instance", help="Path to the VRPLIB instance file (.vrp)")
    parser.add_argument("solution", help="Path to the solution text file")
    parser.add_argument("--round-func", default="none", choices=["none", "round", "trunc", "dimacs"], 
                        help="Rounding function used for distance matrix (default: none)")
    parser.add_argument("--reference", help="Path to a reference solution for comparison")
    
    args = parser.parse_args()
    
    validate_solution(args.instance, args.solution, args.round_func, args.reference)
