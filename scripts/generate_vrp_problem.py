#!/usr/bin/env python3
import argparse
import numpy as np
import vrplib
import sys
import os

try:
    import pyvrp
    from pyvrp import Model, stop
except ImportError:
    print("Error: pyvrp is not installed. Please run in the created .venv.")
    sys.exit(1)

def generate_instance(name, num_clients, num_vehicles, capacity, grid_size=100, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Depot at center or random? Let's go with random for variety
    depot_coord = np.random.randint(0, grid_size, size=2)
    client_coords = np.random.randint(0, grid_size, size=(num_clients, 2))
    all_coords = np.vstack([depot_coord, client_coords])
    
    # Random demands (1 to 10)
    demands = np.random.randint(1, 11, size=num_clients)
    all_demands = np.concatenate(([0], demands))
    
    instance_data = {
        "NAME": name,
        "TYPE": "CVRP",
        "DIMENSION": num_clients + 1,
        "VEHICLES": num_vehicles,
        "EDGE_WEIGHT_TYPE": "EUC_2D",
        "CAPACITY": capacity,
        "NODE_COORD_SECTION": all_coords,
        "DEMAND_SECTION": all_demands,
        "DEPOT_SECTION": [1, -1]
    }
    
    return instance_data

def save_reference_solution(instance_data, output_path, runtime=2.0):
    print(f"Solving instance for {runtime}s to create a reference solution...")
    
    model = Model()
    coords = instance_data["NODE_COORD_SECTION"]
    demands = instance_data["DEMAND_SECTION"]
    
    # Add Depot
    model.add_depot(x=int(coords[0, 0]), y=int(coords[0, 1]))
    
    # Add Clients (starting from index 1)
    for i in range(1, len(coords)):
        model.add_client(
            x=int(coords[i, 0]), 
            y=int(coords[i, 1]), 
            delivery=[int(demands[i])]
        )
    
    # Add Vehicle Type
    model.add_vehicle_type(
        capacity=[int(instance_data["CAPACITY"])], 
        num_available=int(instance_data["VEHICLES"])
    )
    
    # Solve
    res = model.solve(stop.MaxRuntime(runtime))
    best = res.best
    
    if not best.is_feasible():
        print("Warning: Could not find a feasible reference solution in the given time.")
        return
    
    # Save in our validation format
    with open(output_path, 'w') as f:
        for i, route in enumerate(best.routes()):
            # PyVRP Route.visits() returns 1-indexed customer IDs
            nodes_str = " ".join(map(str, route.visits()))
            f.write(f"Route {i+1}: {nodes_str}\n")
        f.write(f"Cost: {best.distance()}\n")
    
    print(f"Reference solution saved to {output_path}")
    print(f"Reference Cost: {best.distance()}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VRP instances and reference solutions.")
    parser.add_argument("--name", default="synthetic_vrp", help="Name of the instance")
    parser.add_argument("--clients", "--nodes", type=int, default=20, help="Number of customer nodes")
    parser.add_argument("--vehicles", type=int, default=5, help="Number of vehicles available")
    parser.add_argument("--capacity", type=int, default=50, help="Vehicle capacity")
    parser.add_argument("--unlimited", action="store_true", help="Set capacity to sum of all demands (effectively unlimited)")
    parser.add_argument("--grid", type=int, default=100, help="Coordinate grid size (0 to grid)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--output", default="problem.vrp", help="Output .vrp file path")
    parser.add_argument("--solve", action="store_true", help="Also generate a reference solution (.txt)")
    parser.add_argument("--runtime", type=float, default=2.0, help="Max solve time for reference solution")

    args = parser.parse_args()

    instance = generate_instance(args.name, args.clients, args.vehicles, args.capacity, args.grid, args.seed)
    
    if args.unlimited:
        # Sum of all client demands (demands[0] is depot, which is 0)
        total_demand = sum(instance["DEMAND_SECTION"])
        instance["CAPACITY"] = int(total_demand)
        print(f"Unlimited mode: Setting capacity to {total_demand}")

    # Save instance
    vrplib.write_instance(args.output, instance)
    print(f"Problem instance saved to {args.output}")

    if args.solve:
        sol_path = os.path.splitext(args.output)[0] + "_solution.txt"
        save_reference_solution(instance, sol_path, args.runtime)

if __name__ == "__main__":
    main()
