import itertools
from datetime import datetime  # Import datetime module

def calculate_energy(base_power, alpha, weight, distance, velocity):
    return (base_power + alpha * weight) * (distance / velocity)

def calculate_total_energy(route, weights, distance_matrix, battery_capacity, base_power, alpha, velocity):
    total_energy = 0
    current_battery = battery_capacity

    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i + 1]
        distance = distance_matrix[from_node][to_node]
        weight = weights[from_node]
        
        energy_needed = calculate_energy(base_power, alpha, weight, distance, velocity)
        distance_to_depot = distance_matrix[to_node][0]
        energy_needed_to_depot = calculate_energy(base_power, alpha, 0, distance_to_depot, velocity)

        if current_battery < energy_needed + energy_needed_to_depot:
            total_energy += calculate_energy(base_power, alpha, 0, distance_matrix[from_node][0], velocity)
            current_battery = battery_capacity
            total_energy += calculate_energy(base_power, alpha, 0, distance_matrix[0][to_node], velocity)
            current_battery -= calculate_energy(base_power, alpha, 0, distance_matrix[0][to_node], velocity)
        
        current_battery -= energy_needed
        total_energy += energy_needed

    last_node = route[-2]
    total_energy += calculate_energy(base_power, alpha, 0, distance_matrix[last_node][0], velocity)
    
    return total_energy

def find_optimal_route(num_nodes, weights, distance_matrix, battery_capacity, base_power, alpha, velocity):
    nodes = list(range(1, num_nodes))  # Exclude depot (node 0)
    best_route = None
    min_energy = float('inf')
    
    # Generate all permutations of nodes
    for route in itertools.permutations(nodes):
        route_with_depot = (0,) + route + (0,)  # Start and end at depot
        
        # Calculate the total energy for this route
        total_energy = calculate_total_energy(route_with_depot, weights, distance_matrix, battery_capacity, base_power, alpha,  velocity)
        
        # Update the best route if this one is more efficient
        if total_energy < min_energy:
            min_energy = total_energy
            best_route = route_with_depot

    return best_route, min_energy
def read_test_case(filename="test_case_10_nodes.txt"):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    # Read distance matrix
    distance_matrix = []
    for line in lines[1:11]:  # Assuming 10 nodes (10 lines after the header)
        row = list(map(float, line.strip().split()))
        distance_matrix.append(row)
    
    # Read weights
    weights_line = lines[13].strip()  # Assuming weights are on the 13th line
    weights = list(map(float, weights_line.split()))
    
    return distance_matrix, weights
def main():
    # Print start time
    start_time = datetime.now()
    print(f"Code execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Drone parameters
    base_power = 10  # Base power consumption
    alpha = 2  # Power consumption increase per kg
    battery_capacity = 100  # Battery capacity
    velocity = 10  # Drone velocity (distance units per time unit)

    # Read test case from file
    distance_matrix, weights = read_test_case()

    # Find the optimal route
    num_nodes = len(distance_matrix)
    best_route, min_energy = find_optimal_route(num_nodes, weights, distance_matrix, battery_capacity, base_power, alpha,  velocity)

    # Output the result
    print("\nOptimal Route (including returning to depot when necessary):")
    for node in best_route:
        print(f"L{node} -> ", end="")
    print(f"Depot\nTotal Energy: {min_energy} units")

    # Print end time
    end_time = datetime.now()
    print(f"Code execution ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Print total execution time
    total_time = end_time - start_time
    print(f"Total execution time: {total_time}")

if __name__ == "__main__":
    main()

