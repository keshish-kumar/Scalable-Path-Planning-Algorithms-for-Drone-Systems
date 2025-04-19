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
def alternate_route_algorithm(weights, distance_matrix, battery_capacity, base_power, alpha, velocity):
    sorted_nodes = sorted(range(1, len(weights)), key=lambda x: weights[x])
    
    route = [0]  # Start at depot
    left, right = 0, len(sorted_nodes) - 1
    while left <= right:
        if left == right:
            route.append(sorted_nodes[left])
        else:
            route.append(sorted_nodes[left])
            route.append(sorted_nodes[right])
        left += 1
        right -= 1
    route.append(0)  # Return to depot

    total_energy = calculate_total_energy(route, weights, distance_matrix, battery_capacity, base_power, alpha, velocity)
    
    return route, total_energy

def read_test_case(filename="test_case_10_nodes.txt"):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    distance_matrix = []
    for line in lines[1:11]:  # Adjusted for 10 nodes
        row = list(map(float, line.strip().split()))
        distance_matrix.append(row)
    
    weights_line = lines[13].strip()  # Adjusted for 10 nodes
    weights = list(map(float, weights_line.split()))
    
    return distance_matrix, weights

def main():
    start_time = datetime.now()
    print(f"Code execution started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    base_power = 10
    alpha = 2
    battery_capacity = 100
    velocity = 10

    distance_matrix, weights = read_test_case()
    
    route, total_energy = alternate_route_algorithm(weights, distance_matrix, battery_capacity, base_power, alpha, velocity)
    
    print("\nAlternate Route (lightest to heaviest alternating):")
    for node in route:
        print(f"L{node} -> ", end="")
    print("Depot")
    print(f"Total Energy: {total_energy} units")

    end_time = datetime.now()
    print(f"Code execution ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()


