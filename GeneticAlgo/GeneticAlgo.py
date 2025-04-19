import random
from datetime import datetime

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

def generate_individual(num_nodes):
    # Generate a random route (excluding depot)
    individual = list(range(1, num_nodes))
    random.shuffle(individual)
    return individual

def generate_population(population_size, num_nodes):
    # Generate a population of random individuals
    return [generate_individual(num_nodes) for _ in range(population_size)]

def fitness(individual, weights, distance_matrix, battery_capacity, base_power, alpha, velocity):
    # Calculate total energy for the route
    route = [0] + individual + [0]  # Start and end at depot
    return calculate_total_energy(route, weights, distance_matrix, battery_capacity, base_power, alpha, velocity)

def select_parents(population, fitness_scores):
    # Select two parents using tournament selection
    tournament_size = 3
    parents = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitness_scores)), tournament_size)
        parents.append(min(tournament, key=lambda x: x[1])[0])
    return parents

def crossover(parent1, parent2):
    # Perform ordered crossover (OX)
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    remaining = [gene for gene in parent2 if gene not in child]
    child[:start] = remaining[:start]
    child[end:] = remaining[start:]
    return child

def mutate(individual, mutation_rate=0.1):
    # Perform swap mutation
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]
    return individual

def genetic_algorithm(num_nodes, weights, distance_matrix, battery_capacity, base_power, alpha, velocity, population_size=50, generations=100):
    # Initialize population
    population = generate_population(population_size, num_nodes)
    
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness(individual, weights, distance_matrix, battery_capacity, base_power, alpha, velocity) for individual in population]
        
        # Select parents and create offspring
        offspring = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([mutate(child1), mutate(child2)])
        
        # Replace population with offspring
        population = offspring
    
    # Find the best individual
    best_individual = min(population, key=lambda x: fitness(x, weights, distance_matrix, battery_capacity, base_power, alpha, velocity))
    best_route = [0] + best_individual + [0]
    min_energy = fitness(best_individual, weights, distance_matrix, battery_capacity, base_power, alpha, velocity)
    
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

    # Find the optimal route using Genetic Algorithm
    best_route, min_energy = genetic_algorithm(len(distance_matrix), weights, distance_matrix, battery_capacity, base_power, alpha, velocity)

    # Output the result
    print("\nOptimal Route (using Genetic Algorithm):")
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

