import random

def generate_test_case(num_nodes, distance_range=(1, 50), weight_range=(1, 15)):
    # Generate random distance matrix
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = random.randint(distance_range[0], distance_range[1])
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance  # Ensure symmetry

    # Generate random weights (depot has weight 0)
    weights = [0]  # Depot
    weights += [random.randint(weight_range[0], weight_range[1]) for _ in range(num_nodes - 1)]

    return distance_matrix, weights

def save_test_case_to_file(distance_matrix, weights, filename="test_case_20_nodes.txt"):
    with open(filename, "w") as file:
        file.write("Distance Matrix:\n")
        for row in distance_matrix:
            file.write(" ".join(map(str, row)) + "\n")
        
        file.write("\nWeights:\n")
        file.write(" ".join(map(str, weights)) + "\n")

# Parameters
num_nodes = 20  # 10 nodes (including depot)
distance_range = (1, 50)  # Random distances between 1 and 50
weight_range = (1, 10)    # Random weights between 1 and 10

# Generate test case
distance_matrix, weights = generate_test_case(num_nodes, distance_range, weight_range)

# Save test case to file
save_test_case_to_file(distance_matrix, weights)
print("Test case saved to 'test_case_20_nodes.txt'.")
