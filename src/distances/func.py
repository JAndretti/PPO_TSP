import numpy as np


# Function to calculate Euclidean distance between two points
def distance(x1, y1, x2, y2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Function to calculate the distance matrix
def calculate_distance_matrix(locations, MAT_MUL=100):
    """Calculate the distance matrix between locations."""
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations), dtype=np.float64)

    for i in range(num_locations):
        for j in range(num_locations):
            x1, y1 = locations[i]
            x2, y2 = locations[j]
            distance_matrix[i, j] = distance(x1, y1, x2, y2)
    dist_matrix = np.array(distance_matrix) * MAT_MUL
    # mat = np.rint(dist_matrix).astype(int)
    return dist_matrix


def calculate_distance(instance, distance_matrix):
    """Calculate the total distance of a TSP tour."""
    total_distance = 0
    for i in range(len(instance) - 1):
        total_distance += distance_matrix[instance[i], instance[i + 1]]
    total_distance += distance_matrix[
        instance[-1], instance[0]
    ]  # Return to the starting point
    return total_distance
