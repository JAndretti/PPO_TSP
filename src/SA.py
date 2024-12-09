import random
import numpy as np
from distances import calculate_distance


class SA:

    def __init__(self, instance, distance_matrix, T_final, K):
        self.instance = instance
        self.distance_matrix = distance_matrix
        self.temperature_initial = 1
        self.temperature = self.temperature_initial
        self.T_final = T_final
        self.K = K
        self.alpha = (self.T_final / self.temperature_initial) ** (1 / self.K)
        self.distance = calculate_distance(self.instance, self.distance_matrix)

    def two_opt(self):
        n1 = random.randint(0, len(self.instance) - 1)
        n2 = random.randint(0, len(self.instance) - 1)
        if n1 > n2:
            n1, n2 = n2, n1
        self.instance[n1:n2] = reversed(self.instance[n1:n2])

    def temp_step(self, k):
        self.temperature = self.alpha**k * self.temperature_initial

    def metropolis_hastings_step(
        self,
        current_solution,
        candidate_solution,
    ):
        delta_energy = calculate_distance(
            candidate_solution, self.distance_matrix
        ) - calculate_distance(current_solution, self.distance_matrix)

        # Sample uniform random variable
        u = random.uniform(0, 1)

        # Metropolis-Hastings acceptance step
        if u < np.exp(-delta_energy / self.temperature):  # Accept condition
            return candidate_solution
        else:
            return current_solution

    def main(self):
        k = 0
        # print("Initial distance: ", self.distance)
        while self.temperature > self.T_final:
            current_solution = self.instance.copy()
            self.two_opt()
            self.instance = self.metropolis_hastings_step(
                current_solution, self.instance
            )
            self.temp_step(k)
            k += 1
        return self.instance, calculate_distance(self.instance, self.distance_matrix)


# def generate_tsp_instance(num_points):
#     points = [
#         (i, (random.uniform(0, 1), random.uniform(0, 1))) for i in range(num_points)
#     ]
#     return points


# tsp = generate_tsp_instance(20)
# index, locations = zip(*tsp)
# index = list(index)
# locations = list(locations)
# distance_matrix = calculate_distance_matrix(locations)
# sa = SA(index, distance_matrix, 0.001, 1000)
# instance, distance = sa.main()
# print(distance)
