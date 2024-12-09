import numpy as np
import random
import copy as cp
from or_tools import OR_TOOLS
from SA import SA
from distances import calculate_distance_matrix, calculate_distance

MAT_MUL = 1


def generate_tsp_instance(num_points):
    points = [
        (i, (random.uniform(0, 1), random.uniform(0, 1))) for i in range(num_points)
    ]
    return points


class ENV:

    def __init__(self, num_points, or_tools_time, T_final, K):
        self.num_points = num_points
        # self.max_episode_len = max_episode_len
        self.tsp_instance = generate_tsp_instance(self.num_points)
        self.temperature_initial = 1
        _ = self.reset()
        sa = SA(self.index, self.distance_matrix, T_final, K)
        instance, distance = sa.main()
        solveur = OR_TOOLS(
            self.locations,
            np.rint(self.distance_matrix * 100).astype(int),
            or_tools_time,
        )
        self.solution, self.solution_distance = solveur.main()
        self.solution_distance /= 100
        self.T_final = T_final
        self.K = K
        self.alpha = (self.T_final / self.temperature_initial) ** (1 / self.K)
        print("Base distance: ", self.initial_distance)
        print("Distance SA: ", distance)
        print("Distance OR_TOOLS: ", self.solution_distance)

    def two_opt(self, action):
        val1, val2 = action
        idx1, idx2 = self.index.index(val1), self.index.index(val2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        self.index[idx1:idx2] = reversed(self.index[idx1:idx2])

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
            return candidate_solution, -delta_energy
        else:
            return current_solution, -delta_energy

    def step(self, action: list[int, int]) -> tuple:
        done = False
        current_solution = self.index.copy()
        self.two_opt(action)
        self.index, reward = self.metropolis_hastings_step(current_solution, self.index)
        self.distance = calculate_distance(self.index, self.distance_matrix)
        if self.distance < self.best_distance:
            self.best_solution = cp.deepcopy(self.index)
            self.best_distance = self.distance
        self.temp_step(self.current_step)
        self.current_step += 1
        if self.temperature <= self.T_final:
            done = True
        if done:
            self.final = self.initial_distance - self.best_distance
        info = {}  # Placeholder for additional information
        return self.index, reward, self.temperature, done, info

    # def step(self, action: list[int, int]) -> tuple:
    #     done = False
    #     candidate_solution = self.two_opt(action)
    #     tmp_distance = calculate_distance(candidate_solution, self.distance_matrix)
    #     reward = (self.distance - tmp_distance) * MAT_MUL

    #     if reward == 0:
    #         reward = -10
    #     self.distance = tmp_distance
    #     if self.distance < self.best_distance:
    #         self.best_solution = cp.deepcopy(self.index)
    #         self.best_distance = self.distance
    #     self.current_step += 1
    #     done = self.current_step >= self.max_episode_len
    #     if done:
    #         self.final = self.initial_distance - self.best_distance
    #     info = {}  # Placeholder for additional information
    #     index = cp.deepcopy(self.index)
    #     return index, reward, done, info

    def shuffle(self):
        random.seed()
        random.shuffle(self.index)
        self.distance = calculate_distance(self.index, self.distance_matrix)
        self.initial_distance = cp.deepcopy(self.distance)

    def reset(self):
        self.current_step = 0
        self.temperature = self.temperature_initial
        self.tsp_instance = generate_tsp_instance(self.num_points)
        self.index, self.locations = zip(*self.tsp_instance)
        self.index = list(self.index)
        self.locations = list(self.locations)
        self.distance_matrix = calculate_distance_matrix(self.locations, MAT_MUL)
        self.shuffle()
        sa = SA(self.index, self.distance_matrix, 0.01, 40)
        instance, distance = sa.main()
        # solveur = OR_TOOLS(
        #     self.locations,
        #     np.rint(self.distance_matrix * 100).astype(int),
        #     1,
        # )
        # self.solution, self.solution_distance = solveur.main()
        # self.solution_distance /= 100
        self.best_distance = np.inf
        index = cp.deepcopy(self.index)
        self.best_solution = cp.deepcopy(self.index)
        self.initial_solution = cp.deepcopy(self.index)
        return (
            self.index,
            {idx: loc for idx, loc in zip(index, self.locations)},
            self.temperature,
            distance,
        )

    def print_info(self):
        print("Current Solution: ", self.index)
        print("Distance: ", self.distance)


# env = ENV(20, 40)
# env.print_info()
# done = False
# while not done:
#     action = random.sample(range(20), 2)
#     index, reward, done, info = env.step(action)
#     env.print_info()
#     print("Reward: ", reward)
