import numpy as np
import random
import copy as cp
from or_tools import OR_TOOLS
from distances import calculate_distance_matrix

MAT_MUL = 100


def generate_tsp_instance(num_points):
    points = [
        (i, (random.uniform(0, 1), random.uniform(0, 1))) for i in range(num_points)
    ]
    return points


class ENV:
    def __init__(self, num_points, max_episode_len, or_tools_time):
        self.num_points = num_points
        self.max_episode_len = max_episode_len
        self.tsp_instance = generate_tsp_instance(self.num_points)
        _ = self.reset()
        solveur = OR_TOOLS(self.locations, self.distance_matrix, or_tools_time)
        self.solution, self.solution_distance = solveur.main()
        print("Base distance: ", self.initial_distance)
        print("Distance OR_TOOLS: ", self.solution_distance)

    def calculate_distance(self):
        total_distance = 0
        for i in range(len(self.index) - 1):
            total_distance += self.distance_matrix[self.index[i], self.index[i + 1]]
        total_distance += self.distance_matrix[
            self.index[-1], self.index[0]
        ]  # Return to the starting point
        return total_distance

    def step(self, action: list[int, int]) -> tuple:
        done = False
        val1, val2 = action
        idx1, idx2 = self.index.index(val1), self.index.index(val2)
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        self.index[idx1:idx2] = reversed(self.index[idx1:idx2])
        tmp_distance = self.calculate_distance()
        # reward = self.distance - tmp_distance / MAT_MUL
        reward = (self.distance - tmp_distance) / self.distance
        if reward == 0:
            reward = -1
        self.distance = tmp_distance
        if self.distance < self.best_distance:
            self.best_solution = cp.deepcopy(self.index)
            self.best_distance = self.distance
        self.current_step += 1
        # reward -= self.current_step / self.max_episode_len
        done = self.current_step >= self.max_episode_len
        if done:
            self.final = self.initial_distance - self.best_distance
            reward = (self.initial_distance - self.distance) / self.initial_distance
        info = {}  # Placeholder for additional information
        index = cp.deepcopy(self.index)
        return index, reward, done, info

    def shuffle(self):
        self.current_step = 0
        random.seed()
        random.shuffle(self.index)
        self.distance = self.calculate_distance()
        self.initial_distance = cp.deepcopy(self.distance)

    def reset(self):

        self.index, self.locations = zip(*self.tsp_instance)
        self.index = list(self.index)
        self.locations = list(self.locations)
        self.distance_matrix = calculate_distance_matrix(self.locations, MAT_MUL)
        self.best_distance = np.inf
        index = cp.deepcopy(self.index)
        self.best_solution = cp.deepcopy(self.index)
        self.shuffle()
        return self.index, {idx: loc for idx, loc in zip(index, self.locations)}

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
