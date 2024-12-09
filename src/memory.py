import numpy as np
import torch


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.temp = []
        self.dones = []
        self.episodes = []
        self.locations = {}
        self.matrix = {}

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            torch.tensor(self.probs, dtype=torch.float32),
            torch.tensor(self.vals, dtype=torch.float32),
            np.array(self.rewards),
            np.array(self.temp),
            np.array(self.dones),
            np.array(self.episodes),
            batches,
        )

    def get_locations(self, episode):
        return self.locations[int(episode)]

    def get_matrix(self, episode):
        return self.matrix[int(episode)]

    def store_memory(
        self, state, action, probs, vals, reward, temp, done, current_episode
    ):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.temp.append(temp)
        self.dones.append(done)
        self.episodes.append(current_episode)

    def store_location(self, location, index):
        self.locations[index] = location

    def store_matrix(self, matrix, index):
        self.matrix[index] = matrix

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.temp = []
        self.dones = []
        self.vals = []
        self.episodes = []
        self.locations = {}
        self.matrix = {}
