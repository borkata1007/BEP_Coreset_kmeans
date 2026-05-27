import numpy as np


class LightweightCoreset:
    def __init__(self, m):
        self.m = m
        self.weights = None
        self.indices = None
        self.sensitivities = None

    def generate(self, data):
        n = data.shape[0]

        mean_point = np.mean(data, axis=0)
        diffs = data - mean_point
        squared_distances = np.linalg.norm(diffs, axis=1) ** 2
        total_squared_distance = np.sum(squared_distances)

        uniform_term = 0.5 / n
        distance_term = 0.5 * squared_distances / total_squared_distance
        self.sensitivities = uniform_term + distance_term

        self.indices = np.random.choice(n, self.m, replace=True, p=self.sensitivities)
        sampled_sensitivities = self.sensitivities[self.indices]
        self.weights = 1.0 / (self.m * sampled_sensitivities)

        return data[self.indices]
