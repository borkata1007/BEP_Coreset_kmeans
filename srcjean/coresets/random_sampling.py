import numpy as np


class RandomSampling:
    def __init__(self, m):
        self.m = m
        self.weights = None
        self.indices = None
        self.sensitivities = None

    def generate(self, data):
        n = data.shape[0]

        self.sensitivities = np.full(n, 1.0 / n)
        self.indices = np.random.choice(n, size=self.m, replace=False)
        self.weights = np.full(self.m, n / self.m)

        return data[self.indices]
