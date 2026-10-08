import numpy as np


class SinDataset:
    def __init__(self):
        self.x_min = -5
        self.x_max = 5
        self.y_min = -3.5
        self.y_max = 3.5
        self.confidence_coeff = 1.
        self.y_std = 2e-1

    @staticmethod
    def f(x):
        return 2 * np.sin(4 * x)

    def train_samples(self):
        np.random.seed(3)
        X_train1 = np.random.uniform(-2, -0.5, (10, 1))
        X_train2 = np.random.uniform(0.5, 2, (10, 1))
        X_train = np.concatenate([X_train1, X_train2], axis=0)
        epsilon = np.random.normal(0, self.y_std, (20, 1))
        y_train = np.squeeze(self.f(X_train) + epsilon)
        return X_train, y_train

    def test_samples(self):
        inputs = np.linspace(self.x_min, self.x_max, 1000)
        return inputs, self.f(inputs)
