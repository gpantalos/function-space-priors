import numpy as np


def load_dataset(n=150):
    w0 = 0.125
    b0 = 5.
    x_range = [-20, 60]
    np.random.seed(43)

    def s(_x):
        g = (_x - x_range[0]) / (x_range[1] - x_range[0])
        return 3 * (0.25 + g ** 2.)

    x = (x_range[1] - x_range[0]) * np.random.rand(n) + x_range[0]
    eps = np.random.randn(n) * s(x)
    y = (w0 * x * (1. + np.sin(x)) + b0) + eps
    y = (y - y.mean()) / y.std()
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    return x[:, None], y[:, None]
