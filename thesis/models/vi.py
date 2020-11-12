import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from data.data import load_dataset
from utils.vi import *


def main():
    X, Y = load_dataset()
    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.float)

    model = VI()
    epochs = 1000
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in trange(epochs):
        optim.zero_grad()
        y_pred, mu, log_var = model(X)
        loss = -elbo(y_pred, Y, mu, log_var)
        loss.backward()
        optim.step()

    # draw samples from Q(theta)
    n_samples = 100
    with torch.no_grad():
        y_pred = torch.cat([model(X)[0] for _ in range(n_samples)], dim=1)
    q1, mu, q2 = np.quantile(y_pred, [0.05, 0.5, 0.95], axis=1)
    plt.figure(figsize=(15, 8))
    plt.scatter(X, Y)
    plt.plot(X, mu)
    plt.fill_between(torch.flatten(X), q1, q2, alpha=0.2)
    plt.show()
