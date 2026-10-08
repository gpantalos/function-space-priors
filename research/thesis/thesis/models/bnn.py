import matplotlib.pyplot as plt
import torch
from tqdm import trange

from data.data import load_dataset
from utils.bnn import loss_fn, BayesianNeuralNetwork


def train(model, x_train, y_train, epochs=1000, lr=0.001):
    # train
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in trange(epochs):
        optim.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train, model)
        loss.backward()
        optim.step()


def evaluate(model, x_test, y_test, n_samples=100):
    model.eval()
    # draw samples from Q(theta)
    with torch.no_grad():
        y_pred = torch.stack([model(x_test) for _ in range(n_samples)]).squeeze()
    q1 = torch.quantile(y_pred, 0.05, 0)
    mu = torch.quantile(y_pred, 0.5, 0)
    q2 = torch.quantile(y_pred, 0.95, 0)

    plt.figure(figsize=(10, 8))
    plt.plot(x_test, mu)
    plt.fill_between(torch.flatten(x_test), q1, q2, alpha=0.2)
    plt.scatter(x_test, y_test)
    fig = plt.gcf()
    fig.show()


def main():
    torch.set_default_tensor_type(torch.DoubleTensor)
    X, Y = load_dataset()
    X = torch.tensor(X)
    Y = torch.tensor(Y)

    model = BayesianNeuralNetwork(in_size=1, hidden_size=20, out_size=1, n_batches=1)

    train(model, X, Y)
    evaluate(model, X, Y)
