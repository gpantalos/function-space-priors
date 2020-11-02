import numpy as np
import torch
from matplotlib import pyplot as plt
# from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from tqdm import trange


def load_data():
    x_train = torch.linspace(-5, 5, 10000)
    y_train = torch.sin(x_train)

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    return dataset_train


class Densenet(torch.nn.Module):
    """
    Simple module implementing a feedforward neural network with
    num_layers layers of size width and input of size input_size.
    """

    def __init__(self, input_size, num_layers, width):
        super().__init__()
        input_layer = torch.nn.Sequential(nn.Linear(input_size, width),
                                          nn.ReLU())
        hidden_layers = [nn.Sequential(nn.Linear(width, width),
                                       nn.ReLU()) for _ in range(num_layers)]
        output_layer = torch.nn.Linear(width, 10)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out

    def predict_class_probs(self, x):
        probs = F.softmax(self.forward(x), dim=1)
        return probs


class Gaussian:
    def __init__(self, mu, logsigma):
        super().__init__()
        self.mu = mu
        self.logsigma = logsigma
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return F.softplus(self.logsigma)

    def sample(self):
        epsilon = self.normal.sample(self.mu.size())
        return self.mu + self.sigma * epsilon


class BayesianLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.prior_mu = 0
        self.prior_sigma = -3

        self.weight_mu = nn.Parameter(torch.ones(output_dim, input_dim) * self.prior_mu)
        self.weight_logsigma = nn.Parameter(torch.ones(output_dim, input_dim) * self.prior_sigma)
        self.weight = Gaussian(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.ones(output_dim) * self.prior_mu)
            self.bias_logsigma = nn.Parameter(torch.ones(output_dim) * self.prior_sigma)
            self.bias = Gaussian(self.bias_mu, self.bias_logsigma)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_logsigma', None)

    def forward(self, x):
        w = self.weight.sample()
        if self.use_bias:
            b = self.bias.sample()
        else:
            b = None
        return F.linear(x, w, b)

    def kl_divergence(self):
        """
        Computes the KL divergence between the priors and posteriors for this layer.
        """
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss += self._kl_divergence(self.bias_mu, self.bias_logsigma)
        return kl_loss

    def _kl_divergence(self, mu, logsigma):
        """
        Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        """
        post_mu = mu
        post_sigma = F.softplus(logsigma)

        prior_mu = torch.ones_like(post_mu) * self.prior_mu
        prior_sigma = F.softplus(torch.ones_like(post_sigma) * self.prior_sigma)

        kl = torch.log(post_sigma / prior_sigma) + ((post_mu - prior_mu) ** 2 + prior_sigma ** 2) / (
                2 * post_sigma ** 2)

        return kl.mean()


class BayesNet(torch.nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    """

    def __init__(self, input_size, num_layers, width):
        super().__init__()
        input_layer = torch.nn.Sequential(BayesianLayer(input_size, width),
                                          nn.ReLU())
        hidden_layers = [nn.Sequential(BayesianLayer(width, width),
                                       nn.ReLU()) for _ in range(num_layers)]
        output_layer = nn.Sequential(BayesianLayer(width, 10))
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_class_probs(self, x, num_forward_passes=10):
        assert x.shape[1] == 28 ** 2
        batch_size = x.shape[0]
        passes = torch.empty((num_forward_passes, batch_size, 10))
        for i in range(num_forward_passes):
            passes[i] = F.softmax(self.forward(x), dim=1)
        probs = passes.mean(dim=0)
        assert probs.shape == (batch_size, 10)
        return probs

    def kl_loss(self):
        """
        Computes the KL divergence loss for all layers.
        """
        losses = []
        layers = self.net
        for layer in layers:
            bayesian_layer = layer[0]
            loss = bayesian_layer.kl_divergence()
            losses.append(loss)
        loss = sum(losses)
        return loss.squeeze()


def train_network(model, optimizer, train_loader, num_epochs=100, pbar_update_interval=100):
    criterion = torch.nn.CrossEntropyLoss()

    pbar = trange(num_epochs)
    for _ in pbar:
        for k, (batch_x, batch_y) in enumerate(train_loader):
            model.zero_grad()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            if type(model) == BayesNet:
                bayes_loss = model.kl_loss() / len(train_loader)
                loss += bayes_loss
            loss.backward()
            optimizer.step()

            if k % pbar_update_interval == 0:
                acc = (model(batch_x).argmax(axis=1) == batch_y).sum().float() / (len(batch_y))
                pbar.set_postfix(loss=loss.item(), acc=acc.item())


def evaluate_model(model, model_type, test_loader, batch_size, extended_eval, private_test):

    for batch_x, batch_y in test_loader:
        pred = model.predict_class_probs(batch_x)

    acc_mean = np.mean(accs_test)


def main(test_loader=None, private_test=False):
    num_epochs = 100  # You might want to adjust this
    batch_size = 512  # Try playing around with this
    print_interval = 100
    learning_rate = 5e-4  # Try playing around with this
    model_type = "bayesnet"  # Try changing this to "densenet" as a comparison
    extended_evaluation = True  # Set this to True for additional model evaluation

    dataset_train = load_data()
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)

    if model_type == "bayesnet":
        model = BayesNet(input_size=784, num_layers=2, width=100)
    elif model_type == "densenet":
        model = Densenet(input_size=784, num_layers=2, width=100)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_network(model, optimizer, train_loader, num_epochs=num_epochs, pbar_update_interval=print_interval)
    evaluate_model(model, model_type, test_loader, batch_size, extended_evaluation, private_test)
    return predictions


if __name__ == "__main__":
    main()
