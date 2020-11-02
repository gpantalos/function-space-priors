import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_dim, output_dim, bias=True):
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
