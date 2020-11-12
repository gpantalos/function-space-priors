import torch
import torch.nn as nn


class VI(nn.Module):
    def __init__(self):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(1, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    @staticmethod
    def reparameterize(mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var


def elbo(y, y_pred, mu, log_var):
    """Analytical elbo between two gaussians."""
    reconstruction_error = (0.5 * (y - y_pred) ** 2).sum()
    kl_divergence = (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp()))
    return -(reconstruction_error + kl_divergence).sum()
