import gpytorch
import numpy as np
import torch

from core.grad_estimator.entropy import entropy_surrogate
from core.grad_estimator.spectral import SSGE


class AbstractFVI:
    """
    Base Class for Functional Variational Inference.
    """

    def __init__(self, posterior, rand_generator, obs_var, input_dim, injected_noise, likelihood=None):
        """
        :param posterior: the posterior network to be optimized.
        :param rand_generator: Generates measurement points.
        :param obs_var: Float. Observation variance.
        :param input_dim. Int.
        :param injected_noise: Float. Injected to function outputs for stability.
        :param likelihood: None or Class. If None: Gaussian(obs_var).
        """
        # todo: learning rate
        self.learning_rate = 1e-3

        self.rand_generator = rand_generator
        self.likelihood = likelihood
        self.posterior = posterior
        self.obs_var = obs_var

        self.injected_noise = injected_noise
        self.input_dim = input_dim

        self.x_rand = torch.cat([self.x, self.rand_generator()], dim=0)
        self.func_x_rand = self.posterior(self.x_rand)
        self.func_x_pred = self.posterior(self.x_pred)
        self.func_x = self.func_x_rand[:, :self.batch_size]
        self.noisy_func_x_rand = self.func_x_rand + self.injected_noise * torch.rand(np.shape(self.func_x_rand))

        y_obs = self.y
        if self.likelihood is not None:
            self.log_likelihood = torch.mean(self.likelihood.forward(self.func_x, y_obs))
            self.log_likelihood_sample = self.likelihood.forward(self.func_x, y_obs)
        else:
            y_x_dist = torch.distributions.Normal(self.func_x, self.obs_var ** 0.5)
            self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
            self.log_likelihood = torch.mean(self.log_likelihood_sample)

        self.eval_rmse = torch.sqrt(torch.mean((torch.mean(self.func_x, 0) - self.y) ** 2))
        self.eval_lld = torch.mean(torch.logsumexp(self.log_likelihood_sample, 0))

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * self.kl_surrogate / self.batch_size
        self.optimizer = torch.optim.Adam(self.learning_rate)
        self.infer_latent = self.optimizer.minimize(-self.elbo, var_list=self.params_posterior)
        self.infer_likelihood = self.optimizer.minimize(-self.elbo, var_list=self.params_likelihood)
        self.infer_joint = torch.stack(self.infer_latent, self.infer_prior, self.infer_likelihood)

    @property
    def batch_size(self):
        return len(self.x)

    @property
    def params_posterior(self):
        return self.posterior.parameters()

    @property
    def params_likelihood(self):
        return self.likelihood.parameters()


class EntropyEstimationFVI(AbstractFVI):
    """
    Function Variational Inference with estimating entropy and computing cross entropy analytically.
    """

    def __init__(self, prior_kernel, posterior, rand_generator, obs_var, input_dim, injected_noise, likelihood=None, n_eigen_threshold=0.99, eta=0.):
        """
        :param prior_kernel: Gpytorch kernel for the prior.
        :param posterior: Bayesian Neural Network for the posterior.
        :param rand_generator: Sampler from uniform distribution.
        :param obs_var: Observation variance.
        :param input_dim: Input dimension.
        :param injected_noise: Sampler from uniform distribution.
        :param likelihood: Sampler from uniform distribution.
        """
        super().__init__(posterior, rand_generator, obs_var, input_dim, injected_noise, likelihood=likelihood)
        self.n_eigen_threshold = n_eigen_threshold
        self.prior_kernel = prior_kernel
        self.eta = eta

        # estimate entropy surrogate
        estimator = SSGE(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)

        # compute analytic cross entropy
        kernel_matrix = self.prior_kernel(self.x_rand).evaluate() + self.injected_noise ** 2 * torch.eye(len(self.x_rand))
        prior_dist = torch.distributions.MultivariateNormal(torch.zeros(len(self.x_rand)), kernel_matrix)
        cross_entropy = -torch.mean(prior_dist.log_prob(self.noisy_func_x_rand))
        self.kl_surrogate = -entropy_sur + cross_entropy

        self.build_optimizer()

    def build_prior_gp(self, x, y):
        model = GPR(x, y, self.prior_kernel)
        optimizer = torch.optim.Adam(model.parameters(), self.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
        for _ in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
        self.prior_gp = model


class GPR(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, prior_kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = prior_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
