import torch

from core.grad_estimator import SpectralScoreEstimator, entropy_surrogate


class AbstractFVI:
    """
    Base Class for Functional Variational Inference.

    :param posterior: the posterior network to be optimized.
    :param rand_generator: Generates measurement points.
    :param obs_var: Float. Observation variance.
    :param input_dim. Int.
    :param n_rand. Int. Number of random measurement points.
    :param injected_noise: Float. Injected to function outputs for stability.
    :param likelihood: None or Class. If None: Gaussian(obs_var).
    """

    def __init__(self, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise, likelihood=None):

        self._rand_generator = rand_generator
        self.injected_noise = injected_noise
        self._likelihood = likelihood
        self.input_dim = input_dim
        self.posterior = posterior
        self.obs_var = obs_var
        self.n_rand = n_rand

        self.x = torch.empty(size=[None, self.input_dim])
        self.x_pred = torch.empty(size=[None, self.input_dim])
        self.y = torch.empty()
        self.n_particles = torch.empty()

        self.learning_rate_ph = 0
        self.coeff_ll = 0
        self.coeff_kl = 0
        self.kl_surrogate = 0

        self.build_rand()
        self.build_function()
        self.build_log_likelihood()
        self.build_evaluation()

    @property
    def batch_size(self):
        return len(self.x)

    def build_rand(self):
        self.rand = self._rand_generator(self)
        self.x_rand = torch.cat([self.x, self.rand], dim=0)

    def build_function(self):
        self.repeat_x_rand = torch.unsqueeze(self.x_rand, dim=0)  # , [self.n_particles, 1, 1])

        # [n_particles, batch_size + n_rand]
        self.func_x_rand = self.posterior(self.x_rand, self.n_particles)
        self.func_x = self.func_x_rand[:, :self.x.shape[0]]
        self.func_x_pred = self.posterior(self.x_pred, self.n_particles)

        self.noisy_func_x_rand = self.func_x_rand + self.injected_noise * torch.randn(self.func_x_rand.shape)

    def build_log_likelihood(self):
        y_obs = torch.unsqueeze(self.y, dim=0)  # , [self.n_particles, 1])
        if self._likelihood is not None:
            self.log_likelihood = torch.mean(self._likelihood.forward(self.func_x, y_obs))
            self.log_likelihood_sample = self._likelihood.forward(self.func_x, y_obs)
        else:
            y_x_dist = torch.distributions.Normal(self.func_x, self.obs_var ** 0.5)
            self.log_likelihood_sample = y_x_dist.log_prob(y_obs)
            self.log_likelihood = torch.mean(self.log_likelihood_sample)

    def build_evaluation(self):
        self.eval_rmse = torch.sqrt(torch.mean((torch.mean(self.func_x, 0) - self.y) ** 2))
        self.eval_lld = torch.mean(torch.logsumexp(self.log_likelihood_sample, 0) - torch.log(self.n_particles))

    @property
    def params_posterior(self):
        return self.posterior.parameters()

    @property
    def params_prior(self):
        return None

    @property
    def params_likelihood(self):
        return self._likelihood.parameters()

    def build_kl(self):
        raise NotImplementedError

    def build_optimizer(self):
        self.elbo = self.coeff_ll * self.log_likelihood - self.coeff_kl * self.kl_surrogate / self.batch_size

        self.optimizer_posterior = torch.optim.Adam(params=self.params_posterior, lr=self.learning_rate_ph)
        # self.infer_latent = self.optimizer_posterior.minimize(-self.elbo)

        self.optimizer_likelihood = torch.optim.Adam(params=self.params_likelihood, lr=self.learning_rate_ph)
        # self.infer_likelihood = self.optimizer_likelihood.minimize(-self.elbo, var_list=self.params_likelihood)

        # self.infer_joint = tf.group(self.infer_latent, self.infer_prior, self.infer_likelihood)


class KLEstimatorFVI(AbstractFVI):
    """
    Function Variational Inference with estimating the whole KL divergence term.
    """

    def __init__(self, prior_generator, posterior, rand_generator, obs_var,
                 input_dim, n_rand, injected_noise, likelihood=None,
                 n_eigen_threshold=0.99, eta=0.):
        super(KLEstimatorFVI, self).__init__(
            posterior, rand_generator, obs_var,
            input_dim, n_rand, injected_noise, likelihood=likelihood)
        self.prior_gen = prior_generator
        self.n_eigen_threshold = n_eigen_threshold
        self.eta = eta

        self.build_kl()
        self.build_optimizer()

    def build_kl(self):
        # estimate entropy surrogate
        estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
        entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)

        # estimate cross entropy
        self.prior_func_x_rand = self.prior_gen(self.x_rand, self.n_particles)
        self.noisy_prior_func_x_rand = self.prior_func_x_rand + self.injected_noise * torch.randn_like(
            self.prior_func_x_rand)
        cross_entropy_gradients = estimator.compute_gradients(self.noisy_prior_func_x_rand, self.noisy_func_x_rand)
        cross_entropy_sur = -torch.mean(torch.sum(torch.detach(cross_entropy_gradients) * self.noisy_func_x_rand, -1))
        self.kl_surrogate = -entropy_sur + cross_entropy_sur

# class EntropyEstimationFVI(AbstractFVI):
#     """
#     Function Variational Inference with estimating entropy and computing cross entropy analytically.
#     """
#
#     def __init__(self, prior_kernel, posterior, rand_generator, obs_var,
#                  input_dim, n_rand, injected_noise, likelihood=None,
#                  n_eigen_threshold=0.99, eta=0.):
#         super(EntropyEstimationFVI, self).__init__(
#             posterior, rand_generator, obs_var,
#             input_dim, n_rand, injected_noise, likelihood=likelihood)
#         self.n_eigen_threshold = n_eigen_threshold
#         self.eta = eta
#
#         self.prior_kernel = prior_kernel
#
#         self.build_kl()
#         self.build_optimizer()
#
#     def build_kl(self):
#         # estimate entropy surrogate
#         estimator = SpectralScoreEstimator(eta=self.eta, n_eigen_threshold=self.n_eigen_threshold)
#         entropy_sur = entropy_surrogate(estimator, self.noisy_func_x_rand)
#
#         # compute analytic cross entropy
#         kernel_matrix = self.prior_kernel.K(self.x_rand) + self.injected_noise ** 2 * torch.eye(self.x_rand.shape[0])
#         prior_dist = torch.distributions.Normal(torch.zeros(self.x_rand.shape[0]), kernel_matrix)
#         cross_entropy = -torch.mean(prior_dist.log_prob(self.noisy_func_x_rand))
#
#         self.kl_surrogate = -entropy_sur + cross_entropy
#
#     def build_prior_gp(self, init_var=0.1, inducing_points=None):
#         self.x_gp = torch.empty([None, self.input_dim])
#         self.y_gp = torch.empty()
#         self.x_pred_gp = torch.empty([None, self.input_dim])
#         # with tf.variable_scope('prior'):
#         #     if inducing_points is None:
#         #         self.gp = gfs.models.GPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel,
#         #                                  obs_var=init_var)
#         #     else:
#         #         self.gp = gfs.models.SGPR(self.x_gp, tf.expand_dims(self.y_gp, 1), kern=self.prior_kernel,
#         #                                   Z=inducing_points)
#         #     self.gp_loss = self.gp.objective
#
#         self.gp_var = self.gp.likelihood.variance
#         self.gp_logstd = torch.log(self.gp.likelihood.variance) * 0.5
#         self.func_x_pred_gp = torch.squeeze(self.gp.predict_f_samples(self.x_pred_gp, self.n_particles), -1)
#
#         # self.optimizer_gp_prior = torch.optim.Adam(self.params_prior, self.learning_rate_ph)
#         self.optimizer_gp_likelihood = torch.optim.Adam(self.params_likelihood, self.learning_rate_ph)
#         # only optimize kernel params without optimizing GP observation variance
#         # self.infer_gp = self.optimizer_gp.minimize(self.gp_loss, var_list=)
#         # self.infer_gp_kern = self.optimizer_gp.minimize(self.gp_loss, var_list=self.params_likelihood)
