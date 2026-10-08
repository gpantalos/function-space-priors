import gpytorch
import torch
import matplotlib.pyplot as plt


def get_kernel(kernel, composition):
    base_kernel = []
    if "RBF" in kernel:
        base_kernel.append(gpytorch.kernels.RBFKernel())
    if "linear" in kernel:
        base_kernel.append(gpytorch.kernels.LinearKernel())
    if "quadratic" in kernel:
        base_kernel.append(gpytorch.kernels.PolynomialKernel(power=2))
    if "Matern-1/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=1 / 2))
    if "Matern-3/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=3 / 2))
    if "Matern-5/2" in kernel:
        base_kernel.append(gpytorch.kernels.MaternKernel(nu=5 / 2))
    if "Cosine" in kernel:
        base_kernel.append(gpytorch.kernels.CosineKernel())
    if composition == "addition":
        base_kernel = gpytorch.kernels.AdditiveKernel(*base_kernel)
    elif composition == "product":
        base_kernel = gpytorch.kernels.ProductKernel(*base_kernel)
    return gpytorch.kernels.ScaleKernel(base_kernel)


def regression_function(x, noise=1e-1):
    """Get function value."""
    return torch.sin(2 * x) / x + noise * torch.randn(len(x))


class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel):
        super().__init__(train_x, train_y, likelihood=gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        """Forward computation of GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def output_scale(self):
        """Get output scale."""
        return self.covar_module.outputscale

    @output_scale.setter
    def output_scale(self, value):
        """Set output scale."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor([value])
        self.covar_module.outputscale = value

    @property
    def length_scale(self):
        """Get length scale."""
        ls = self.covar_module.base_kernel.kernels[0].lengthscale
        if ls is None:
            ls = torch.tensor(0.0)
        return ls

    @length_scale.setter
    def length_scale(self, value):
        """Set length scale."""
        self.covar_module.lengthscale = value
        self.covar_module.base_kernel.lengthscale = value
        for kernel in self.covar_module.base_kernel.kernels:
            kernel.lengthscale = value


def plot_model(model, train_x, train_y, test_x, inducing_points=None, plot_points=True):
    model.eval()
    with torch.no_grad():
        out = model(test_x)
        lower, upper = out.confidence_region()

    if plot_points:
        plt.plot(train_x, train_y, 'k*', label='Train Data')

    test_y = regression_function(test_x, noise=0).detach()
    plt.plot(test_x, test_y, 'k-', label='Noise-free Function')

    plt.plot(test_x, out.mean, 'b-', label='Mean Prediction')
    plt.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), color='b', alpha=0.2, label='Predictive Distribution')

    plt.ylim([-2, 3.])
    if inducing_points is not None:
        plt.plot(
            inducing_points,
            torch.zeros_like(inducing_points),
            'r*',
            label='inducing_points'
        )
    plt.legend(loc='upper left')
    plt.show()
