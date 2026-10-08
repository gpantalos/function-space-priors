from utils.gpr import *


def main():
    kernels = ["RBF", "linear", "quadratic", "Matern-1/2", "Matern-3/2", "Matern-5/2", "Cosine"]
    compositions = ["addition", "product"]
    lengthscale = 1.
    outputscale = 1.
    noise = 0.1
    kernel = kernels[0]
    composition = compositions[0]
    num_training = 25

    train_x = (torch.rand(num_training) - 0.5) * 10
    train_y = regression_function(train_x)
    test_x = torch.linspace(-6, 6, 1000)

    kernel = get_kernel(kernel, composition)
    model = ExactGP(train_x, train_y, kernel)

    # Set hyper-parameters
    model.length_scale = lengthscale
    model.output_scale = outputscale
    model.likelihood.noise = torch.tensor([noise])

    # Evaluate GP Model.
    plot_model(model, train_x, train_y, test_x)
