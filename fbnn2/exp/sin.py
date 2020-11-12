import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt

from core.fvi import EntropyEstimationFVI
from data.toy import SinDataset
from utils.nets import get_posterior
from utils.utils import default_plotting_new as init_plotting

injected_noise = 0.01
init_logstd = -5.
n_rand = 20
n_hidden = 2
n_units = 100
learning_rate = 0.001
epochs = 10000
n_eigen_threshold = 0.99
train_samples = 100
test_samples = 100
print_interval = 100
test_interval = 2000

# Load and normalize data
dataset = SinDataset()
original_x_train, original_y_train = dataset.train_samples()
mean_x, std_x = np.mean(original_x_train), np.std(original_x_train)
mean_y, std_y = np.mean(original_y_train), np.std(original_y_train)
train_x = (original_x_train - mean_x) / std_x
train_y = (original_y_train - mean_y) / std_y
original_x_test, original_y_test = dataset.test_samples()
test_x = (original_x_test - mean_x) / std_x
test_y = (original_y_test - mean_y) / std_y
y_logstd = np.log(dataset.y_std / std_y)
lower_ap = (dataset.x_min - mean_x) / std_x
upper_ap = (dataset.x_max - mean_x) / std_x


def rand_generator():
    dist = torch.distributions.Uniform(lower_ap, upper_ap)
    return dist.sample(n_rand)


# Setup model
prior_kernel = gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
layer_sizes = [1] + [n_units] * n_hidden + [1]
posterior = get_posterior(layer_sizes)
obs_var = torch.exp(2. * y_logstd)
input_dim = 1
init_var = np.exp(2 * y_logstd)

model = EntropyEstimationFVI(prior_kernel, posterior, rand_generator, obs_var, input_dim, injected_noise)
model.build_prior_gp(init_var)

# Training
gp_epochs = 10000
for epoch in range(gp_epochs):
    feed_dict = {model.x_gp: train_x,
                 model.y_gp: train_y,
                 model.learning_rate: 0.003}
    loss = sess.run([model.infer_gp_kern, model.gp_loss], feed_dict=feed_dict)
    if epoch % print_interval == 0:
        print('>>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f}'.format(epoch, gp_epochs, loss))

for epoch in range(epochs):
    feed_dict = {model.x: train_x, model.y: train_y, model.learning_rate: learning_rate}
    elbo_sur, kl_sur, logll = sess.run([model.elbo, model.kl_surrogate, model.log_likelihood], feed_dict=feed_dict)
    if epoch % print_interval == 0:
        print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(
            epoch, epochs, elbo_sur, logll, kl_sur))

    if epoch % test_interval == 0:
        y_pred = sess.run(model.func_x_pred, feed_dict={model.x_pred: np.reshape(test_x, [-1, 1])})
        y_pred = y_pred * std_y + mean_y
        mean_y_pred, std_y_pred = np.mean(y_pred, 0), np.std(y_pred, 0)

        plt.clf()
        figure = plt.figure(figsize=(8, 5.5), facecolor='white')
        init_plotting()

        plt.plot(original_x_test.squeeze(), original_y_test, 'g', label="True function")
        plt.plot(original_x_test.squeeze(), mean_y_pred, 'steelblue', label='Mean function')
        for i in range(5):
            plt.fill_between(original_x_test.squeeze(), mean_y_pred - i * 0.75 * std_y_pred,
                             mean_y_pred - (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
            plt.fill_between(original_x_test.squeeze(), mean_y_pred + i * 0.75 * std_y_pred,
                             mean_y_pred + (i + 1) * 0.75 * std_y_pred, linewidth=0.0,
                             alpha=1.0 - i * 0.15, color='lightblue')
        plt.scatter(original_x_train, original_y_train, c='tomato', zorder=10, label='Observations')
        plt.grid(True)
        plt.tick_params(axis='both', bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        plt.tight_layout()
        plt.ylim([dataset.y_min, dataset.y_max])
        plt.tight_layout()

        plt.savefig('results/{}/plot_epoch{}.pdf'.format(dataset, epoch))
