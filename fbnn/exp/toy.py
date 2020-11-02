import argparse

# import matplotlib
import gpytorch
import numpy as np
import torch
# matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.fvi import KLEstimatorFVI
from data import x3_gap_toy, sin_toy
from utils.logging import get_logger
from utils.nets import BayesNet
from utils.utils import default_plotting_new as init_plotting

parser = argparse.ArgumentParser('Toy')
parser.add_argument('-d', '--dataset', type=str, default='sin')
parser.add_argument('-in', '--injected_noise', type=float, default=0.01)
parser.add_argument('-il', '--init_logstd', type=float, default=-5.)
parser.add_argument('-na', '--n_rand', type=int, default=20)
parser.add_argument('-nh', '--n_hidden', type=int, default=2)
parser.add_argument('-nu', '--n_units', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=10000)
parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
parser.add_argument('--train_samples', type=int, default=100)
parser.add_argument('--test_samples', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=2000)
args = parser.parse_args()
logger = get_logger(args.dataset, 'results/%s/' % args.dataset, __file__)
# print = logger.info

# load and normalize data
dataset = dict(x3=x3_gap_toy, sin=sin_toy)[args.dataset]()
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

# setup FBNN model
prior_kernel = gpytorch.kernels.PeriodicKernel() + gpytorch.kernels.RBFKernel()
if args.dataset == 'x3':
    prior_kernel = gpytorch.kernels.LinearKernel() + gpytorch.kernels.RBFKernel()


def rand_generator():
    return lower_ap + torch.rand((args.n_rand, 1)) * (upper_ap - lower_ap)


posterior = BayesNet(input_size=1, num_layers=args.n_hidden, width=args.n_units)
obs_var = torch.exp(2. * y_logstd)
model = KLEstimatorFVI(prior_kernel, posterior, rand_generator, obs_var, 1, args.n_rand, args.injected_noise)
model.build_kl()

# training
model.x_gp = train_x
model.y_gp = train_y
model.learning_rate_ph = 3e-3
gp_epochs = 10000
for epoch in range(gp_epochs):
    # todo: compute pretraining loss
    if epoch % args.print_interval == 0:
        print('>>> Pretrain GP Epoch {:5d}/{:5d}: Loss={:.5f}'.format(epoch, gp_epochs, loss))

    # todo: compute elbo, log likelihood and kl divergence
    if epoch % args.print_interval == 0:
        print('>>> Epoch {:5d}/{:5d} | elbo_sur={:.5f} | logLL={:.5f} | kl_sur={:.5f}'.format(epoch, args.epochs, elbo_sur, logll, kl_sur))

    if epoch % args.test_interval == 0:
        # todo: test the model
        y_pred = ?
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

plt.savefig('results/{}/plot_epoch{}.pdf'.format(args.dataset, epoch))
