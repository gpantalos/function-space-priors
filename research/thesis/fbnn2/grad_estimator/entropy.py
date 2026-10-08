import torch


def entropy_gradients(optimizer, estimator, samples):
    dlog_q = estimator.compute_gradients(samples)
    surrogate_cost = torch.mean(torch.sum(torch.detach(-dlog_q) * samples, -1))
    grads_and_vars = optimizer.compute_gradients(surrogate_cost, var_list=var_list)
    return grads_and_vars


def entropy_surrogate(estimator, samples):
    """
    Surrogate of entropy - E_p / log p(x)

    :param estimator: Estimator f(x)
    :param samples: Number of samples
    """
    dlog_q = estimator.compute_gradients(samples)
    entropy_sur = torch.mean(torch.sum(torch.detach(-dlog_q) * samples, -1))
    return entropy_sur


def kl_surrogate(estimator, q_data, p_data):
    entropy_sur = entropy_surrogate(estimator, q_data)
    cross_entropy_gradients = estimator.compute_gradients(p_data, q_data)
    cross_entropy_sur = torch.mean(torch.sum(torch.detach(cross_entropy_gradients) * q_data, -1))
    return -entropy_sur - cross_entropy_sur


def minimize_entropy(optimizer, estimator, samples, var_list=None):
    """
    The distribution must be reparameterizable.
    The entropy will average over all dimensions before the last two dimensions.

    :param optimizer: A Tensorflow Optimizer.
    :param estimator: A ScoreEstimator.
    :param samples: A Tensor of shape [..., M, samples]
    :param var_list: A list of Variables.

    :return: A Tensorflow Operation.
    """
    dlog_q = estimator.compute_gradients(samples)
    backprop_loss = -dlog_q / torch.prod(samples.shape[:-1])
    opt_op = optimizer.minimize(samples, var_list=var_list, grad_loss=backprop_loss)
    return opt_op
