import torch


def get_posterior(layer_sizes):
    def bnn(x):
        """
        Implements a forward pass through a Bayesian Neural Network
        """
        # Initialize h
        h = x
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # Weights
            w_mean = torch.Tensor([n_in, n_out])
            w_std = torch.exp(torch.Tensor([n_in, n_out]))
            ws = w_mean + w_std * torch.rand([n_in, n_out])
            # Biases
            b_mean = torch.Tensor([1, n_out])
            b_std = torch.exp(torch.Tensor([1, n_out]))
            bs = b_mean + b_std * torch.rand([1, n_out])
            # Update h
            h = h @ ws + bs
            # Activation
            if i < len(layer_sizes) - 2:
                h = torch.relu(h)
        return h.squeeze()
    return bnn

