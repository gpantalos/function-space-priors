import torch


class ScoreEstimator:
    def __init__(self):
        pass

    @staticmethod
    def rbf_kernel(x1, x2, kernel_width):
        return torch.exp(-torch.sum(torch.square((x1 - x2) / kernel_width), dim=-1) / 2)

    def gram(self, x1, x2, kernel_width):
        # x1: [..., n1, x_dim]
        # x2: [..., n2, x_dim]
        # kernel_width: [..., 1, 1, x_dim]
        # return: [..., n1, n2]
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        return self.rbf_kernel(x_row, x_col, kernel_width)

    def grad_gram(self, x1, x2, kernel_width):
        # x1: [..., n1, x_dim]
        # x2: [..., n2, x_dim]
        # kernel_width: [..., 1, 1, x_dim]
        # return gram, grad_x1, grad_x2:
        #   [..., n1, n2], [..., n1, n2, x_dim], [..., n1, n2, x_dim]
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        # G: [..., n1, n2]
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        # diff: [..., n1, n2, n_x]
        diff = (x_row - x_col) / (kernel_width ** 2)
        # G_expand: [..., n1, n2, 1]
        G_expand = torch.unsqueeze(G, dim=-1)
        # grad_x1: [..., n1, n2, n_x]
        grad_x2 = G_expand * diff
        # grad_x2: [..., n1, n2, n_x]
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    @staticmethod
    def heuristic_kernel_width(x_samples, x_basis):
        # x_samples: [..., n_samples, x_dim]
        # x_basis: [..., n_basis, x_dim]
        # return: [..., 1, 1, x_dim]
        x_dim = x_samples.shape[-1]
        n_samples = x_samples.shape[-2]
        n_basis = x_basis.shape[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.shape)
        reshape_dims = list(range(length - 3)) + [length - 1, length - 3, length - 2]
        pairwise_dist = torch.transpose(pairwise_dist, dim0=reshape_dims[0], dim1=reshape_dims[1])

        k = n_samples * n_basis // 2
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, x_dim, n_samples * n_basis]), k).values
        kernel_width = torch.reshape(top_k_values[:, :, -1], [x_samples.shape[:-2], [1, 1, x_dim]])
        kernel_width = kernel_width * (x_dim ** 0.5)
        # kernel_width = kernel_width + kernel_width < 1e-6
        return kernel_width.detach()

    def compute_gradients(self, samples, x=None):
        raise NotImplementedError()
