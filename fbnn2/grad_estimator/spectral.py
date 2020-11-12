import torch

from .base import ScoreEstimator


class SSGE(ScoreEstimator):
    def __init__(self, n_eigen=None, eta=None, n_eigen_threshold=None):
        super().__init__()
        self._n_eigen_threshold = n_eigen_threshold
        self._n_eigen = n_eigen
        self._eta = eta

    def nystrom_ext(self, samples, x, eigen_vectors, eigen_values, kernel_width):
        """
        :param samples: [..., M, x_dim].
        :param x: [..., N, x_dim]
        :param eigen_vectors: [..., M, n_eigen]
        :param eigen_values: [..., n_eigen]
        :param kernel_width: [..., n_eigen]
        :returns: [..., N, n_eigen], by default n_eigen=M.
        """
        M = samples.shape[-2]
        # Kxq: [..., N, M]
        Kxq = self.gram(x, samples, kernel_width)
        # ret: [..., N, n_eigen]
        ret = torch.sqrt(M) * Kxq @ eigen_vectors / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def compute_gradients(self, samples, x=None):
        """
        :param samples: [..., M, x_dim]
        :param x: [..., N, x_dim]
        """
        if x is None:
            kernel_width = self.heuristic_kernel_width(samples, samples)
            x = samples
        else:
            # _samples: [..., N + M, x_dim]
            _samples = torch.cat([samples, x], dim=-2)
            kernel_width = self.heuristic_kernel_width(_samples, _samples)

        M = samples.shape[-2]
        # Kq: [..., M, M]
        # grad_K1: [..., M, M, x_dim]
        # grad_K2: [..., M, M, x_dim]
        Kq, grad_K1, grad_K2 = self.grad_gram(samples, samples, kernel_width)
        if self._eta is not None:
            Kq += self._eta * torch.eye(M)
        # eigen_vectors: [..., M, M]
        # eigen_values: [..., M]
        eigen_values, eigen_vectors = torch.eig(Kq, eigenvectors=True)
        if (self._n_eigen is None) and (self._n_eigen_threshold is not None):
            eigen_arr = torch.mean(torch.reshape(eigen_values, [-1, M]), dim=0)
            eigen_arr = torch.flip(eigen_arr, dims=[-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            self._n_eigen = torch.sum(torch.less(eigen_cum, self._n_eigen_threshold))
        if self._n_eigen is not None:
            # eigen_values: [..., n_eigen]
            # eigen_vectors: [..., M, n_eigen]
            # eigen_values, eigen_vectors = self.pick_n_eigen(eigen_values, eigen_vectors, self._n_eigen)
            eigen_values = eigen_values[..., -self._n_eigen:]
            eigen_vectors = eigen_vectors[..., -self._n_eigen:]
        # eigen_ext: [..., N, n_eigen]
        eigen_ext = self.nystrom_ext(samples, x, eigen_vectors, eigen_values, kernel_width)
        # grad_K1_avg = [..., M, x_dim]
        grad_K1_avg = torch.mean(grad_K1, dim=-3)
        # beta: [..., n_eigen, x_dim]
        beta = -torch.sqrt(M) * eigen_vectors.T @ grad_K1_avg / torch.unsqueeze(eigen_values, -1)
        # grads: [..., N, x_dim]
        grads = eigen_ext @ beta
        return grads
