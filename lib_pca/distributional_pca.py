import ot
import torch

import numpy as np

from .utils_barycenters import cost_pca


class DistributionalPCA():
    """
        Distributional PCA (also called Uncertainty-Aware PCA) as proposed in [1, 2]. It amounts at taking the SVD of S=sum_i (m_i m_i^T + Sigma_i) - Bar(m)Bar(m)^T.

        [1] Niculae, V. (2023). Two derivations of Principal Component Analysis on datasets of distributions. arXiv preprint arXiv:2306.13503.

        [2] Görtler, J., Spinner, T., Streeb, D., Weiskopf, D., & Deussen, O. (2019). Uncertainty-aware principal component analysis. IEEE Transactions on Visualization and Computer Graphics, 26(1), 822-831.
    """
    def __init__(self, n_components=1):
        self.n_components = n_components

    def fit(self, data_means, data_covs):
        """
            Parameters
            ----------
            data_means : torch tensor of shape (n, d)
                Means of the data points
            data_covs : torch tensor of shape (n, d, d)
                Covariances matrices of the data points
        """
        # Mean to center full distributions (to be equivalent with PCA)
        center_means = torch.mean(data_means, axis=0, keepdims=True)
        data_means = data_means - center_means

        means_transformed = torch.einsum("ni, nj -> nij", data_means, data_means) # Shape (n, d, d)
        solution_matrix = torch.mean(means_transformed + data_covs, axis=0)
        
        eigenvals, eigenvecs = torch.linalg.eigh(solution_matrix)
        eigenvals[eigenvals < 0.0] = 0.0

        eigenvals_sorted, inds_sort = torch.sort(eigenvals, descending=True) 
        self.explained_variance_ = eigenvals_sorted[:self.n_components] / torch.sum(eigenvals_sorted)
        self.components_ = eigenvecs[:, inds_sort[:self.n_components]].T

    def fit_from_samples(self, X, y):
        """ 
            Fit from samples by first approximating the means and covariance matrices.

            Parameters
            ----------
            X : torch tensor of shape (n, d)
                Tensor of n data points
            y : LongTensor of shape (n,), values in 0...C-1, C: number of distributions
                Tensor containing to which distributions each point belongs
        """

        X_tensorized, X_weights_class, _ = pad_and_group_by_class(X, y)  # shape (n_distr, n_samples, d)
        means_X = torch.sum(X_weights_class[:, :, None] * X_tensorized, dim=1)

        Cs = X_tensorized - means_X[:, None]
        covs_X = torch.einsum(
            "lnd, lnk -> ldk", X_weights_class[:, :, None] * Cs, Cs
        )

        self.fit(means_X, covs_X)

    def cost_barycenter(self, samples_data, weights_data=None):
        n_distr = len(samples_data)

        support_bary = self.components_[:,:,None]

        if weights_data is None:
            weights_data = [torch.ones(m.shape[0]) / m.shape[0] for m in samples_data]  # weights each measure

        L_ot_costs = np.zeros(n_distr)
        for k in range(n_distr):
            C = cost_pca(support_bary, samples_data[k])
            b = weights_data[k]
            L_ot_costs[k] = ot.emd2(torch.tensor([1.]), b, C)

        return np.mean(L_ot_costs)


def pad_and_group_by_class(x, labels):
    """
    Args:
        x: Tensor of shape (n, d)
        labels: LongTensor of shape (n,), values in 0..C-1

    Returns:
        out: Tensor of shape (C, nc, d), with padded zeros
        weights_by_class: Tensor of shape (C, nc), with padded zeros or uniform weights
        weights_by_samples: Tensor of shape (C, nc), with padded zeroes or uniform weights
    """
    n, d = x.shape
    device = x.device
    dtype = x.dtype

    # Compute number of classes (C) and max samples per class (nc)
    C = int(labels.max().item()) + 1
    class_counts = torch.bincount(labels.type(torch.int), minlength=C)
    nc = class_counts.max().item()

    # Initialize output tensors
    out = torch.zeros(C, nc, d, device=device, dtype=dtype)
    weights_by_class = torch.zeros(C, nc, device=device, dtype=dtype)  # weights matrix, uniform by class
    weights_by_samples = torch.zeros(C, nc, device=device, dtype=dtype)  # weights matrix, uniform on the full set of samples

    for c in range(C):
        class_mask = (labels == c)
        class_x = x[class_mask]  # (num_samples_c, d)
        num_samples = class_x.size(0)

        out[c, :num_samples] = class_x
        weights_by_class[c, :num_samples] = 1.0 / class_counts[c]
        weights_by_samples[c, :num_samples] = 1.0 / len(labels)

    return out, weights_by_class, weights_by_samples
