import torch

from .utils_barycenters import cost_barycenter
from .utils_data import pad_and_group_by_class


class UAPCA():
    """
        Uncertainty-Aware PCA (also called Distributional PCA in [1]) as proposed in [1, 2]. It amounts at taking the SVD of S=sum_i (m_i m_i^T + Sigma_i) - Bar(m)Bar(m)^T.

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
        self.mean_ = center_means

        means_transformed = torch.einsum("ni, nj -> nij", data_means, data_means) # Shape (n, d, d)
        solution_matrix = torch.mean(means_transformed + data_covs, axis=0)
        
        eigenvals, eigenvecs = torch.linalg.eigh(solution_matrix)
        eigenvals[eigenvals < 0.0] = 0.0

        eigenvals_sorted, inds_sort = torch.sort(eigenvals, descending=True) 
        self.explained_variance_ratio_ = eigenvals_sorted[:self.n_components] / torch.sum(eigenvals_sorted)
        self.components_ = eigenvecs[:, inds_sort[:self.n_components]] # Shape (d, n_components)



    def fit_from_samples(self, X_list):
        """ 
            Fit from samples by first approximating the means and covariance matrices.

            Parameters
            ----------
            X_list : list of torch tensors of shape (n_k, d)
                List of K tensors of n_k data points
        """
        means_X = torch.stack([torch.mean(X, axis=0) for X in X_list], axis=0)
        covs_X = torch.stack([torch.cov(X.T) for X in X_list], axis=0)

        self.fit(means_X, covs_X)


    def fit_from_samples_labels(self, X, y):
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
        support_bary = self.components_[None] # Dirac, hence shape (1, d, n_components)
        return cost_barycenter(support_bary, samples_data, weights_data=weights_data, mean=self.mean_)
