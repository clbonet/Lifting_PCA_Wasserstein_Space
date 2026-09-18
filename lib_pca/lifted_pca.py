import os

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
os.environ["SCIPY_ARRAY_API"] = "1"

import ot
import torch

import numpy as np

from geomstats.geometry.stiefel import Stiefel
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.learning.frechet_mean import FrechetMean

from .utils_barycenters import barycenter_fixed_support, get_ground_barycenter, cost_pca, center_data, cost_barycenter
from .utils_data import get_list_by_class


class LiftedPCA():
    def __init__(self, n_components=1, n_samples=1):
        self.n_components = n_components
        self.n_samples = n_samples


    def fit(self, samples_data, support_bary=None, weights_data=None, weight_bary=None, solver="fixed_weights",
            fixed_point_its = 5, stop_threshold = 1e-20, n_iter_bary=500, lr_bary=1, reg=1, stop_threshold_fixed_support=1e-20,
            n_iter_alternate=10, seed=None, mode="sinkhorn"):
        """
            Fit the lifted PCA model to the given samples by computing the optimal barycenter.
            There are 3 possible solvers: 
            - with fixed support, which optimizes only over the weights, using the accelerated mirror descent algorithm of [2],
            - with fixed weights, which optimizes only over the support, using the fixed-point algorithm of [1],
            - with free support and free weights, which optimizes over both the support and the weights alternatively

            The support of the barycenter is comonents_, and the weights of the barycenter are weights_.

            [1] Tanguy, E., Delon, J., & Gozlan, N. (2024). Computing barycentres of measures for generic transport costs. arXiv preprint arXiv:2501.04016.

            [2] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters. In International conference on machine learning (pp. 685-693). PMLR.


            Parameters
            ----------
            samples_data :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
            support_bary : array-like, optional
                Initial guess for the barycenter positions, of shape (n_samples, d, n_components). If None, will be initialized randomly.
            weights_data : (list of array-like) (optional, default uniform)
                List of K arrays of measure weights, each of shape (m_k,).
            weight_bary : tensor of shape (n_samples,) (optional, default uniform)
                Weight of the barycenter (the optimized measure)
            solver : str, optional (default "fixed_weights")
                Solver to use for the barycenter computation. Can be "fixed_support", "fixed_weights" or "free".
            fixed_point_its : int
                Number of iterations for the fixed point algorithm (for "fixed_weights" or "free" solvers).
            stop_threshold : float
                Threshold for stopping the fixed point algorithm (for "fixed_weights" or "free" solvers).
            n_iter_bary : int, optional (default 100)
                Number of iterations of Mirror descent (for "fixed_support" or "free" solvers).
            lr_bary : int, optional (default 1)
                Learning rate for the mirror descent (for "fixed_support" or "free" solvers).
            reg : float, optional (default 1)
                Regularization for the Sinkhorn solver (for "fixed_support" or "free" solvers).
            stop_threshold_fixed_support : float, optional (default 1e-4)
                Threshold for stopping the Mirror Descent (for "fixed_support" solver).
            n_iter_alternate : int, optional (default 10)
                Number of iterations for the alternate optimization (for "free" solver).
            seed : int, optional (default None)
                Random seed for reproducibility.
            mode: str, optional (default "sinkhorn")
                Mode for the OT solver ("sinkhorn" or "proximal") (for "fixed support" solver)
        """
        d = samples_data[0].shape[1]
        device = samples_data[0].device
        dtype = samples_data[0].dtype

        # Center samples
        samples_data = self.center_data(samples_data)

        if seed is not None:
            torch.manual_seed(seed)

        if support_bary is None:
            # Initialize uniformly on the Stiefel manifold
            Z = torch.randn(self.n_samples, d, self.n_components, dtype=dtype).to(device)
            support_bary, _ = torch.linalg.qr(Z)

        if weights_data is None:
            weights_data = [torch.ones(m.shape[0], device=device) / m.shape[0] for m in samples_data]  # weights each measure

        if solver == "fixed_support":
            weight_bary = self.barycenter_fixed_support(
                samples_data, support_bary, weights_data, weight_bary, n_iter_bary, stop_threshold_fixed_support, lr_bary, reg, mode
                )
        elif solver == "fixed_weights":
            support_bary = self.barycenter_fixed_weights(
                samples_data, support_bary, weights_data, weight_bary, fixed_point_its, stop_threshold
                )
        elif solver == "free":
            for i in range(n_iter_alternate):
                support_bary = self.barycenter_fixed_weights(
                    samples_data, support_bary, weights_data, weight_bary, fixed_point_its, stop_threshold
                    )
                weight_bary = self.barycenter_fixed_support(
                    samples_data, support_bary, weights_data, weight_bary, n_iter_bary, stop_threshold_fixed_support, lr_bary, reg, mode
                    )

        self.components_ = support_bary
        self.weights_ = weight_bary

    def fit_from_labels(self, X, y, **args):
        L_X = get_list_by_class(X, y)
        self.fit(L_X, **args)


    def barycenter_fixed_weights(
            self, samples_data, support_bary, weights_data=None, weight_bary=None, fixed_point_its = 5, stop_threshold = 1e-20, log=False
            ):
        """
            Optimize the lifted PCA barycenter over the support of the barycenter, fixing the weights. For this, we use the fixed point algorithm from [1], 
            with its POT implementation (available at https://pythonot.github.io/auto_examples/barycenters/plot_free_support_barycenter_generic_cost.html).

            [1] Tanguy, E., Delon, J., & Gozlan, N. (2024). Computing barycentres of measures for generic transport costs. arXiv preprint arXiv:2501.04016.

            Parameters
            ----------
            samples_data :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
            support_bary : array-like
                Initial guess for the barycenter positions, of shape (n_samples, d, n_components). If None, will be initialized randomly.
            weights_data : (list of array-like) (optional, default uniform)
                List of K arrays of measure weights, each of shape (m_k,).
            weight_bary : tensor of shape (n_samples,) (optional, default uniform)
                Weight of the barycenter (the optimized measure)
            fixed_point_its : int
                Number of iterations for the fixed point algorithm.
            stop_threshold : float
                Threshold for stopping the fixed point algorithm.
        """

        B = get_ground_barycenter(self.n_components)
        cost_list = [cost_pca for k in range(len(samples_data))]

        X_bary, log_dict = ot.lp.free_support_barycenter_generic_costs(
            samples_data,
            weights_data,
            support_bary,
            cost_list,
            B,
            numItermax=fixed_point_its,
            stopThr=stop_threshold,
            log=True,
            a=weight_bary
        )

        if log:
            return X_bary, log_dict

        return X_bary



    def barycenter_fixed_support(self, samples_data, support_bary=None, weights_data=None, weight_bary=None,
                                 n_iter=100, stop_threshold=1e-5, lr_bary=1, reg=1, mode="sinkhorn", log=False):
        """
            Optimize the lifted PCA barycenter over the weights of the barycenter, fixing the support. For this, we use the accelerated mirror descent algorithm from [2].

            [2] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters. In International conference on machine learning (pp. 685-693). PMLR.

            Parameters
            ----------
            samples_data :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
            support_bary : array-like
                Initial guess for the barycenter positions, of shape (n_samples, d, n_components). If None, will be initialized randomly.
            weights_data : (list of array-like) (optional, default uniform)
                List of K arrays of measure weights, each of shape (m_k,).
            weight_bary : tensor of shape (n_samples,) (optional, default uniform)
                Weight of the barycenter (the optimized measure)
            n_iter : int, optional (default 100)
                Maximum number of iterations of Mirror descent
            stop_threshold : float, optional (default 1e-5)
                Stopping threshold for the mirror descent
            lr_bary : int, optional (default 1)
                Learning rate for the mirror descent
            reg : float, optional (default 1)
                Regularization for the Sinkhorn solver
            mode: str, optional (default "sinkhorn")
                Mode for the OT solver ("sinkhorn" or "proximal")
            log: boolean, optional (default False)
                If True, return the loss along iterations
        """
        if log:
            weights, log_dict = barycenter_fixed_support(support_bary, samples_data, weight_bary, weights_data,
                                                         lr_bar=lr_bary, stopThr=stop_threshold, mode=mode, reg=reg, n_iter=n_iter, log=log)
        else:
            weights = barycenter_fixed_support(support_bary, samples_data, weight_bary, weights_data, lr_bar=lr_bary, stopThr=stop_threshold,
                                               mode=mode, reg=reg, n_iter=n_iter)

        if log:
            return weights, log_dict

        return weights


    def center_data(self, samples_data, mean=None):
        """
            Center the data for each sample.

            Parameters
            ----------
            samples_data :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
        """
        samples_data_centered, mean_ = center_data(samples_data, mean=mean)
        self.mean_ = mean_

        return samples_data_centered


    def cost_barycenter(self, samples_data, weights_data=None):
        """
            Compute the cost of the barycenter with respect to the given samples.

            Parameters
            ----------
            samples_data :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
            weights_data : (list of array-like) (optional, default uniform)
                List of K arrays of measure weights, each of shape (m_k,).
        """
        return cost_barycenter(self.components_, samples_data, self.weights_, weights_data, mean=self.mean_)


    def get_barycenter_components(self):
        """
            Computes the barycenter of the distribution of PCA components.
        """
        # stiefel = Stiefel(self.components_.shape[1], self.n_components)
        grassmann = Grassmannian(self.components_.shape[1], self.n_components)

        # Intrinsic Fréchet/Karcher mean
        # mean_estimator = FrechetMean(space=stiefel)
        mean_estimator = FrechetMean(space=grassmann)

        projs = torch.einsum("idq,ieq->ide", self.components_, self.components_)

        mean_estimator.fit(projs, weights=self.weights_)

        P = mean_estimator.estimate_
        eigenvalues, eigenvectors = torch.linalg.eigh(P)

        return eigenvectors[:, -self.n_components:]


    def project_samples(self, samples):
        """
            Project samples on all directions

            Parameters
            ----------
            samples : torch tensor of shape (m, d)
        """
        # Map to projector (on the Grassmann manifold)
        projectors = torch.einsum("ndq, npq -> ndp", self.components_, self.components_)
        projected_samples = torch.einsum("ndp,mp->nmd", projectors, samples)

        return projected_samples
