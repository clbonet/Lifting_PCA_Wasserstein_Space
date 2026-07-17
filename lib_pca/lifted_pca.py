import torch
import ot

from sklearn.decomposition import PCA


def get_ground_barycenter(n_components):
    """
        Return a function which, given a list of len K of tensors x of shape (n, d), 
        returns n PCA over the K points, and returns the eigenvectors of the PCA as a tensor of shape (n, d, n_components).

        Parameters
        ----------
        n_components : int
            Number of components to keep in the PCA.

        # TODO: more efficient computations of eigenvectors by vectorizing and doing the PCA ourselves?
    """
    # def B(x, n_components=n_components):
    #     """
    #         Parameters
    #         ----------
    #         x : list of len K of tensors of shape (n, d)

    #         Returns
    #         -------
    #         output : tensor of shape (n, d, n_components)
    #             Tensor containing the eigenvectors of the PCA for each of the n points.
    #     """
    #     n, d = x[0].shape
    #     pca = PCA(n_components)

    #     output = torch.zeros((n, d, n_components))

    #     for i in range(n):
    #         z = torch.concatenate([x[k][i][None] for k in range(len(x))], axis=0)
    #         pca.fit(z)
    #         eigenvectors = pca.components_
    #         output[i] = torch.from_numpy(eigenvectors.T)

    #     return output

    def B(x):
        """
            Parameters
            ----------
            x : list of len K of tensors of shape (n, d)

            Returns
            -------
            output : tensor of shape (n, d, n_components)
                Tensor containing the eigenvectors of the PCA for each of the n points.
        """
        X = torch.stack(x, dim=0)          # (K, n, d)
        X = X - X.mean(dim=0, keepdim=True)
        X = X.permute(1, 0, 2)             # (n, K, d)
    
        # Batched SVD
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    
        # Right singular vectors are the PCA directions
        return Vh.transpose(-2, -1)[:, :, :n_components]
    
    return B


def cost_pca(U, x):
    """
        Groundcost of the PCA (batched)

        Parameters
        ----------
        U : tensor of shape (n, d, n_components)
            Tensor containing the eigenvectors of the PCA for each of the n points.
        x : tensor of shape (m, d)
            Tensor containing the m points.

        Returns
        -------
        C : tensor of shape (n, m)
            Tensor containing the cost of the PCA for each of the n points and m points.
    """
    projs = torch.einsum("nij, nkj, mk -> nmi", U, U, x)
    C = torch.linalg.norm(x[None] - projs, axis=-1)**2
    return C


class LiftedPCA():
    def __init__(self, n_components=1, n_samples=1):
        self.n_components = n_components
        self.n_samples = n_samples

    def fit(self, samples, weights=None, fixed_point_its = 5, stop_threshold = 1e-20, X_init=None, seed=None, log=False):
        """
            Fit the lifted PCA model to the given samples using the fixed point algorithm from [1],
            with its POT implementation (available at https://pythonot.github.io/auto_examples/barycenters/plot_free_support_barycenter_generic_cost.html).

            For now, only optimize over the matrices on the Stiefel manifold, not over the weights (TODO).

            [1] Tanguy, E., Delon, J., & Gozlan, N. (2024). Computing barycentres of measures for generic transport costs. arXiv preprint arXiv:2501.04016. 

            Parameters
            ----------
            samples :  (list of array-like) 
                List of K arrays of measure positions, each of shape (m_k, d_k).
            fixed_point_its : int
                Number of iterations for the fixed point algorithm.
            stop_threshold : float
                Threshold for stopping the fixed point algorithm.
            X_init : array-like, optional
                Initial guess for the barycenter positions, of shape (n_samples, d, n_components). If None, will be initialized randomly.
        """
        d = samples[0].shape[1]

        # Center samples
        means = torch.concatenate([torch.mean(data, axis=0)[None] for data in samples], axis=0)
        samples = samples - torch.mean(means, axis=0, keepdims=True)        

        if seed is not None:
            torch.manual_seed(seed)

        if X_init is None:
            # Initialize uniformly on the Stiefel manifold
            Z = torch.randn(self.n_samples, d, self.n_components)
            X_init, _ = torch.linalg.qr(Z)

        B = get_ground_barycenter(self.n_components)
        cost_list = [cost_pca for k in range(len(samples))]

        if weights is None:
            weights = [torch.ones(m.shape[0]) / m.shape[0] for m in samples]  # weights each measure

        X_bar, log_dict = ot.lp.free_support_barycenter_generic_costs(
            samples,
            weights,
            X_init,
            cost_list,
            B,
            numItermax=fixed_point_its,
            stopThr=stop_threshold,
            log=True,
        )

        self.components_ = X_bar

        if log:
            return log_dict
