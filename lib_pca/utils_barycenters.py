import torch
import ot

import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
from geomstats.geometry.stiefel import Stiefel
from geomstats.geometry.grassmannian import Grassmannian


def get_ground_barycenter(n_components):
    """
        Return a function which, given a list of len K of tensors x of shape (n, d), 
        returns n PCA over the K points, and returns the eigenvectors of the PCA as a tensor of shape (n, d, n_components).

        Parameters
        ----------
        n_components : int
            Number of components to keep in the PCA.

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
        # X = X - X.mean(dim=0, keepdim=True)
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
    # projs = torch.einsum("nij, nkj, mk -> nmi", U, U, x)
    # C = torch.linalg.norm(x[None] - projs, axis=-1)**2
    # return C

    x_norm2 = torch.sum(x**2, dim=-1)              # (m,)
    Ux = torch.einsum('ndk, md -> nmk', U, x)      # (n, m, k)  -- U^T x per n
    proj_norm2 = torch.sum(Ux**2, dim=-1)          # (n, m)
    C = x_norm2[None, :] - proj_norm2              # Using that |x-UU^Tx|^2 = |x|^2 - |U^Tx|^2

    return C


def center_data(samples_data, mean=None):
    """
        Center the data for each sample.

        Parameters
        ----------
        samples_data :  (list of array-like) 
            List of K arrays of measure positions, each of shape (m_k, d_k).
        mean: tensor of shape (1, d), optional (default None)
            Mean of the samples, used for centering. If None, the mean is computed from the samples as frac1K sum_{k=1}^K m_k where m_k mean of the k-th distribution.

        Outputs
        -------
        samples_data_centered :  (list of array-like)
            List of K arrays of centered measure positions, each of shape (m_k, d_k).
        mean_ : tensor of shape (1, d)
            Mean of the samples, used for centering.
    """
    if mean is None:
        means = torch.concatenate([torch.mean(data, axis=0)[None] for data in samples_data], axis=0)
        # samples_data_centered = samples_data - torch.mean(means, axis=0, keepdims=True)
        mean_ = torch.mean(means, axis=0, keepdims=True)
    else:
        mean_ = mean

    samples_data_centered = [samples_data[k] - mean_ for k in range(len(samples_data))]
    return samples_data_centered, mean_


def cost_barycenter(samples_U, samples_data, weights_U=None, weights_data=None, weights_distrs=None, mean=None):
    """
        Compute the cost of the barycenter with respect to the given samples.

        Parameters
        ----------
        samples_U : tensor of shape (n, d, n_components)
            Tensor containing the distribution on the Stiefel manifold.
        samples_data :  (list of array-like) 
            List of K arrays of measure positions, each of shape (m_k, d).
        weights_U : tensor of shape (n,), optional (default uniform)
            Weights of the distributions on the Stiefel manifold.
        weights_data : (list of array-like) (optional, default uniform)
            List of K arrays of measure weights, each of shape (m_k,).
        weights_distrs : tensor of shape (K,), optional (default uniform)
            Weights of the distributions.
        mean: tensor of shape (1, d), optional (default None)
            Mean of the samples, used for centering. If None, the mean is computed from the samples as frac1K sum_{k=1}^K m_k where m_k mean of the k-th distribution.
    """
    n_distr = len(samples_data)
    device = samples_data[0].device

    samples_data, _ = center_data(samples_data, mean=mean)

    support_bary = samples_U
    weight_bary = weights_U

    if weights_data is None:
        weights_data = [torch.ones(m.shape[0], device=device) / m.shape[0] for m in samples_data]  # weights each measure

    if weight_bary is None:
        weight_bary = torch.ones(support_bary.shape[0], device=device) / support_bary.shape[0]

    if weights_distrs is None:
        weights_distrs = torch.ones(n_distr, device=device) / n_distr

    L_ot_costs = torch.zeros(n_distr, device=device)

    for k in range(n_distr):
        C = cost_pca(support_bary, samples_data[k])
        b = weights_data[k]

        L_ot_costs[k] = ot.emd2(weight_bary, b, C)

    return torch.sum(L_ot_costs * weights_distrs)


def barycenter_fixed_support(support, samples, init_weights=None, weights_samples=None,
                             weights_barycenter=None, n_iter=100, stopThr=1e-5,
                             lr_bar=1, mode="sinkhorn", reg=1, log=False):
    """
        Compute barycenter optimizing over weights with fixed support by accelerated mirror descent, as in [1].

        [1] Cuturi, M., & Doucet, A. (2014). Fast computation of Wasserstein barycenters. In International conference on machine learning (pp. 685-693). PMLR.
    
        Parameters  
        ----------
        support : tensor of shape (n, d, n_components) with n the length of the support on the Stiefel manifold
            Support of the barycenter (distribution on the Stiefel manifold)
        samples :  (list of array-like) 
            List of K arrays of measure positions, each of shape (m_k, d_k).
        init_weights : tensor of shape (n,), optional (default uniform)
            Initial weights of the barycenter measure?
        weights_samples : (list of array-like), optional (default uniform)
            List of K arrays of weights of shape (m_k,)
        weights_barycenter : tensor of shape (K,), optional (default uniform)
            Weights of each term of the barycenter
        n_iter : int, optional (default 100)
            Maximum number of iterations of Mirror descent
        stopThr : float, optional (default 1e-5)
            Stopping threshold for the mirror descent
        lr_bar : int, optional (default 1)
            Learning rate for the mirror descent
        mode : str, optional (default "sinkhorn")
            Mode for the OT solver ("sinkhorn" or "proximal")
        reg : float, optional (default 1)
            Regularization for the Sinkhorn solver
        log: boolean, optional (default False)
            If True, return the loss along iterations
    """
    n, d, n_components = support.shape
    n_distr = len(samples)
    device = support.device
    dtype = support.dtype

    if weights_barycenter is None:
        weights_barycenter = torch.ones(n_distr, device=device) / n_distr

    if weights_samples is None:
        weights_samples = torch.ones(n_distr, samples[0].shape[0], device=device) / samples[0].shape[0]
    else:
        weights_samples = torch.cat([weights_samples[k][None] for k in range(len(weights_samples))], dim=0)

    # Precompute once: support is fixed in this solver
    C_list = torch.cat([cost_pca(support, samples[k])[None] for k in range(n_distr)], dim=0) # shape (K, n, m)
    # C_maxes = torch.tensor([C_list[k].max() for k in range(n_distr)], device=device)
    C_max = C_list.max()

    barycenter = torch.ones(n, device=device, dtype=dtype) / n if init_weights is None else init_weights.clone().detach()
    # barycenter.requires_grad = True

    tab_loss = []
    barhat = barycenter.clone()
    bartilde = barycenter.clone()

    log_bartilde = torch.log(bartilde)

    for i in range(n_iter):
        # Nesterov Acceleration — beta starts at 1 so (1-1/beta)*barhat + (1/beta)*bartilde
        # is always a valid convex combination.
        beta = (i+2)/2
        barycenter = (1-1./beta) * barhat + bartilde / beta

        # Solve K OT problems with Sinkhorn
        results = ot.batch.solve_batch(
            C_list/C_max,# C_maxes[:, None, None],
            reg=reg,
            a=barycenter,
            b=weights_samples,
            method=mode,
            # grad="last_step"
        )

        # Gradient via Danskin's theorem:
        # d/da sum_k w_k * OT_eps(a, b_k; C_k) = sum_k w_k * f*_k
        # where f*_k = log(u*_k) is the Sinkhorn dual potential for the a-marginal.
        # results.potentials[0] returns the raw Sinkhorn scalings u/c (not log(u)), so
        # we take the log to recover the true dual potential (up to an additive constant
        # that cancels after centering onto the simplex tangent space).
        f_batch, _ = results.potentials  # shape (K, n) — contains u/c, not log(u)

        # barycenter_grad = torch.einsum('k,kn,k->n', weights_barycenter, f_batch, C_maxes)
        barycenter_grad = torch.einsum('k,kn->n', weights_barycenter, f_batch) * C_max

        # loss = torch.sum(weights_barycenter * results.value_linear * C_maxes)
        loss = torch.sum(weights_barycenter * results.value_linear * C_max)

        # barycenter_grad = torch.autograd.grad(loss, barycenter)[0]
        grad_u = barycenter_grad - barycenter_grad.mean()

        # Mirror descent in log-space to avoid overflow with large gradients.
        log_bartilde = log_bartilde - lr_bar * beta * grad_u
        log_bartilde = log_bartilde - torch.logsumexp(log_bartilde, dim=0)
        bartilde = torch.exp(log_bartilde)

        # Nesterov update
        barhat = (1-1./beta) * barhat + bartilde / beta
        tab_loss.append(loss.item())

        if i>1:
            if abs(tab_loss[-1] - tab_loss[-2]) < stopThr:
                break

    if log:
        dico_log = {"loss": tab_loss}
        return barhat.detach(), dico_log

    return barhat.detach()
