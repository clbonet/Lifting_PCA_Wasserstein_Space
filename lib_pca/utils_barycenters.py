import torch
import ot

import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA


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
    # projs = torch.einsum("nij, nkj, mk -> nmi", U, U, x)
    # C = torch.linalg.norm(x[None] - projs, axis=-1)**2
    # return C

    x_norm2 = torch.sum(x**2, dim=-1)              # (m,)
    Ux = torch.einsum('ndk, md -> nmk', U, x)      # (n, m, k)  -- U^T x per n
    proj_norm2 = torch.sum(Ux**2, dim=-1)          # (n, m)
    C = x_norm2[None, :] - proj_norm2              # Using that |x-UU^Tx|^2 = |x|^2 - |U^Tx|^2
    return C


def barycenter_fixed_support(support, samples, init_weights=None, weights_samples=None, weights_barycenter=None, n_iter=100, stopThr=1e-5,
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
        init_weights : tensor of shape (K,), optional (default uniform)
            Initial weights of the barycenter measure?
        weights_samples : (list of array-like), optional (default uniform)
            List of K arrays of weights of shape (m_k,)
        weights_barycenter : tensor of shape (K,), optional (default uniform)
            Weights of ieach term of the barycenter
        n_iter : int, optional (default 100)
            Maximum number of iterations of Mirror descent
        stopThr : float, optional (default 1e-5)
            Stopping threshold for the mirror descent
        lr_bar : int, optional (default 1)
            Learning rate for the mirror descent
        mode : str, optional (default "sinkhorn")
            Mode for the OT solver
        reg : float, optional (default 1)
            Regularization for the Sinkhorn solver
        log: boolean, optional (default False)
            If True, return the loss along iterations
    """
    n, d, n_components = support.shape
    n_distr = samples.shape[0]
    device = support.device

    if weights_barycenter is None:
        weights_barycenter = torch.ones(n_distr, device=device) / n_distr

    if weights_samples is None:
        weights_samples = [torch.ones(x.shape[0], device=device)/x.shape[0] for x in samples]

    # Precompute once: support is fixed in this solver
    C_list = [cost_pca(support, samples[k]) for k in range(n_distr)]

    barycenter = torch.ones(n, device=device) / n if init_weights is None else init_weights.clone().detach()
    barycenter.requires_grad_(True)

    tab_loss= []
    barhat = barycenter.clone().detach()
    bartilde = barycenter.clone().detach()

    for i in range(n_iter):
        # Nesterov Acceleration
        beta = (i+1)/2
        barycenter.data = (1-1./beta) * barhat + bartilde / beta

        # Compute full loss barycenter (by solving K OT problems)
        loss = torch.zeros(1, device=device)
        for k in range(n_distr):
            C = C_list[k]
            b = weights_samples[k]

            if mode == "emd":
                loss += weights_barycenter[k] * ot.emd2(barycenter, b, C)
            elif mode == "sinkhorn":
                loss += weights_barycenter[k] * ot.sinkhorn2(barycenter, b, C/C.max(), reg) * C.max()

        # loss.backward()
        # Compute potentials by taking the gradient, center the alpha
        # grad_u = barycenter.grad - barycenter.grad.mean()
        # barycenter.grad.zero_()

        barycenter_grad = torch.autograd.grad(loss, barycenter)[0]
        grad_u = barycenter_grad - barycenter_grad.mean()
    
        # Mirror descent step
        bartilde = bartilde * torch.exp(-lr_bar * beta * grad_u)
        bartilde = bartilde / torch.sum(bartilde)

        # final Nesterov update
        barhat = (1-1./beta) * barhat + bartilde / beta
        tab_loss.append(loss.item())

        if i>1:
            if abs(tab_loss[-1] - tab_loss[-2]) < stopThr:
                break

    if log:
        dico_log = {"loss": tab_loss}
        return barycenter.detach(), dico_log

    plt.plot(tab_loss)
    plt.show()
    
    return barycenter.detach()
