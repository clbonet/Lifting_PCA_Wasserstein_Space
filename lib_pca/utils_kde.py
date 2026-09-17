import torch
import numpy as np

from scipy.special import i0


def kde_circle(circle_points, directions, kappa=100):
    """
        Evaluate a von Mises kernel density estimate on the unit circle S^1.
        
        Parameters
        ----------
        circle_points : array-like, shape (n_points, 2)
            Points on the unit circle where the density will be evaluated.
        directions : array-like, shape (n_directions, 2)
            Directions (unit vectors) that define the von Mises kernels.
        
        Returns
        -------
        density : array-like, shape (n_points,)
            The estimated density values at the specified circle points.
    """
    log_kernel = kappa * (circle_points @ directions.T)
    density = np.exp(log_kernel).mean(axis=1) / (2 * np.pi * i0(kappa))

    return density


def projection_metric(U, V):
    """
        Return squared Grassmann projection distances.

        The projection metric is

        ``d_proj(U, V)^2 = 0.5 * ||U U.T - V V.T||_F^2``.

        Parameters
        ----------
        U : torch.Tensor, shape (n, d, q) or (d, q)
            Live on the Stiefel manifold, i.e. U^TU = I_q
        V : torch.Tensor, shape (m, d, q) or (d, q)

        Returns
        -------
        torch.Tensor, shape (n, m)
            Pairwise squared projection distances.
    """

    lift_U = torch.einsum("...dq, ...pq ->...dp", U, U)
    lift_V = torch.einsum("...dq, ...pq ->...dp", V, V)

    dist = torch.sum(torch.square(lift_U[:,None]-lift_V[None]), dim=(-2, -1)) / 2

    return dist


def kde_stiefel(eval_points, V_kde, bandwidth):
    log_kernel = - projection_metric(eval_points, V_kde) / (2 * bandwidth**2)
    density = torch.mean(torch.exp(log_kernel), dim=1) / np.sqrt(2*np.pi*bandwidth**2)

    return density
