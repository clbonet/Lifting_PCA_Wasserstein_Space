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
