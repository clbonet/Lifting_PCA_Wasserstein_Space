import torch

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from scipy.special import i0


def draw_cov(mu, C, color=None, label=None, nstd=1, alpha=0.5, ax=None):
    # https://pythonot.github.io/auto_examples/gaussian_gmm/plot_GMM_flow.html#sphx-glr-auto-examples-gaussian-gmm-plot-gmm-flow-py
    def eigsorted(cov):
        if torch.is_tensor(cov):
            cov = cov.detach().numpy()
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1].copy()
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(C)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(
        xy=(mu[0], mu[1]),
        width=w,
        height=h,
        alpha=alpha,
        angle=theta,
        facecolor=color,
        edgecolor=color,
        label=label,
        fill=True,
    )

    if ax is None:
        ax = plt.gca()
    
    # pl.gca().add_artist(ell)
    ax.add_patch(ell)


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


def draw_kde_circle(directions, ax=None, color_kde="darkorange", kappa=100):
    """
        Draw a von Mises kernel density estimate on the unit circle S^1.

        Parameters
        ----------
        directions : array-like, shape (n_directions, 2, 1) or (n_directions, 2)
            Distributions of samples on the circle to be approximated by the von Mises kernel density estimate.
        ax : matplotlib.axes.Axes, optional
            The axes on which to draw the density. If None, the current axes will be used.
        color_kde : str, optional
            Color of the density plot. Default is "darkorange".
        kappa : float, optional
            Concentration parameter of the von Mises distribution. Higher values result in a more concentrated density around the directions. Default is 100.
    """

    if len(directions.shape) == 3:
        directions = directions.squeeze(-1)

    symmetric_directions = np.concatenate([directions, -directions], axis=0)

    circle = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    circle_points = np.stack([np.cos(circle), np.sin(circle)], axis=1)

    density = kde_circle(circle_points, symmetric_directions, kappa=kappa)

    # Draw the density as a radial band around the unit circle.
    density_scale = 0.8 / density.max()
    radius = 1 + density_scale * density

    outer_x = radius * np.cos(circle)
    outer_y = radius * np.sin(circle)
    inner_x = np.cos(circle)
    inner_y = np.sin(circle)

    if ax is None:
        ax = plt.gca()

    ax.fill(
        np.concatenate([outer_x, inner_x[::-1]]),
        np.concatenate([outer_y, inner_y[::-1]]),
        color=color_kde,
        alpha=0.2,
    )

    ax.plot(inner_x, inner_y, color="0.75", linewidth=1.5, linestyle="--") # Circle
    ax.plot(outer_x, outer_y, color=color_kde, linewidth=2.5, label=r"von Mises KDE on $S^1$")

    # ax.scatter(directions[:, 0], directions[:, 1], color="black", s=24, alpha=0.8, label="Main directions")
    # ax.scatter(-directions[:, 0], -directions[:, 1], color="black", s=24, alpha=0.35)
    # ax.set_aspect("equal")


def plot_samples_distributions(samples, colors=None, modulo_colors=1, ax=None, alpha=0.25):
    """
        Plot samples from multiple distributions in a 2D scatter plot.

        Parameters
        ----------
        samples : list of array-like, shape (n_samples_i, 2)
            A list where each element contains samples from a different distribution. Each element should be a 2D array with shape (n_samples_i, 2), where n_samples_i is the number of samples for the i-th distribution.
        colors : array-like, shape (n_distributions, 3), optional
            An array of RGB colors for each distribution. If None, a default colormap will be used.
        modulo_colors : int, optional
            If specified, colors will be reused after this many distributions. Default is 1 (no reuse).
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the samples. If None, the current axes will be used.
        alpha : float, optional
            The transparency level of the scatter points. Default is 0.25.
    """
    if ax is None:
        ax = plt.gca()

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(samples)))

    for k in range(len(samples)):
        ax.scatter(samples[k][:,0], samples[k][:,1], c=colors[k//modulo_colors], alpha=alpha)
