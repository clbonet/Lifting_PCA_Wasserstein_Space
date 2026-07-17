import torch

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse


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
