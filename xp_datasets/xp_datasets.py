import os
import sys
import torch

import numpy as np
import matplotlib.pyplot as plt

os.environ["GEOMSTATS_BACKEND"] = "pytorch"
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.decomposition import PCA
from geomstats.geometry.grassmannian import Grassmannian
from geomstats.learning.pca import TangentPCA

sys.path.append("../")
from lib_pca.lifted_pca import LiftedPCA
from lib_pca.utils_kde import kde_stiefel


def plot_projections(X, basis, labels, names, ax=None, title=None, title_color="black", colors=None, fontsize=13):
    """
    """
    basis = basis.to(dtype=X.dtype, device=X.device)

    projected_points = (X @ basis).detach().cpu().numpy()

    for class_idx in range(len(names)):
        class_mask = (labels == class_idx)
    
        ax.scatter(
            projected_points[class_mask, 0],
            projected_points[class_mask, 1],
            s=24,
            alpha=0.75,
            color=colors[class_idx] if colors is not None else None,
            label=names[class_idx],
        )

    ax.set_title(title, color=title_color, fontsize=fontsize)
    # ax.set_xlabel("Projection 1")
    # ax.set_ylabel("Projection 2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.2)


class PCA_Grassmann:
    """
        Tangent PCA to represent distributions of subspaces
    """
    def __init__(self, lifted_pca):
        self.lifted_pca = lifted_pca
        self.grassmann = Grassmannian(lifted_pca.components_.shape[1], lifted_pca.n_components)
        self.tpca = TangentPCA(self.grassmann, n_components=2)

    def fit(self):
        ## Map from Stiefel manifold to Grassmann manifold
        projectors = torch.einsum("idq,ieq->ide", self.lifted_pca.components_, self.lifted_pca.components_)
        barycenter = self.lifted_pca.get_barycenter_components()
        barycenter_grassmann = barycenter @ barycenter.T

        self.tpca.fit(projectors, base_point=barycenter_grassmann)

    def fit_transform(self):
        self.fit()
        return self.transform(self.lifted_pca.components_)

    def transform(self, U):
        projectors = torch.einsum("idq,ieq->ide", U, U)
        return self.tpca.transform(projectors)


def get_weights_KDE(lifted_pca):
    weights = kde_stiefel(lifted_pca.components_, lifted_pca.components_, 0.25).numpy()
    min_weights = weights.min()
    max_weights = weights.max()
    diff = max_weights - min_weights

    alpha = 0.05 + 0.95 * (weights - min_weights) / diff
    alpha_bary = 0.05 + 0.95 * (kde_stiefel(lifted_pca.get_barycenter_components()[None], lifted_pca.components_, 0.25).numpy() - min_weights) / diff

    return alpha, alpha_bary


def plot_kde_grassmann(pca_grassmann, samples, alpha=None, ax=None, marker=None, colorbar=False,
                       color_marker="k", label=None, xytext=(7, 7)):
    """
        Plot the distribution of subspaces on the Grassmann manifold using Tangent PCA
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    projected_points = pca_grassmann.transform(samples).detach().cpu().numpy()

    if len(projected_points)>1:
        scatter = ax.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            c=alpha,
            cmap="plasma",
        )

    else:
        ax.scatter(
            projected_points[:, 0],
            projected_points[:, 1],
            s=190,
            c=color_marker,
            marker=marker if marker is not None else "o",
            label=label
        )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_box_aspect(1)

    if colorbar and len(projected_points) > 1:
        colorbar_axes = ax.inset_axes([1.03, 0.0, 0.04, 1.0])
        ax.figure.colorbar(scatter, cax=colorbar_axes)


def annotate_outlier(ax, point, marker_color="blue", label=None):
    ax.scatter(
        point[0], point[1], s=190, facecolors="none",
        edgecolors=marker_color, linewidths=2.8, zorder=3,
    )
    ax.annotate(
        label, (point[0], point[1]), xytext=(7, 7),
        textcoords="offset points", color=marker_color, fontweight="bold",
    )


def get_plot_kde_grassmann(lifted_pca, pca=None, uapca=None, ax=None, title=None, get_outliers=False, fontsize=13):
    alpha, alpha_bary = get_weights_KDE(lifted_pca)

    pca_grassmann = PCA_Grassmann(lifted_pca)
    pca_grassmann.fit()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    plot_kde_grassmann(pca_grassmann, lifted_pca.components_, alpha=alpha, ax=ax, colorbar=True)

    if pca is not None:
        plot_kde_grassmann(pca_grassmann, pca.components_[None], alpha=None, ax=ax, marker="P", colorbar=False, color_marker="black", label="PCA")

    if uapca is not None:
        plot_kde_grassmann(pca_grassmann, uapca.components_[None], alpha=None, ax=ax, marker="*", colorbar=False, color_marker="teal", label="UAPCA", xytext=(7, -7))

    plot_kde_grassmann(pca_grassmann, lifted_pca.get_barycenter_components()[None], alpha=None, ax=ax, marker="x", colorbar=False, color_marker="darkblue", label="Lifted barycenter")

    if get_outliers:
        tangent_projected_data = pca_grassmann.transform(lifted_pca.components_).detach().cpu().numpy()
        low_candidates = np.argsort(alpha)[:max(2, int(np.ceil(0.2 * len(alpha))))]
        left_low_idx = int(low_candidates[np.argmin(tangent_projected_data[low_candidates, 0])])
        top_candidates = low_candidates[low_candidates != left_low_idx]
        top_low_idx = int(top_candidates[np.argmax(tangent_projected_data[top_candidates, 1])])

        annotate_outlier(ax, tangent_projected_data[left_low_idx], marker_color="royalblue", label="L")
        annotate_outlier(ax, tangent_projected_data[top_low_idx], marker_color="seagreen", label="T")
    
    ax.legend(frameon=False, fontsize=fontsize, loc="best")
    # ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title,fontsize=fontsize)

    if get_outliers:
        return lifted_pca.components_[left_low_idx], lifted_pca.components_[top_low_idx], alpha[left_low_idx], alpha[top_low_idx], alpha_bary[0]
