import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from lib_pca.utils_plot import plot_samples_distributions, plot_eigenvectors_pca, draw_kde_circle, draw_confidence_interval, plot_barycenter
from lib_pca.uncertainty_aware_pca import UAPCA
from lib_pca.lifted_pca import LiftedPCA


def get_plot_pca(samples, ax=None, xmin=-3, xmax=3, ymin=-3, ymax=3, colors=None):
    concat_samples = torch.cat([samples[k] for k in range(len(samples))], axis=0)

    pca = PCA()
    pca.fit(concat_samples.numpy())

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot([0, 0], [ymin, ymax], "--", c="0.75")
    ax.plot([xmin, xmax], [0, 0], "--", c="0.75")

    plot_samples_distributions(samples - torch.mean(concat_samples, dim=0, keepdim=True), colors=colors, ax=ax)
    plot_eigenvectors_pca(pca.components_, pca.explained_variance_ratio_, ax=ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    ax.set_title("PCA", fontsize=13)


def get_plot_uapca(samples, ax=None, xmin=-3, xmax=3, ymin=-3, ymax=3, colors=None, title="UAPCA", value_barycenter=False):
    pca_distributional = UAPCA(2)
    pca_distributional.fit_from_samples(samples)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot([0, 0], [ymin, ymax], "--", c="0.75")
    ax.plot([xmin, xmax], [0, 0], "--", c="0.75")

    plot_samples_distributions(pca_distributional.center_data(samples), colors=colors, ax=ax)
    plot_eigenvectors_pca(pca_distributional.components_.T, pca_distributional.explained_variance_ratio_, ax=ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    if value_barycenter:
        pca_distributional = UAPCA(1)
        pca_distributional.fit_from_samples(samples)
        title = title + r", $\mathcal{T}_{\mathcal{C}_{\mathrm{St}}}=$" + str(round(pca_distributional.cost_barycenter(samples).item(), 2))

    ax.set_title(title, fontsize=13)


def get_plot_lifted_pca(samples, ax=None, xmin=-3, xmax=3, ymin=-3, ymax=3, colors=None,
                        plot_ci=True, plot_circle=True, plot_samples=True, plot_bary=False,
                        title="Lifted PCA", value_barycenter=False):
    lifted_pca = LiftedPCA(n_components=1, n_samples=50)
    lifted_pca.fit(samples, solver="fixed_weights")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot([0, 0], [ymin, ymax], "--", c="0.75")
    ax.plot([xmin, xmax], [0, 0], "--", c="0.75")

    if plot_samples:
        plot_samples_distributions(lifted_pca.center_data(samples), colors=colors, ax=ax)

    if plot_circle:
        draw_kde_circle(lifted_pca.components_.detach().cpu().numpy(), kappa=100, ax=ax)

    if plot_ci:
        draw_confidence_interval(lifted_pca.components_, barycenter=lifted_pca.get_barycenter_components()[:, 0], ax=ax, ci=95)

    if plot_bary:
        plot_barycenter(lifted_pca.get_barycenter_components()[:, 0], ax=ax)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis("off")

    if value_barycenter:
        title = title + r", $\mathcal{T}_{\mathcal{C}_{\mathrm{St}}}=$" + str(round(lifted_pca.cost_barycenter(samples).item(), 2))

    ax.set_title(title, fontsize=13)
