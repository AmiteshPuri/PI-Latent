"""Plotting functions shared by the TensorBoard validation callback and the
final analysis notebook, so both draw from one implementation instead of
maintaining the plotting logic twice.

All functions take numpy arrays (already detached/denormalised by the
caller) and return a matplotlib Figure -- callers decide whether to
`writer.add_figure(...)`, `fig.savefig(...)`, or just `plt.show()`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_validation_fields(
    gt: np.ndarray,
    recon: np.ndarray,
    residual_field: np.ndarray,
    divergence_field: np.ndarray,
    channel_names: tuple[str, str] = ("Vx", "Vy"),
) -> plt.Figure:
    """Ground truth, reconstruction, error (per channel), plus PDE residual and divergence.

    Args:
        gt, recon: (2, H, W) physical-unit velocity fields for one sample.
        residual_field, divergence_field: (H, W) scalar diagnostic fields.
        channel_names: Labels for the 2 velocity channels.

    Returns:
        A 3-row x 3-col matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))

    for row, name in enumerate(channel_names):
        error = np.abs(gt[row] - recon[row])
        vmin, vmax = gt[row].min(), gt[row].max()

        im0 = axes[row, 0].imshow(gt[row], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[row, 0].set_title(f"GT {name}")
        im1 = axes[row, 1].imshow(recon[row], cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        axes[row, 1].set_title(f"Reconstructed {name}")
        im2 = axes[row, 2].imshow(error, cmap="inferno", origin="lower")
        axes[row, 2].set_title(f"|Error| {name}")
        for im, ax in ((im0, axes[row, 0]), (im1, axes[row, 1]), (im2, axes[row, 2])):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    im_res = axes[2, 0].imshow(residual_field, cmap="PuOr", origin="lower")
    axes[2, 0].set_title("PDE residual (vorticity transport)")
    fig.colorbar(im_res, ax=axes[2, 0], fraction=0.046, pad=0.04)

    im_div = axes[2, 1].imshow(divergence_field, cmap="PuOr", origin="lower")
    axes[2, 1].set_title("Divergence (should be ~0)")
    fig.colorbar(im_div, ax=axes[2, 1], fraction=0.046, pad=0.04)

    axes[2, 2].axis("off")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_training_curves(history: dict[str, list[float]]) -> plt.Figure:
    """Figure 1: training curves for reconstruction loss and PDE residual loss.

    Args:
        history: Dict with any of 'train_reconstruction_loss',
            'val_reconstruction_loss', 'train_residual_loss',
            'val_residual_loss' -> list of per-epoch values. Missing
            keys are skipped.

    Returns:
        1x2 matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    recon_keys = [k for k in ("train_reconstruction_loss", "val_reconstruction_loss") if k in history]
    for k in recon_keys:
        axes[0].plot(history[k], label=k.replace("_", " "))
    axes[0].set_title("Reconstruction loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    res_keys = [k for k in ("train_residual_loss", "val_residual_loss") if k in history]
    for k in res_keys:
        axes[1].plot(history[k], label=k.replace("_", " "))
    axes[1].set_title("PDE residual loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_energy_spectra(k_bins: np.ndarray, e_gt: np.ndarray, e_gen: np.ndarray) -> plt.Figure:
    """Figure 4: radially-averaged kinetic energy spectrum, GT vs generated (log-log)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    eps = 1e-10
    ax.loglog(k_bins[1:], e_gt[1:] + eps, label="Ground truth", linewidth=2)
    ax.loglog(k_bins[1:], e_gen[1:] + eps, label="Generated", linewidth=2, linestyle="--")
    ax.set_xlabel("Wavenumber |k|")
    ax.set_ylabel("E(k)")
    ax.set_title("Kinetic energy spectrum")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_vorticity_comparison(omega_gt: np.ndarray, omega_gen: np.ndarray, n_show: int = 4) -> plt.Figure:
    """Figure 5: side-by-side GT vs generated vorticity field snapshots.

    Args:
        omega_gt, omega_gen: (N, H, W) vorticity fields; the first
            `n_show` of each are plotted.
        n_show: Number of samples to show per row.

    Returns:
        2 x n_show matplotlib Figure.
    """
    n_show = min(n_show, omega_gt.shape[0], omega_gen.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(2.6 * n_show, 5.2))
    vmax = max(np.abs(omega_gt[:n_show]).max(), np.abs(omega_gen[:n_show]).max())

    for i in range(n_show):
        axes[0, i].imshow(omega_gt[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        axes[1, i].imshow(omega_gen[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        axes[1, i].set_xticks([])
        axes[1, i].set_yticks([])

    axes[0, 0].set_ylabel("Ground truth", fontsize=11)
    axes[1, 0].set_ylabel("Generated", fontsize=11)
    fig.suptitle("Vorticity fields: ground truth vs generated")
    fig.tight_layout()
    return fig


def compute_and_plot_tsne(
    real_latents: np.ndarray,
    gen_latents: np.ndarray,
    seed: int = 42,
    perplexity: float | None = None,
) -> plt.Figure:
    """Figure 6: t-SNE projection of real vs generated latent tokens.

    Uses scikit-learn's t-SNE (already a project dependency) rather than
    adding a UMAP dependency; UMAP is a drop-in alternative if
    `umap-learn` happens to be installed (see the commented-out branch
    in the notebook).

    Args:
        real_latents: (N_real, D) flattened latent vectors (e.g. mean-
            pooled per-sample codebook embeddings from real data).
        gen_latents: (N_gen, D) same, from flow-matching-generated samples.
        seed: Random seed for reproducibility.
        perplexity: t-SNE perplexity; defaults to min(30, N/4) if unset.

    Returns:
        Matplotlib Figure with a single 2D scatter plot.
    """
    from sklearn.manifold import TSNE

    combined = np.concatenate([real_latents, gen_latents], axis=0)
    n_real = real_latents.shape[0]

    if perplexity is None:
        perplexity = max(5, min(30, combined.shape[0] // 4))

    projected = TSNE(
        n_components=2, random_state=seed, perplexity=perplexity, init="pca"
    ).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(projected[:n_real, 0], projected[:n_real, 1], s=10, alpha=0.6, label="Real")
    ax.scatter(projected[n_real:, 0], projected[n_real:, 1], s=10, alpha=0.6, label="Generated")
    ax.set_title("Latent-space t-SNE: real vs generated")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_distribution_comparison(
    residual_gt: np.ndarray,
    residual_gen: np.ndarray,
    mmd_value: float,
    wasserstein_value: float,
) -> plt.Figure:
    """Figure 7: MMD/Wasserstein summary bar + PDE residual distribution comparison.

    Args:
        residual_gt, residual_gen: 1D arrays of per-sample PDE residual
            norms (ground truth reconstructions vs generated samples).
        mmd_value, wasserstein_value: Scalar distribution-distance metrics
            (computed in evaluation/distribution_metrics.py).

    Returns:
        1x2 matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].bar(["MMD", "Wasserstein"], [mmd_value, wasserstein_value], color=["#4C72B0", "#DD8452"])
    axes[0].set_title("Distribution distance (generated vs real latents)")
    axes[0].grid(alpha=0.3, axis="y")

    bins = np.histogram_bin_edges(np.concatenate([residual_gt, residual_gen]), bins=30)
    axes[1].hist(residual_gt, bins=bins, alpha=0.6, label="Ground truth (reconstructed)", density=True)
    axes[1].hist(residual_gen, bins=bins, alpha=0.6, label="Generated", density=True)
    axes[1].set_title("PDE residual norm distribution")
    axes[1].set_xlabel("Residual norm")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return fig
