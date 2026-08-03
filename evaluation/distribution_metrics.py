"""Distribution-level comparison metrics between real and generated samples.

MMD uses the median-distance heuristic for the RBF kernel bandwidth
rather than a separate per-dimension feature-standardization step. A
sibling project's diagnostics previously hit a "feature-MMD
standardization collapse" (a near-constant feature dimension blown up by
standardization until it swamped the kernel distance and made every MMD
estimate degenerate). The median heuristic sidesteps this: it sets the
kernel bandwidth from the data's own pairwise-distance scale directly, so
there is no separate normalisation step that a low-variance dimension can
corrupt.

Wasserstein distance is reported as the Sliced Wasserstein Distance
(Rabin et al., "Wasserstein Barycenter and Its Application to Texture
Mixing", 2011): the average of the exact 1D Wasserstein distance (which
has an O(n log n) closed form via sorting) over many random projections.
This is the standard, dependency-light way to compare two sets of
high-dimensional samples without solving a full optimal-transport LP
(which would need the POT package, not otherwise used in this project).
It is explicitly a projection-based approximation to full multivariate
Wasserstein/EMD, not the exact value -- named accordingly rather than as
a plain "Wasserstein distance" so the approximation is not mistaken for
an exact computation.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from physics.spectral import spectral_energy_error as _spectral_energy_error


def rbf_mmd(x: np.ndarray, y: np.ndarray, sigma: float | None = None) -> float:
    """Maximum Mean Discrepancy (Gretton et al., 2012) with an RBF kernel.

    Args:
        x: (N, D) real samples.
        y: (M, D) generated samples.
        sigma: RBF bandwidth; defaults to the median pairwise distance
            across the pooled (x, y) samples (median heuristic).

    Returns:
        Scalar MMD (not squared), clamped at 0 to absorb small negative
        values from finite-sample estimation noise.
    """
    xx = x @ x.T
    yy = y @ y.T
    xy = x @ y.T
    x_sq = np.diag(xx)
    y_sq = np.diag(yy)

    if sigma is None:
        pooled = np.concatenate([x, y], axis=0)
        pooled_sq = np.sum(pooled**2, axis=1)
        dists_sq = pooled_sq[:, None] + pooled_sq[None, :] - 2 * (pooled @ pooled.T)
        sigma = float(np.sqrt(np.median(np.clip(dists_sq, 0, None)) + 1e-12))
        sigma = max(sigma, 1e-6)

    gamma = 1.0 / (2 * sigma**2)
    k_xx = np.exp(-gamma * (x_sq[:, None] + x_sq[None, :] - 2 * xx))
    k_yy = np.exp(-gamma * (y_sq[:, None] + y_sq[None, :] - 2 * yy))
    k_xy = np.exp(-gamma * (x_sq[:, None] + y_sq[None, :] - 2 * xy))

    m, n = x.shape[0], y.shape[0]
    mmd_sq = k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 2 * k_xy.sum() / (m * n)
    return float(np.sqrt(max(mmd_sq, 0.0)))


def sliced_wasserstein_distance(
    x: np.ndarray, y: np.ndarray, n_projections: int = 50, seed: int = 0
) -> float:
    """Sliced Wasserstein Distance between two sets of samples.

    Args:
        x: (N, D) real samples.
        y: (M, D) generated samples.
        n_projections: Number of random 1D projections to average over.
        seed: RNG seed for the projection directions.

    Returns:
        Scalar SWD (mean of the exact 1D Wasserstein distances across projections).
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    directions = rng.normal(size=(n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12

    distances = np.empty(n_projections)
    for i, direction in enumerate(directions):
        distances[i] = wasserstein_distance(x @ direction, y @ direction)
    return float(distances.mean())


def spectral_energy_error(u_a: np.ndarray, v_a: np.ndarray, u_b: np.ndarray, v_b: np.ndarray) -> float:
    """Mean absolute log-spectral error between two batches' energy spectra.

    Thin re-export of physics/spectral.py's implementation so all
    generation-quality metrics can be imported from one module.
    """
    import torch

    return _spectral_energy_error(
        torch.from_numpy(u_a).float(),
        torch.from_numpy(v_a).float(),
        torch.from_numpy(u_b).float(),
        torch.from_numpy(v_b).float(),
    )
