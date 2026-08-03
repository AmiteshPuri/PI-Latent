"""Radially-averaged spectral energy density for 2D fields.

Standard isotropic-turbulence diagnostic: bin the 2D FFT power spectrum
into shells of constant wavenumber magnitude |k| and sum within each
shell. Used for Figure 4 (energy spectra, GT vs generated) and the
spectral energy error metric in evaluation/distribution_metrics.py.

Both GT and generated fields must go through the exact same function so
any normalisation constant cancels in the comparison -- the absolute
scale is a diagnostic convention, not a physical prediction, so what
matters is that it is applied identically to both.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _radial_bins(h: int, w: int) -> np.ndarray:
    """Integer radial wavenumber bin index for every (ky, kx) grid point."""
    ky = np.fft.fftfreq(h) * h
    kx = np.fft.fftfreq(w) * w
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)
    return np.round(k_mag).astype(int)


def radial_power_spectrum(field: Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum of a single scalar field.

    Args:
        field: (H, W) real array/tensor.

    Returns:
        (k_bins, power): 1D arrays of wavenumber bins and summed power
        within each shell, length floor(min(H, W) / 2) + 1.
    """
    if isinstance(field, Tensor):
        field = field.detach().cpu().numpy()
    h, w = field.shape[-2], field.shape[-1]
    fhat = np.fft.fft2(field)
    power = np.abs(fhat) ** 2 / (h * w)

    bins = _radial_bins(h, w)
    k_max = min(h, w) // 2
    k_bins = np.arange(0, k_max + 1)
    spectrum = np.zeros(len(k_bins))
    for i, k in enumerate(k_bins):
        mask = bins == k
        if mask.any():
            spectrum[i] = power[mask].sum()
    return k_bins, spectrum


def kinetic_energy_spectrum(u: Tensor | np.ndarray, v: Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged kinetic energy spectrum E(k) = 0.5 * (|u_hat|^2 + |v_hat|^2), shell-summed.

    Args:
        u, v: (H, W) velocity components.

    Returns:
        (k_bins, E_k).
    """
    k_bins, power_u = radial_power_spectrum(u)
    _, power_v = radial_power_spectrum(v)
    return k_bins, 0.5 * (power_u + power_v)


def batch_mean_energy_spectrum(u: Tensor, v: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Kinetic energy spectrum averaged over a batch.

    Args:
        u, v: (B, H, W) velocity components.

    Returns:
        (k_bins, mean_E_k over the batch).
    """
    u_np = u.detach().cpu().numpy() if isinstance(u, Tensor) else u
    v_np = v.detach().cpu().numpy() if isinstance(v, Tensor) else v
    spectra = []
    k_bins = None
    for i in range(u_np.shape[0]):
        k_bins, e_k = kinetic_energy_spectrum(u_np[i], v_np[i])
        spectra.append(e_k)
    return k_bins, np.mean(np.stack(spectra), axis=0)


def spectral_energy_error(u_a: Tensor, v_a: Tensor, u_b: Tensor, v_b: Tensor, log_space: bool = True) -> float:
    """Scalar discrepancy between two batches' mean energy spectra.

    Args:
        u_a, v_a: (B, H, W) velocity components, e.g. ground truth.
        u_b, v_b: (B, H, W) velocity components, e.g. generated.
        log_space: If True (default), compare log-power (matches the
            common "spectral log-PSD error" convention -- turbulence
            spectra span many orders of magnitude, so linear-space MSE
            is dominated entirely by the largest scales).

    Returns:
        Mean absolute error between the two radially-averaged spectra.
    """
    k_bins, e_a = batch_mean_energy_spectrum(u_a, v_a)
    _, e_b = batch_mean_energy_spectrum(u_b, v_b)
    eps = 1e-10
    if log_space:
        e_a, e_b = np.log10(e_a + eps), np.log10(e_b + eps)
    return float(np.mean(np.abs(e_a - e_b)))
