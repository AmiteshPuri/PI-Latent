"""Reconstruction quality metrics, operating on numpy arrays post-inference."""

from __future__ import annotations

import numpy as np


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((pred - target) ** 2))


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    """Relative L^2 error: ||pred - target||_2 / ||target||_2."""
    diff_norm = np.linalg.norm(pred - target)
    target_norm = np.linalg.norm(target)
    return float(diff_norm / (target_norm + 1e-8))


def psnr(pred: np.ndarray, target: np.ndarray, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio, generalised to arbitrary-range fields.

    PSNR = 10 * log10(data_range^2 / MSE). For image-free scientific
    fields (no fixed [0, 255] range), data_range defaults to the
    ground-truth field's own dynamic range (max - min), the standard
    generalisation used by e.g. skimage's peak_signal_noise_ratio when
    an explicit data_range is supplied.

    Args:
        pred, target: Arrays of matching shape.
        data_range: Optional explicit dynamic range; computed from
            `target` if not given.

    Returns:
        PSNR in dB. +inf if pred == target exactly.
    """
    error = mse(pred, target)
    if error == 0:
        return float("inf")
    if data_range is None:
        data_range = float(target.max() - target.min())
        if data_range == 0:
            data_range = 1.0
    return float(10 * np.log10((data_range**2) / error))


def batch_reconstruction_metrics(preds: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample MSE, relative L2, and PSNR for a batch.

    Args:
        preds, targets: (B, ...) arrays.

    Returns:
        Dict of 'mse', 'relative_l2', 'psnr', each shape (B,).
    """
    B = preds.shape[0]
    out = {"mse": np.zeros(B), "relative_l2": np.zeros(B), "psnr": np.zeros(B)}
    for i in range(B):
        out["mse"][i] = mse(preds[i], targets[i])
        out["relative_l2"][i] = relative_l2(preds[i], targets[i])
        out["psnr"][i] = psnr(preds[i], targets[i])
    return out
