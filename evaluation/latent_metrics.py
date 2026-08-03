"""Codebook health metrics: perplexity and utilization.

Both are computed from indices ACCUMULATED over a full validation pass
(not a single batch) for a stable estimate -- a single batch of, say, 16
samples x 64 tokens = 1024 code assignments is a noisy sample of a
512-or-more-entry codebook's usage distribution.
"""

from __future__ import annotations

import numpy as np


def codebook_perplexity(indices: np.ndarray, num_codes: int) -> float:
    """exp(entropy) of the code usage distribution.

    Ranges from 1 (every token maps to the same single code -- collapsed)
    to num_codes (perfectly uniform usage -- the healthiest possible codebook).

    Args:
        indices: Flattened array of codebook indices (any shape, will be raveled).
        num_codes: Codebook size K.

    Returns:
        Scalar perplexity.
    """
    counts = np.bincount(indices.ravel(), minlength=num_codes).astype(np.float64)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def codebook_utilization(indices: np.ndarray, num_codes: int) -> float:
    """Fraction of codebook entries used at least once.

    Args:
        indices: Flattened array of codebook indices (any shape, will be raveled).
        num_codes: Codebook size K.

    Returns:
        Scalar in [0, 1].
    """
    n_used = np.unique(indices.ravel()).size
    return float(n_used / num_codes)


def compute_codebook_health(indices: np.ndarray, num_codes: int) -> dict[str, float]:
    """Convenience wrapper returning both perplexity and utilization."""
    return {
        "codebook_perplexity": codebook_perplexity(indices, num_codes),
        "codebook_utilization": codebook_utilization(indices, num_codes),
    }
