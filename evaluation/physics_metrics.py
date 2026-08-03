"""Physics diagnostic metrics: PDE residual norm and divergence error.

Thin evaluation-facing wrappers around physics/residual.py's
differentiable field computations -- these versions are for post-hoc
metric reporting (numpy in, float out), not for backprop; see
training/losses.py's PhysicsLoss for the differentiable training-time
version.
"""

from __future__ import annotations

import numpy as np
import torch

from physics.derivatives import curl_2d, interior_mask
from physics.residual import divergence_error_field, masked_l2_norm, vorticity_transport_residual_field


def divergence_error(u: np.ndarray, v: np.ndarray, dx: float, periodic: bool, boundary_margin: int = 2) -> np.ndarray:
    """Per-sample L2 norm of the divergence field.

    Args:
        u, v: (B, H, W) physical-unit velocity components.
        dx: Grid spacing.
        periodic: Whether to use the spectral (True) or finite-difference
            + interior-mask (False) backend.
        boundary_margin: Pixels excluded at each edge for the non-periodic case.

    Returns:
        (B,) array of per-sample divergence L2 norms.
    """
    u_t, v_t = torch.from_numpy(u).float(), torch.from_numpy(v).float()
    backend = "spectral" if periodic else "finite_diff"
    field = divergence_error_field(u_t, v_t, dx, backend)
    mask = None if periodic else interior_mask(u_t.shape[-2], u_t.shape[-1], boundary_margin, u_t.device)
    return masked_l2_norm(field, mask).numpy()


def pde_residual_norm(
    u: np.ndarray,
    v: np.ndarray,
    u_prev: np.ndarray,
    v_prev: np.ndarray,
    u_next: np.ndarray,
    v_next: np.ndarray,
    dt: float,
    dx: float,
    nu: float,
    periodic: bool,
    boundary_margin: int = 2,
) -> np.ndarray:
    """Per-sample L2 norm of the vorticity-transport PDE residual.

    Args:
        u, v: (B, H, W) reconstructed/generated velocity at the center timestep.
        u_prev, v_prev, u_next, v_next: (B, H, W) ground-truth velocity at
            the adjacent timesteps (for the time-derivative estimate).
        dt, dx, nu: Physical constants.
        periodic: Spectral vs finite-difference backend.
        boundary_margin: Pixels excluded at each edge for the non-periodic case.

    Returns:
        (B,) array of per-sample residual L2 norms.
    """
    backend = "spectral" if periodic else "finite_diff"

    def to_t(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).float()

    u_t, v_t = to_t(u), to_t(v)
    u_prev_t, v_prev_t = to_t(u_prev), to_t(v_prev)
    u_next_t, v_next_t = to_t(u_next), to_t(v_next)

    omega_prev = curl_2d(u_prev_t, v_prev_t, dx, backend)
    omega_next = curl_2d(u_next_t, v_next_t, dx, backend)
    field = vorticity_transport_residual_field(u_t, v_t, omega_prev, omega_next, dt, dx, nu, backend)

    mask = None if periodic else interior_mask(u_t.shape[-2], u_t.shape[-1], boundary_margin, u_t.device)
    return masked_l2_norm(field, mask).numpy()
