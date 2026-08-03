"""Physics residuals for 2D incompressible Navier-Stokes.

Governing equations (velocity u = (u, v), vorticity omega = dv/dx - du/dy):

    Incompressibility:   div(u) = du/dx + dv/dy = 0
    Vorticity transport: d(omega)/dt + (u . grad) omega = nu * laplacian(omega) + f

The vorticity-transport form is used instead of the raw momentum equation
because it eliminates the pressure term entirely (see e.g. Li et al. 2021,
"Fourier Neural Operator for Parametric PDEs", and the standard
vorticity-streamfunction formulation used across the neural-operator
literature) -- there is no pressure channel to reconstruct, so a residual
written in terms of velocity and pressure would be unusable.

Modelling assumption (stated explicitly, not hidden): the VQ-VAE
reconstructs a single spatial snapshot, not a trajectory, so it has no way
to produce d(omega)/dt itself. That term is instead estimated from the
dataset's own adjacent time frames (ground truth), while the spatial terms
(advection, diffusion) use the model's *reconstructed* field. The residual
therefore measures whether the reconstruction is spatially self-consistent
with the true local time-evolution, not a closed-loop rollout error. This
is a standard compromise for snapshot-level (as opposed to sequence-level)
physics-informed autoencoders and is documented here so it is not mistaken
for a full space-time residual.

External forcing f is omitted by default (forcing_fn=None), matching
unforced decaying turbulence (e.g. The Well shear_flow, TorusLi). Pass a
forcing_fn(x, y) callable for forced settings (e.g. Kolmogorov flow).
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from physics.derivatives import advect, curl_2d, divergence, laplacian


def divergence_error_field(u: Tensor, v: Tensor, dx: float, backend: str = "spectral") -> Tensor:
    """Pointwise divergence field; should be ~0 for incompressible flow.

    Args:
        u, v: (B, H, W) reconstructed velocity components.
        dx: Grid spacing.
        backend: 'spectral' or 'finite_diff'.

    Returns:
        (B, H, W) divergence field.
    """
    return divergence(u, v, dx, backend)


def vorticity_transport_residual_field(
    u: Tensor,
    v: Tensor,
    omega_prev: Tensor,
    omega_next: Tensor,
    dt: float,
    dx: float,
    nu: float,
    backend: str = "spectral",
    forcing_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Pointwise vorticity-transport PDE residual field.

    R = d(omega)/dt + (u . grad) omega - nu * laplacian(omega) - f

    Args:
        u, v: (B, H, W) reconstructed velocity components at the center
            (reconstructed) timestep t.
        omega_prev: (B, H, W) ground-truth vorticity at t - dt.
        omega_next: (B, H, W) ground-truth vorticity at t + dt.
        dt: Time spacing between prev/next and the center frame.
        dx: Spatial grid spacing.
        nu: Kinematic viscosity.
        backend: 'spectral' or 'finite_diff'.
        forcing_fn: Optional callable(x_grid, y_grid) -> forcing field.

    Returns:
        (B, H, W) residual field. Near-zero everywhere for a physically
        consistent reconstruction.
    """
    omega = curl_2d(u, v, dx, backend)
    domega_dt = (omega_next - omega_prev) / (2 * dt)
    advection_term = advect(u, v, omega, dx, backend)
    diffusion_term = nu * laplacian(omega, dx, backend)

    residual = domega_dt + advection_term - diffusion_term
    if forcing_fn is not None:
        h, w = u.shape[-2], u.shape[-1]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 2 * torch.pi, h, device=u.device),
            torch.linspace(0, 2 * torch.pi, w, device=u.device),
            indexing="ij",
        )
        residual = residual - forcing_fn(xx, yy)
    return residual


def masked_l2_norm(field: Tensor, mask: Tensor | None = None) -> Tensor:
    """Per-sample L2 norm of a (B, H, W) field, optionally restricted to a mask.

    Args:
        field: (B, H, W) tensor.
        mask: Optional (H, W) boolean mask; False entries are excluded.

    Returns:
        (B,) tensor of per-sample L2 norms.
    """
    if mask is not None:
        field = field * mask.to(field.dtype)
        n_valid = mask.sum().clamp(min=1)
    else:
        n_valid = field.shape[-2] * field.shape[-1]
    return torch.sqrt((field**2).sum(dim=(-2, -1)) / n_valid)
