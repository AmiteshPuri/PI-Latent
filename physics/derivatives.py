"""Differentiable spatial derivative operators for 2D fields on a regular grid.

Array convention: a field tensor has shape (..., H, W). The last axis (W,
columns) is treated as the x-coordinate; the second-to-last axis (H, rows)
is treated as the y-coordinate. This matches how PDEBench/The Well store
fields and how they will be plotted (imshow's default row=y, col=x).

Two derivative backends are provided because the two named data sources
have different boundary conditions:

  - 'spectral' (FFT-based): exact for periodic BC. Use for The Well's
    shear_flow (explicitly 2D-periodic) and the classic FNO/TorusLi
    Navier-Stokes benchmark (periodic torus).
  - 'finite_diff' (central differences via circular shift): use for
    PDEBench's incompressible Navier-Stokes, which is generated with
    Dirichlet BC (velocity = 0 at the domain edges), not periodic. FFT
    derivatives would silently assume wraparound continuity that does not
    hold there and would corrupt the boundary rows/columns. The finite
    difference path is deliberately the same circular-shift stencil as the
    spectral case's real-space equivalent, but callers MUST apply
    `interior_mask` before reducing to a scalar residual so the invalid
    boundary ring (where the wraparound assumption is wrong) is excluded.

Both backends are pure tensor arithmetic (torch.fft or slicing/roll), so
gradients flow through them -- this is what lets the physics residual be
used as a training loss, not just a post-hoc diagnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor

Backend = str  # 'spectral' | 'finite_diff'


def _wavenumbers(n: int, dx: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Angular wavenumbers 2*pi*fftfreq for an FFT of length n with spacing dx."""
    return 2 * torch.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)


def _deriv_spectral(field: Tensor, axis: str, dx: float, order: int = 1) -> Tensor:
    """d^order(field)/d(axis)^order via FFT, assuming periodic BC.

    Args:
        field: (..., H, W) real tensor.
        axis: 'x' (last dim) or 'y' (second-to-last dim).
        dx: Grid spacing (assumed equal in x and y).
        order: 1 for first derivative, 2 for second (used by laplacian).

    Returns:
        (..., H, W) real tensor, same shape as field.
    """
    H, W = field.shape[-2], field.shape[-1]
    field_hat = torch.fft.fft2(field.to(torch.float32))
    ky = _wavenumbers(H, dx, field.device, torch.float32).view(H, 1)
    kx = _wavenumbers(W, dx, field.device, torch.float32).view(1, W)
    k = kx if axis == "x" else ky
    factor = (1j * k) ** order
    deriv_hat = factor * field_hat
    return torch.fft.ifft2(deriv_hat).real.to(field.dtype)


def _deriv_finite_diff(field: Tensor, axis: str, dx: float, order: int = 1) -> Tensor:
    """d^order(field)/d(axis)^order via central differences (circular shift).

    2nd-order accurate in the interior. Boundary rows/columns use the
    same wraparound stencil as the periodic case for simplicity and are
    only valid when the domain is actually periodic -- for non-periodic
    (Dirichlet) data, callers must exclude them with `interior_mask`
    before reducing to a scalar.
    """
    dim = -1 if axis == "x" else -2
    if order == 1:
        fwd = torch.roll(field, shifts=-1, dims=dim)
        bwd = torch.roll(field, shifts=1, dims=dim)
        return (fwd - bwd) / (2 * dx)
    if order == 2:
        fwd = torch.roll(field, shifts=-1, dims=dim)
        bwd = torch.roll(field, shifts=1, dims=dim)
        return (fwd - 2 * field + bwd) / (dx**2)
    raise ValueError(f"order must be 1 or 2, got {order}")


def ddx(field: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """First derivative along x (last axis)."""
    fn = _deriv_spectral if backend == "spectral" else _deriv_finite_diff
    return fn(field, "x", dx, order=1)


def ddy(field: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """First derivative along y (second-to-last axis)."""
    fn = _deriv_spectral if backend == "spectral" else _deriv_finite_diff
    return fn(field, "y", dx, order=1)


def laplacian(field: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """Laplacian: d^2/dx^2 + d^2/dy^2."""
    fn = _deriv_spectral if backend == "spectral" else _deriv_finite_diff
    return fn(field, "x", dx, order=2) + fn(field, "y", dx, order=2)


def divergence(u: Tensor, v: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """Divergence of a 2D vector field: du/dx + dv/dy.

    Args:
        u: (..., H, W) x-velocity component.
        v: (..., H, W) y-velocity component.
        dx: Grid spacing.
        backend: 'spectral' or 'finite_diff'.

    Returns:
        (..., H, W) divergence field. Zero everywhere for a true
        incompressible flow; the L2 norm of this is the "divergence
        error" diagnostic.
    """
    return ddx(u, dx, backend) + ddy(v, dx, backend)


def curl_2d(u: Tensor, v: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """Vorticity (2D curl) of a velocity field: omega = dv/dx - du/dy."""
    return ddx(v, dx, backend) - ddy(u, dx, backend)


def advect(u: Tensor, v: Tensor, scalar_field: Tensor, dx: float, backend: Backend = "spectral") -> Tensor:
    """Advection operator: (u . grad) scalar_field = u * d(f)/dx + v * d(f)/dy."""
    return u * ddx(scalar_field, dx, backend) + v * ddy(scalar_field, dx, backend)


def vorticity_to_velocity(omega: Tensor, dx: float) -> tuple[Tensor, Tensor]:
    """Recover a divergence-free velocity field from vorticity via the streamfunction.

    Solves laplacian(psi) = -omega spectrally, then u = d(psi)/dy, v = -d(psi)/dx.
    Periodic BC only (spectral solve for psi requires it). This is the
    inverse of `curl_2d` and is used by the synthetic solver (which
    integrates vorticity) to produce the velocity representation the rest
    of the pipeline operates on, and is also useful for converting any
    vorticity-only field to velocity.

    Args:
        omega: (..., H, W) vorticity field.
        dx: Grid spacing.

    Returns:
        (u, v): each (..., H, W).
    """
    H, W = omega.shape[-2], omega.shape[-1]
    ky = _wavenumbers(H, dx, omega.device, torch.float32).view(H, 1)
    kx = _wavenumbers(W, dx, omega.device, torch.float32).view(1, W)
    k2 = kx**2 + ky**2
    k2_safe = k2.clone()
    k2_safe[0, 0] = 1.0  # avoid division by zero; mean-mode streamfunction is arbitrary and set to 0 below

    omega_hat = torch.fft.fft2(omega.to(torch.float32))
    psi_hat = omega_hat / k2_safe
    psi_hat[..., 0, 0] = 0.0

    u_hat = 1j * ky * psi_hat
    v_hat = -1j * kx * psi_hat
    u = torch.fft.ifft2(u_hat).real.to(omega.dtype)
    v = torch.fft.ifft2(v_hat).real.to(omega.dtype)
    return u, v


def interior_mask(h: int, w: int, margin: int, device: torch.device) -> Tensor:
    """Boolean (H, W) mask that is False on a `margin`-pixel border.

    Use to exclude the boundary ring from residual norms when the
    finite-difference backend's circular-shift stencil is being applied
    to a non-periodic (Dirichlet) field, where the wraparound values at
    the edge are not physically meaningful.
    """
    mask = torch.ones(h, w, dtype=torch.bool, device=device)
    if margin > 0:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False
    return mask
