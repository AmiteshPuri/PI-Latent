"""Pseudospectral 2D incompressible Navier-Stokes solver (vorticity form).

This is deliberately NOT the primary data source -- it exists so the
pipeline has a real, physically-generated dataset that requires no
download, for (a) smoke_test.py, which must run in seconds with no
network access, and (b) as a documented fallback if a PDEBench/The Well
download is unavailable. For actual research results, use
data_pdebench.yaml or data_the_well.yaml.

Method: periodic-domain pseudospectral integration of

    d(omega)/dt + u . grad(omega) = nu * laplacian(omega)

with velocity recovered from the streamfunction (laplacian(psi) = -omega).
Time-stepping uses Lie operator splitting: the diffusion term is
integrated exactly in Fourier space (it is linear and diagonal there),
and the advection term is integrated explicitly with Heun's method
(2nd-order Runge-Kutta). The nonlinear advection term is dealiased with
the standard 2/3 rule before transforming back to Fourier space. This is
a standard, well-documented combination for pseudospectral Navier-Stokes
codes (see e.g. Boyd, "Chebyshev and Fourier Spectral Methods", 2001,
Ch. 10; Canuto et al., "Spectral Methods in Fluid Dynamics", 1988) --
first-order accurate in time from the splitting, which is adequate for
generating qualitatively realistic training/smoke-test trajectories but
is not claimed to match a research-grade solver's accuracy.

Initial conditions are Gaussian random fields in vorticity with an
isotropic, smoothly-decaying energy spectrum, matching the general
approach used to generate the standard FNO/TorusLi Navier-Stokes
benchmark (Li et al., 2021).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from physics.derivatives import vorticity_to_velocity


def _wavenumber_grids(n: int, dx: float, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    """Return (kx, ky, k2) as (N, N) grids for an NxN periodic domain."""
    k1d = 2 * torch.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=torch.float32)
    ky, kx = torch.meshgrid(k1d, k1d, indexing="ij")
    return kx, ky, kx**2 + ky**2


def _dealias_mask(n: int, device: torch.device) -> Tensor:
    """2/3-rule dealiasing mask: zero out the outer third of wavenumbers."""
    k1d = torch.fft.fftfreq(n, device=device) * n
    ky, kx = torch.meshgrid(k1d, k1d, indexing="ij")
    cutoff = (n / 2) * (2.0 / 3.0)
    return ((kx.abs() <= cutoff) & (ky.abs() <= cutoff)).to(torch.float32)


def _random_initial_vorticity(n: int, dx: float, rng: np.random.Generator, device: torch.device) -> Tensor:
    """Gaussian random field initial vorticity with a smooth, decaying isotropic spectrum."""
    kx, ky, k2 = _wavenumber_grids(n, dx, device)
    k_mag = torch.sqrt(k2)

    # Smooth isotropic spectrum peaked at low-to-mid wavenumbers, decaying at high k --
    # produces large coherent eddies rather than pure noise, qualitatively similar to
    # freely-decaying 2D turbulence initial conditions.
    k_peak = 4.0
    amplitude = (k_mag**2) * torch.exp(-(k_mag / k_peak) ** 2)

    phase = torch.from_numpy(rng.uniform(0, 2 * torch.pi, size=(n, n))).to(torch.float32)
    omega_hat = amplitude * torch.exp(1j * phase)
    omega_hat[0, 0] = 0.0  # zero-mean vorticity

    omega = torch.fft.ifft2(omega_hat).real
    omega = omega / (omega.std() + 1e-8) * 2.0  # normalise to a consistent RMS vorticity
    return omega


def rollout_trajectory(
    resolution: int,
    dx: float,
    dt: float,
    n_steps: int,
    nu: float,
    seed: int,
    warmup_steps: int = 50,
) -> Tensor:
    """Integrate a single 2D NS trajectory and return the vorticity history.

    Args:
        resolution: Grid size N (domain is NxN, periodic).
        dx: Grid spacing.
        dt: Timestep.
        n_steps: Number of stored steps to return (after warmup).
        nu: Kinematic viscosity.
        seed: RNG seed for the initial condition.
        warmup_steps: Steps to integrate before recording, so the stored
            trajectory starts from a statistically-developed flow rather
            than the raw Gaussian random field initial condition.

    Returns:
        (n_steps, resolution, resolution) vorticity field history.
    """
    device = torch.device("cpu")
    rng = np.random.default_rng(seed)
    n = resolution

    kx, ky, k2 = _wavenumber_grids(n, dx, device)
    dealias = _dealias_mask(n, device)
    diffusion_factor = torch.exp(-nu * k2 * dt)

    omega = _random_initial_vorticity(n, dx, rng, device)
    omega_hat = torch.fft.fft2(omega)

    def nonlinear_rhs(omega_hat: Tensor) -> Tensor:
        """-u . grad(omega) in Fourier space, dealiased."""
        k2_safe = k2.clone()
        k2_safe[0, 0] = 1.0
        psi_hat = omega_hat / k2_safe
        psi_hat[0, 0] = 0.0

        u = torch.fft.ifft2(1j * ky * psi_hat).real
        v = torch.fft.ifft2(-1j * kx * psi_hat).real
        omega_x = torch.fft.ifft2(1j * kx * omega_hat).real
        omega_y = torch.fft.ifft2(1j * ky * omega_hat).real

        advected = u * omega_x + v * omega_y
        return -torch.fft.fft2(advected) * dealias

    def step(omega_hat: Tensor) -> Tensor:
        # Exact integration of the linear diffusion term.
        omega_hat = omega_hat * diffusion_factor
        # Heun (RK2) for the nonlinear advection term.
        k1 = nonlinear_rhs(omega_hat)
        pred = omega_hat + dt * k1
        k2_ = nonlinear_rhs(pred)
        return omega_hat + dt * 0.5 * (k1 + k2_)

    for _ in range(warmup_steps):
        omega_hat = step(omega_hat)

    history = torch.empty(n_steps, n, n)
    for t in range(n_steps):
        omega_hat = step(omega_hat)
        history[t] = torch.fft.ifft2(omega_hat).real

    return history


def prepare_synthetic_split(
    split: str,
    n_windows: int,
    resolution: int,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Generate `n_windows` (prev, center, next) velocity windows via the solver.

    Matches the dataset-preparer contract used by data/generate_dataset.py
    and registered in utils/registry.py.

    Args:
        split: 'train' | 'val' | 'test' (only affects the seed offset, so
            splits are statistically independent).
        n_windows: Number of 3-frame windows to produce.
        resolution: Grid resolution (NxN).
        seed: Base RNG seed.
        cfg: Source config subtree; reads `dx`, `dt`, `nu`,
            `steps_per_trajectory` (windows drawn per rollout), with
            sensible defaults if absent.

    Returns:
        Dict with keys 'velocity' (N, 3, 2, H, W) float32, 'dt', 'dx',
        'nu', 'periodic' (always True -- the solver assumes a periodic
        domain).
    """
    dx = float(cfg.get("dx", 2 * np.pi / resolution))
    dt = float(cfg.get("dt", 0.01))
    nu = float(cfg.get("nu", 1e-3))
    stride = int(cfg.get("window_stride", 4))  # spacing (in solver steps) between the 3 stored frames

    split_offsets = {"train": 0, "val": 10_000, "test": 20_000}
    base_seed = seed + split_offsets.get(split, 0)

    windows = np.empty((n_windows, 3, 2, resolution, resolution), dtype=np.float32)
    windows_per_traj = 8
    n_trajectories = int(np.ceil(n_windows / windows_per_traj))

    idx = 0
    for traj_i in range(n_trajectories):
        traj_seed = base_seed + traj_i
        n_steps_needed = windows_per_traj * stride + 2 * stride
        vort_history = rollout_trajectory(resolution, dx, dt, n_steps_needed, nu, traj_seed)

        for w in range(windows_per_traj):
            if idx >= n_windows:
                break
            center_t = (w + 1) * stride
            for k, t in enumerate((center_t - stride, center_t, center_t + stride)):
                u, v = vorticity_to_velocity(vort_history[t], dx)
                windows[idx, k, 0] = u.numpy()
                windows[idx, k, 1] = v.numpy()
            idx += 1
        if idx >= n_windows:
            break

    return {
        "velocity": windows[:idx],
        "dt": dt * stride,
        "dx": dx,
        "nu": nu,
        "periodic": True,
    }
