"""Loss functions for Stage 1 (VQ-VAE) and Stage 2 (flow matching).

Physics-loss scale note (the important design decision in this file):
the reconstruction loss is computed in normalised (z-score) space, where
targets are ~N(0, 1) by construction, but the PDE residual MUST be
computed in physical units (dx, dt, nu are physical constants -- the
governing equation does not hold in normalised units). Naively combining
a physical-unit residual with a normalised-space reconstruction loss is
exactly the failure mode documented in this project's history
(darcy_flow_pinn_vae: a physics-loss weight that needed retuning by
~7 orders of magnitude because the two loss terms lived at wildly
different scales). Instead of hand-tuning `lambda_physics` per dataset,
PhysicsLoss normalises each residual by a per-batch reference scale
(the RMS of the corresponding leading-order term), making both physics
loss terms dimensionless and O(1) by construction -- so a single
lambda_physics choice is meaningful across datasets/viscosities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from physics.derivatives import curl_2d, interior_mask
from physics.residual import divergence_error_field, vorticity_transport_residual_field


class ReconstructionLoss(nn.Module):
    """MSE reconstruction loss (primary VQ-VAE training objective, normalised space)."""

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(pred, target)


class PhysicsLoss(nn.Module):
    """Divergence-free penalty + vorticity-transport residual penalty, scale-normalised.

    Args:
        divergence_weight: Relative weight on the divergence term within
            the combined physics loss.
        residual_weight: Relative weight on the PDE residual term.
        boundary_margin: Pixels excluded from the loss at each edge when
            the domain is non-periodic (finite-difference backend) --
            see physics/derivatives.py's backend note. Ignored for
            periodic (spectral) domains.
    """

    def __init__(
        self,
        divergence_weight: float = 1.0,
        residual_weight: float = 1.0,
        boundary_margin: int = 2,
    ) -> None:
        super().__init__()
        self.divergence_weight = divergence_weight
        self.residual_weight = residual_weight
        self.boundary_margin = boundary_margin

    def forward(
        self,
        recon_velocity: Tensor,
        prev_velocity: Tensor,
        next_velocity: Tensor,
        dt: float,
        dx: float,
        nu: float,
        periodic: bool,
    ) -> dict[str, Tensor]:
        """
        Args:
            recon_velocity: (B, 2, H, W) reconstructed velocity, PHYSICAL units.
            prev_velocity: (B, 2, H, W) ground-truth velocity at t - dt, PHYSICAL units.
            next_velocity: (B, 2, H, W) ground-truth velocity at t + dt, PHYSICAL units.
            dt, dx, nu: Physical constants for this dataset (see data/datamodule.py).
            periodic: Whether the domain is periodic (spectral derivatives valid)
                or Dirichlet (finite-difference + interior mask).

        Returns:
            Dict with 'divergence_loss', 'residual_loss', 'total' (all
            dimensionless, O(1) scalars).
        """
        backend = "spectral" if periodic else "finite_diff"
        u, v = recon_velocity[:, 0], recon_velocity[:, 1]
        u_prev, v_prev = prev_velocity[:, 0], prev_velocity[:, 1]
        u_next, v_next = next_velocity[:, 0], next_velocity[:, 1]

        omega_prev = curl_2d(u_prev, v_prev, dx, backend)
        omega_next = curl_2d(u_next, v_next, dx, backend)
        domega_dt = (omega_next - omega_prev) / (2 * dt)

        div_field = divergence_error_field(u, v, dx, backend)
        res_field = vorticity_transport_residual_field(u, v, omega_prev, omega_next, dt, dx, nu, backend)

        mask = None
        if not periodic:
            h, w = u.shape[-2], u.shape[-1]
            mask = interior_mask(h, w, self.boundary_margin, u.device)

        omega_scale = _masked_rms(curl_2d(u, v, dx, backend), mask)
        domega_dt_scale = _masked_rms(domega_dt, mask)

        div_norm = div_field / (omega_scale.clamp(min=1e-6))
        res_norm = res_field / (domega_dt_scale.clamp(min=1e-6))

        divergence_loss = _masked_mse(div_norm, mask)
        residual_loss = _masked_mse(res_norm, mask)
        total = self.divergence_weight * divergence_loss + self.residual_weight * residual_loss

        return {"divergence_loss": divergence_loss, "residual_loss": residual_loss, "total": total}


def _masked_rms(field: Tensor, mask: Tensor | None) -> Tensor:
    """Per-sample RMS over (H, W), reshaped for broadcasting; optionally masked."""
    if mask is not None:
        field = field[:, mask]
    else:
        field = field.reshape(field.shape[0], -1)
    rms = torch.sqrt((field**2).mean(dim=-1) + 1e-12)
    return rms.view(-1, 1, 1)


def _masked_mse(field: Tensor, mask: Tensor | None) -> Tensor:
    if mask is not None:
        return (field[:, mask] ** 2).mean()
    return (field**2).mean()


class VQVAELossComputer:
    """Aggregates reconstruction + VQ + physics losses into `train/*` components.

    Physics loss is always computed for diagnostic visibility (so the
    physics-informed and baseline runs can be compared on equal footing
    in TensorBoard), but is only added under `torch.no_grad()` -- and
    therefore costs no autograd memory -- when `physics_weight == 0`
    (the baseline variant). This matters concretely on a 4 GB GPU: the
    residual computation runs several FFTs, and a baseline run should
    not pay for an autograd graph through them.

    Args:
        physics_weight: lambda_physics. 0.0 for the baseline variant,
            >0 for the physics-informed variant -- this single value is
            the only difference between the two variants (architecture
            is identical, per the project spec).
        divergence_weight, residual_weight, boundary_margin: Forwarded to PhysicsLoss.
    """

    def __init__(
        self,
        physics_weight: float = 0.0,
        divergence_weight: float = 1.0,
        residual_weight: float = 1.0,
        boundary_margin: int = 2,
    ) -> None:
        self.physics_weight = physics_weight
        self.reconstruction_loss_fn = ReconstructionLoss()
        self.physics_loss_fn = PhysicsLoss(divergence_weight, residual_weight, boundary_margin)

    def __call__(self, model_out: dict[str, Tensor], batch: dict[str, Tensor], dataset) -> dict[str, Tensor]:
        """
        Args:
            model_out: TransformerVQVAE.forward() output.
            batch: {'prev', 'center', 'next'} normalised velocity windows.
            dataset: The NS2DDataset instance (for `.denormalize`, `.dt`, `.dx`, `.nu`, `.periodic`).

        Returns:
            Dict with reconstruction_loss, vq_loss, physics_loss,
            divergence_loss, residual_loss, total_loss, perplexity --
            keys chosen to map directly onto the TensorBoard tags in
            training/callbacks.py.
        """
        reconstruction = model_out["reconstruction"]
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, batch["center"])
        vq_loss = model_out["commitment_loss"]

        physics_grad_ctx = torch.enable_grad() if self.physics_weight > 0 else torch.no_grad()
        with physics_grad_ctx:
            recon_phys = dataset.denormalize(reconstruction)
            prev_phys = dataset.denormalize(batch["prev"])
            next_phys = dataset.denormalize(batch["next"])
            physics_out = self.physics_loss_fn(
                recon_phys, prev_phys, next_phys, dataset.dt, dataset.dx, dataset.nu, dataset.periodic
            )

        physics_loss = physics_out["total"]
        total_loss = reconstruction_loss + vq_loss + self.physics_weight * physics_loss

        return {
            "reconstruction_loss": reconstruction_loss,
            "vq_loss": vq_loss,
            "physics_loss": physics_loss,
            "divergence_loss": physics_out["divergence_loss"],
            "residual_loss": physics_out["residual_loss"],
            "total_loss": total_loss,
            "perplexity": model_out["perplexity"],
        }


class FlowMatchingLoss(nn.Module):
    """Conditional flow matching / rectified flow regression loss.

    Straight-line interpolation path x_t = (1-t)*x0 + t*x1 between a
    Gaussian source x0 and a data target x1, with target velocity
    (x1 - x0) constant along the path (Lipman et al. 2023; Liu et al.
    2022). t ~ Uniform(0, 1) is sampled internally per call.
    """

    def forward(self, model: nn.Module, x1: Tensor) -> Tensor:
        """
        Args:
            model: LatentFlowMatcher.
            x1: (B, N, code_dim) target latent tokens (codebook embeddings
                of real data, from the frozen VQ-VAE).

        Returns:
            Scalar MSE loss between predicted and target velocity.
        """
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)

        t_broadcast = t.view(B, 1, 1)
        x_t = (1 - t_broadcast) * x0 + t_broadcast * x1
        target_velocity = x1 - x0

        predicted_velocity = model(x_t, t)
        return F.mse_loss(predicted_velocity, target_velocity)
