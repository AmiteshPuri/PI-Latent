"""Stage 2 end-to-end generation evaluation: sample from the flow matcher,
decode through the frozen VQ-VAE, and compute every metric Figures 4-7
need. This is the single function the final analysis notebook calls for
its Stage-2 results, so the notebook itself stays free of modelling code.

PDE-residual-for-generated-samples note (a real modelling choice, stated
plainly rather than glossed over): the residual formula needs a ground-
truth d(omega)/dt from adjacent timesteps (see physics/residual.py), but
a freshly generated snapshot has no "next timestep" -- flow matching
generates i.i.d. samples from the learned marginal distribution over
center frames, not trajectories. This function pairs each generated
sample with a *randomly drawn* real (prev, next) context from the
validation set to supply that derivative. This does not test whether any
specific generated field is dynamically consistent with a particular
trajectory (that question is not well-posed for an unconditional
generator); it tests whether the generated field's own spatial structure
(advection, diffusion terms) is consistent with the *typical* local
time-evolution rate seen in the real data -- i.e. whether generated
fields have plausible lengthscales/gradients for this dynamical system,
not just plausible marginal statistics. The exact-pairing residual on
real *reconstructions* (`pde_residual_norm_real_mean`) is reported
alongside it as the meaningful reference point for that same quantity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from evaluation import distribution_metrics, physics_metrics
from physics.derivatives import curl_2d
from physics.residual import divergence_error_field, masked_l2_norm
from physics.spectral import batch_mean_energy_spectrum


@torch.no_grad()
def evaluate_generation(
    flow_model: torch.nn.Module,
    vqvae: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    n_samples: int = 256,
    n_ode_steps: int = 50,
    ode_method: str = "euler",
    seed: int = 0,
) -> dict[str, Any]:
    """Generate samples and compute every Stage-2 evaluation metric.

    Args:
        flow_model: Trained LatentFlowMatcher (eval mode expected).
        vqvae: Frozen TransformerVQVAE (eval mode, requires_grad=False).
        val_loader: DataLoader supplying real (prev, center, next) windows.
        device: torch.device.
        n_samples: Number of samples to draw (capped at validation set size).
        n_ode_steps: Flow-matching ODE integration steps (NFE).
        ode_method: 'euler' or 'heun'.
        seed: RNG seed for both sampling and the residual-pairing permutation.

    Returns:
        Dict with scalar metrics ('mmd', 'sliced_wasserstein_distance',
        'spectral_energy_error', 'pde_residual_norm_generated_mean',
        'pde_residual_norm_real_mean', 'divergence_error_generated_mean')
        and the raw arrays Figures 4-7 need to plot
        ('residual_gt', 'residual_gen', 'real_latents_pooled',
        'gen_latents_pooled', 'k_bins', 'e_gt', 'e_gen', 'omega_gt',
        'omega_gen', 'gen_fields_phys', 'real_gt_phys').
    """
    flow_model.eval()
    vqvae.eval()
    dataset = val_loader.dataset
    backend = "spectral" if dataset.periodic else "finite_diff"

    real_centers, real_prevs, real_nexts = _gather_real_windows(val_loader, n_samples)
    n = real_centers.shape[0]

    real_indices = vqvae.encode_to_indices(real_centers.to(device))
    real_latents = vqvae.codebook.lookup(real_indices)
    real_recon = vqvae.decode(real_latents)

    generator = torch.Generator(device=device).manual_seed(seed) if device.type == "cuda" else torch.Generator().manual_seed(seed)
    gen_latents_continuous = flow_model.sample(n, device, n_steps=n_ode_steps, method=ode_method, generator=generator)
    gen_latents = vqvae.codebook.nearest_codes(gen_latents_continuous)
    gen_fields = vqvae.decode(gen_latents)

    gen_phys = dataset.denormalize(gen_fields).cpu().numpy()
    real_recon_phys = dataset.denormalize(real_recon).cpu().numpy()
    real_gt_phys = dataset.denormalize(real_centers.to(device)).cpu().numpy()
    prev_phys = dataset.denormalize(real_prevs.to(device)).cpu().numpy()
    next_phys = dataset.denormalize(real_nexts.to(device)).cpu().numpy()

    real_latents_pooled = real_latents.mean(dim=1).cpu().numpy()
    gen_latents_pooled = gen_latents_continuous.mean(dim=1).cpu().numpy()
    mmd_val = distribution_metrics.rbf_mmd(real_latents_pooled, gen_latents_pooled)
    swd_val = distribution_metrics.sliced_wasserstein_distance(real_latents_pooled, gen_latents_pooled)
    spectral_err = distribution_metrics.spectral_energy_error(
        real_gt_phys[:, 0], real_gt_phys[:, 1], gen_phys[:, 0], gen_phys[:, 1]
    )

    residual_real = physics_metrics.pde_residual_norm(
        real_recon_phys[:, 0], real_recon_phys[:, 1],
        prev_phys[:, 0], prev_phys[:, 1], next_phys[:, 0], next_phys[:, 1],
        dataset.dt, dataset.dx, dataset.nu, dataset.periodic,
    )
    rng = np.random.default_rng(seed)
    pairing = rng.permutation(n)  # see module docstring: randomised (prev, next) context for generated samples
    residual_gen = physics_metrics.pde_residual_norm(
        gen_phys[:, 0], gen_phys[:, 1],
        prev_phys[pairing, 0], prev_phys[pairing, 1], next_phys[pairing, 0], next_phys[pairing, 1],
        dataset.dt, dataset.dx, dataset.nu, dataset.periodic,
    )

    gen_u, gen_v = torch.from_numpy(gen_phys[:, 0]).float(), torch.from_numpy(gen_phys[:, 1]).float()
    div_field = divergence_error_field(gen_u, gen_v, dataset.dx, backend)
    div_error_gen = masked_l2_norm(div_field).numpy()

    omega_gt = curl_2d(
        torch.from_numpy(real_gt_phys[:, 0]).float(), torch.from_numpy(real_gt_phys[:, 1]).float(), dataset.dx, backend
    ).numpy()
    omega_gen = curl_2d(gen_u, gen_v, dataset.dx, backend).numpy()

    k_bins, e_gt = batch_mean_energy_spectrum(
        torch.from_numpy(real_gt_phys[:, 0]).float(), torch.from_numpy(real_gt_phys[:, 1]).float()
    )
    _, e_gen = batch_mean_energy_spectrum(gen_u, gen_v)

    return {
        "mmd": mmd_val,
        "sliced_wasserstein_distance": swd_val,
        "spectral_energy_error": spectral_err,
        "pde_residual_norm_generated_mean": float(residual_gen.mean()),
        "pde_residual_norm_real_mean": float(residual_real.mean()),
        "divergence_error_generated_mean": float(div_error_gen.mean()),
        "residual_gt": residual_real,
        "residual_gen": residual_gen,
        "real_latents_pooled": real_latents_pooled,
        "gen_latents_pooled": gen_latents_pooled,
        "k_bins": k_bins,
        "e_gt": e_gt,
        "e_gen": e_gen,
        "omega_gt": omega_gt,
        "omega_gen": omega_gen,
        "gen_fields_phys": gen_phys,
        "real_gt_phys": real_gt_phys,
    }


def _gather_real_windows(
    val_loader: torch.utils.data.DataLoader, n_samples: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate (prev, center, next) across the val_loader, capped at n_samples."""
    centers, prevs, nexts = [], [], []
    collected = 0
    for batch in val_loader:
        centers.append(batch["center"])
        prevs.append(batch["prev"])
        nexts.append(batch["next"])
        collected += batch["center"].shape[0]
        if collected >= n_samples:
            break
    return (
        torch.cat(centers)[:n_samples],
        torch.cat(prevs)[:n_samples],
        torch.cat(nexts)[:n_samples],
    )
