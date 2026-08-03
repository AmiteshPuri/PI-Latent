"""Stage 1 trainer: the Transformer VQ-VAE (baseline or physics-informed,
selected entirely by `physics_weight` in the loss computer -- the model
and trainer code are identical for both variants).

Recipe: AdamW + CosineAnnealingLR + gradient clipping + AMP (fp16),
matching the reference repo's GTX 1650-oriented conventions. Every
epoch's checkpoint write is atomic (training/checkpointing.py), so an
interrupted run's `latest.pt` is always either the previous or the
current epoch, never a half-written file -- resume_from_checkpoint then
picks up from exactly that epoch.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from evaluation import latent_metrics, physics_metrics
from physics.derivatives import curl_2d
from physics.residual import divergence_error_field, vorticity_transport_residual_field
from training.callbacks import Callback
from training.checkpointing import load_checkpoint
from training.losses import VQVAELossComputer
from utils.visualization import plot_validation_fields

logger = logging.getLogger(__name__)


class VQVAETrainer:
    """Trains a TransformerVQVAE end to end, with validation, checkpointing,
    resume, and callback-driven logging.

    Args:
        model: TransformerVQVAE instance.
        train_loader, val_loader: DataLoaders from data/datamodule.py.
        physics_weight: lambda_physics (0.0 = baseline, >0 = physics-informed).
        lr, weight_decay, epochs, grad_clip: Optimisation hyperparameters.
        use_amp: Enable fp16 mixed precision (only takes effect on CUDA).
        device: torch.device.
        callbacks: List of Callback instances (TensorBoard, checkpoint, CSV, ...).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        physics_weight: float,
        lr: float,
        weight_decay: float,
        epochs: int,
        grad_clip: float,
        use_amp: bool,
        device: torch.device,
        callbacks: list[Callback],
        divergence_weight: float = 1.0,
        residual_weight: float = 1.0,
        boundary_margin: int = 2,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.callbacks = callbacks
        self.epochs = epochs
        self.grad_clip = grad_clip

        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=epochs)
        self.loss_computer = VQVAELossComputer(
            physics_weight=physics_weight,
            divergence_weight=divergence_weight,
            residual_weight=residual_weight,
            boundary_margin=boundary_margin,
        )

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 1
        self.global_step = 0

    def resume_from_checkpoint(self, path: str | Path) -> None:
        """Restore model/optimiser/scheduler state and continue from the next epoch."""
        ckpt = load_checkpoint(path, self.model, self.optimiser, self.scheduler, self.device)
        self.start_epoch = int(ckpt["step"]) + 1
        self.global_step = self.start_epoch * len(self.train_loader)
        for cb in self.callbacks:
            if hasattr(cb, "set_resume_state"):
                cb.set_resume_state(ckpt)
        logger.info(f"Resuming from epoch {self.start_epoch}.")

    def _batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        accum: dict[str, float] = defaultdict(float)
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for batch in pbar:
            batch = self._batch_to_device(batch)
            self.optimiser.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                model_out = self.model(batch["center"])
                losses = self.loss_computer(model_out, batch, self.train_loader.dataset)

            self.scaler.scale(losses["total_loss"]).backward()
            self.scaler.unscale_(self.optimiser)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            step_losses = {k: v.item() for k, v in losses.items() if k != "perplexity"}
            for k, v in step_losses.items():
                accum[k] += v
            pbar.set_postfix(recon=f"{step_losses['reconstruction_loss']:.4f}")

            self.global_step += 1
            for cb in self.callbacks:
                if hasattr(cb, "log_train_step"):
                    cb.log_train_step(step_losses, self.global_step)

        return {k: v / n_batches for k, v in accum.items()}

    @torch.no_grad()
    def validate(self, epoch: int) -> tuple[dict[str, float], dict[str, float], Any]:
        self.model.eval()
        dataset = self.val_loader.dataset
        accum: dict[str, float] = defaultdict(float)
        all_indices: list[np.ndarray] = []
        div_errors: list[np.ndarray] = []
        res_norms: list[np.ndarray] = []
        viz_fig = None
        n_batches = len(self.val_loader)

        for batch_idx, batch in enumerate(self.val_loader):
            batch = self._batch_to_device(batch)
            model_out = self.model(batch["center"])
            losses = self.loss_computer(model_out, batch, dataset)

            for k in ("reconstruction_loss", "vq_loss", "physics_loss", "total_loss"):
                accum[k] += losses[k].item()
            all_indices.append(model_out["indices"].cpu().numpy())

            recon_phys = dataset.denormalize(model_out["reconstruction"]).cpu().numpy()
            prev_phys = dataset.denormalize(batch["prev"]).cpu().numpy()
            next_phys = dataset.denormalize(batch["next"]).cpu().numpy()
            center_phys = dataset.denormalize(batch["center"]).cpu().numpy()

            div_errors.append(
                physics_metrics.divergence_error(recon_phys[:, 0], recon_phys[:, 1], dataset.dx, dataset.periodic)
            )
            res_norms.append(
                physics_metrics.pde_residual_norm(
                    recon_phys[:, 0], recon_phys[:, 1],
                    prev_phys[:, 0], prev_phys[:, 1],
                    next_phys[:, 0], next_phys[:, 1],
                    dataset.dt, dataset.dx, dataset.nu, dataset.periodic,
                )
            )

            if batch_idx == 0:
                viz_fig = self._build_validation_figure(
                    center_phys[0], recon_phys[0], prev_phys[0], next_phys[0], dataset
                )

        val_losses = {k: v / n_batches for k, v in accum.items()}
        flat_indices = np.concatenate([idx.ravel() for idx in all_indices])
        codebook_health = latent_metrics.compute_codebook_health(flat_indices, self.model.codebook.num_codes)

        val_metrics = {
            "divergence_error": float(np.mean(np.concatenate(div_errors))),
            "residual_norm": float(np.mean(np.concatenate(res_norms))),
            "codebook_perplexity": codebook_health["codebook_perplexity"],
            "codebook_utilization": codebook_health["codebook_utilization"],
        }
        return val_losses, val_metrics, viz_fig

    def _build_validation_figure(
        self,
        gt: np.ndarray,
        recon: np.ndarray,
        prev: np.ndarray,
        nxt: np.ndarray,
        dataset,
    ):
        """Build the 4-panel-family validation figure for a single sample (physical units)."""
        backend = "spectral" if dataset.periodic else "finite_diff"
        to_t = lambda a: torch.from_numpy(a).unsqueeze(0).float()  # noqa: E731

        u, v = to_t(recon[0]), to_t(recon[1])
        u_prev, v_prev = to_t(prev[0]), to_t(prev[1])
        u_next, v_next = to_t(nxt[0]), to_t(nxt[1])

        omega_prev = curl_2d(u_prev, v_prev, dataset.dx, backend)
        omega_next = curl_2d(u_next, v_next, dataset.dx, backend)
        residual_field = vorticity_transport_residual_field(
            u, v, omega_prev, omega_next, dataset.dt, dataset.dx, dataset.nu, backend
        )[0].numpy()
        divergence_field = divergence_error_field(u, v, dataset.dx, backend)[0].numpy()

        return plot_validation_fields(gt, recon, residual_field, divergence_field)

    def fit(self) -> None:
        """Run training from self.start_epoch through self.epochs, inclusive."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_losses = self.train_epoch(epoch)
            val_losses, val_metrics, viz_fig = self.validate(epoch)
            self.scheduler.step()

            ctx = {
                "epoch": epoch,
                "model": self.model,
                "optimiser": self.optimiser,
                "scheduler": self.scheduler,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_loss": val_losses["reconstruction_loss"],  # flat scalar for CheckpointCallback
                "val_metrics": val_metrics,
                "viz_fig": viz_fig,
                "lr": self.optimiser.param_groups[0]["lr"],
            }
            for cb in self.callbacks:
                cb.on_epoch_end(ctx)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"train_recon={train_losses['reconstruction_loss']:.6f} "
                f"val_recon={val_losses['reconstruction_loss']:.6f} "
                f"physics={val_losses['physics_loss']:.4f} "
                f"perplexity={val_metrics['codebook_perplexity']:.1f} "
                f"util={val_metrics['codebook_utilization']:.2%}"
            )
