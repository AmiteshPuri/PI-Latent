"""Stage 2 trainer: latent Flow Matching over a frozen Stage-1 VQ-VAE.

The VQ-VAE is loaded once via training/checkpointing.load_frozen_model
(eval mode, requires_grad=False on every parameter) and re-used every
batch to turn real velocity-field windows into target codebook
embeddings on the fly -- see FlowMatchingTrainer._encode_batch_to_latents.
This avoids a separate latent-caching pipeline: the frozen forward pass
is cheap (no backward through it), and it keeps Stage 2 using the exact
same DataLoader/dataset as Stage 1, so there is only one dataset
implementation to keep correct.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from training.callbacks import Callback
from training.checkpointing import load_checkpoint
from training.losses import FlowMatchingLoss

logger = logging.getLogger(__name__)


class FlowMatchingTrainer:
    """Trains a LatentFlowMatcher against a frozen TransformerVQVAE's codebook.

    Args:
        flow_model: LatentFlowMatcher instance.
        vqvae: A TransformerVQVAE already loaded via
            training.checkpointing.load_frozen_model (eval mode, frozen).
        train_loader, val_loader: DataLoaders from data/datamodule.py
            (the same ones used for Stage 1 -- only `batch['center']` is used).
        lr, weight_decay, epochs, grad_clip: Optimisation hyperparameters.
        use_amp: Enable fp16 mixed precision (CUDA only).
        device: torch.device.
        callbacks: List of Callback instances.
    """

    def __init__(
        self,
        flow_model: torch.nn.Module,
        vqvae: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        lr: float,
        weight_decay: float,
        epochs: int,
        grad_clip: float,
        use_amp: bool,
        device: torch.device,
        callbacks: list[Callback],
    ) -> None:
        self.flow_model = flow_model.to(device)
        self.vqvae = vqvae  # already .to(device), .eval(), frozen
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.callbacks = callbacks
        self.epochs = epochs
        self.grad_clip = grad_clip

        self.optimiser = torch.optim.AdamW(self.flow_model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=epochs)
        self.loss_fn = FlowMatchingLoss()

        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.start_epoch = 1
        self.global_step = 0

    def resume_from_checkpoint(self, path: str | Path) -> None:
        ckpt = load_checkpoint(path, self.flow_model, self.optimiser, self.scheduler, self.device)
        self.start_epoch = int(ckpt["step"]) + 1
        self.global_step = self.start_epoch * len(self.train_loader)
        for cb in self.callbacks:
            if hasattr(cb, "set_resume_state"):
                cb.set_resume_state(ckpt)
        logger.info(f"Resuming flow-matching training from epoch {self.start_epoch}.")

    @torch.no_grad()
    def _encode_batch_to_latents(self, center: torch.Tensor) -> torch.Tensor:
        """Real velocity field -> frozen VQ-VAE codebook embeddings (B, N, code_dim)."""
        indices = self.vqvae.encode_to_indices(center)
        return self.vqvae.codebook.lookup(indices)

    def train_epoch(self, epoch: int) -> float:
        self.flow_model.train()
        total = 0.0
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [flow train]", leave=False)
        for batch in pbar:
            center = batch["center"].to(self.device, non_blocking=True)
            x1 = self._encode_batch_to_latents(center)

            self.optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss = self.loss_fn(self.flow_model, x1)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.grad_clip)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            loss_val = loss.item()
            total += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}")

            self.global_step += 1
            for cb in self.callbacks:
                if hasattr(cb, "log_train_step"):
                    cb.log_train_step({"flow_matching_loss": loss_val}, self.global_step)

        return total / n_batches

    @torch.no_grad()
    def validate(self) -> float:
        self.flow_model.eval()
        total = 0.0
        for batch in self.val_loader:
            center = batch["center"].to(self.device, non_blocking=True)
            x1 = self._encode_batch_to_latents(center)
            loss = self.loss_fn(self.flow_model, x1)
            total += loss.item()
        return total / len(self.val_loader)

    def fit(self) -> None:
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()

            ctx = {
                "epoch": epoch,
                "model": self.flow_model,
                "optimiser": self.optimiser,
                "scheduler": self.scheduler,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimiser.param_groups[0]["lr"],
            }
            for cb in self.callbacks:
                cb.on_epoch_end(ctx)

            logger.info(f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
