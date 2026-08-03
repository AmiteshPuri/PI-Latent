"""Checkpoint saving and loading, with atomic writes (see utils/io.py) so an
interrupted save can never leave a corrupt `latest.pt` that a resumed run
would fail to load.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from utils.io import atomic_torch_save

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    scheduler: object,
    step: int,
    val_loss: float,
    path: str | Path,
    best_val_loss: float | None = None,
    extra: dict | None = None,
) -> None:
    """Save a full training checkpoint to disk (atomically).

    Args:
        model: The model being trained.
        optimiser: The optimiser.
        scheduler: The LR scheduler.
        step: Current epoch (or step) number.
        val_loss: Validation loss at this checkpoint.
        path: Destination .pt file path.
        best_val_loss: Best validation loss seen so far, stored for safe
            resumption so an interrupted run cannot forget the true best.
        extra: Any additional payload (e.g. VQ-VAE checkpoint path, for
            the flow-matching stage).
    """
    payload = {
        "step": step,
        "val_loss": val_loss,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
    }
    if best_val_loss is not None:
        payload["best_val_loss"] = best_val_loss
    if extra:
        payload.update(extra)
    atomic_torch_save(payload, path)
    logger.debug(f"Saved checkpoint -> {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None = None,
    scheduler: object | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a checkpoint from disk.

    Args:
        path: Source .pt file path.
        model: Model to load weights into.
        optimiser: Optional optimiser to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors to.

    Returns:
        The raw checkpoint dict (contains 'step', 'val_loss', etc.).
    """
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimiser is not None and "optimiser" in ckpt:
        optimiser.load_state_dict(ckpt["optimiser"])
    if scheduler is not None and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Loaded checkpoint from {path}  (step={ckpt.get('step')}, val_loss={ckpt.get('val_loss', 'N/A')})")
    return ckpt


def load_frozen_model(model: nn.Module, path: str | Path, device: torch.device | None = None) -> nn.Module:
    """Load weights for inference-only use (Stage 2 loading the frozen Stage-1 VQ-VAE).

    Sets the model to eval() and disables gradient tracking on all
    parameters, since a frozen upstream model should never be perturbed
    by Stage-2 backprop even if it accidentally ends up in a computation
    graph.
    """
    if device is None:
        device = torch.device("cpu")
    load_checkpoint(path, model, device=device)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
