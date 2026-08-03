"""Training callbacks. All receive a shared context dict from the Trainer
at epoch end (`on_epoch_end(ctx)`), keeping the training loop itself
free of logging/checkpointing concerns -- same pattern as the reference
repo this project's structure follows.

TensorBoardCallback's tag names are fixed to exactly what the project
spec lists:
    train/reconstruction_loss, train/vq_loss, train/physics_loss, train/total_loss
    val/reconstruction_loss, val/physics_loss
    metrics/divergence_error, metrics/residual_norm, metrics/codebook_perplexity
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from training.checkpointing import save_checkpoint
from utils.io import append_csv

logger = logging.getLogger(__name__)


class Callback:
    """Base callback with a no-op epoch-end hook."""

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        pass


class TensorBoardCallback(Callback):
    """Writes train/*, val/*, metrics/* scalars and validation field figures.

    `log_train_step` is called every training step (fine-grained
    curves); `on_epoch_end` is called once per epoch after validation.
    """

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir))

    def log_train_step(self, losses: dict[str, float], step: int) -> None:
        self.writer.add_scalar("train/reconstruction_loss", losses["reconstruction_loss"], step)
        self.writer.add_scalar("train/vq_loss", losses["vq_loss"], step)
        self.writer.add_scalar("train/physics_loss", losses["physics_loss"], step)
        self.writer.add_scalar("train/total_loss", losses["total_loss"], step)

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        epoch = ctx["epoch"]
        val_losses = ctx["val_losses"]
        val_metrics = ctx["val_metrics"]

        self.writer.add_scalar("val/reconstruction_loss", val_losses["reconstruction_loss"], epoch)
        self.writer.add_scalar("val/physics_loss", val_losses["physics_loss"], epoch)
        self.writer.add_scalar("metrics/divergence_error", val_metrics["divergence_error"], epoch)
        self.writer.add_scalar("metrics/residual_norm", val_metrics["residual_norm"], epoch)
        self.writer.add_scalar("metrics/codebook_perplexity", val_metrics["codebook_perplexity"], epoch)
        # Not in the originally-listed tags but cheap and directly relevant to the same
        # "is the codebook healthy" question perplexity answers -- logged alongside it.
        if "codebook_utilization" in val_metrics:
            self.writer.add_scalar("metrics/codebook_utilization", val_metrics["codebook_utilization"], epoch)
        self.writer.add_scalar("lr", ctx["lr"], epoch)

        fig = ctx.get("viz_fig")
        if fig is not None:
            self.writer.add_figure("validation/fields", fig, epoch)
            plt.close(fig)

    def close(self) -> None:
        self.writer.close()


class CSVLoggerCallback(Callback):
    """Appends one row per epoch to a metrics CSV, mirroring the TensorBoard tags."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        train_losses = ctx["train_losses"]
        val_losses = ctx["val_losses"]
        val_metrics = ctx["val_metrics"]
        row = {
            "epoch": ctx["epoch"],
            "train_reconstruction_loss": train_losses["reconstruction_loss"],
            "train_vq_loss": train_losses["vq_loss"],
            "train_physics_loss": train_losses["physics_loss"],
            "train_total_loss": train_losses["total_loss"],
            "val_reconstruction_loss": val_losses["reconstruction_loss"],
            "val_physics_loss": val_losses["physics_loss"],
            "divergence_error": val_metrics["divergence_error"],
            "residual_norm": val_metrics["residual_norm"],
            "codebook_perplexity": val_metrics["codebook_perplexity"],
            "codebook_utilization": val_metrics.get("codebook_utilization", float("nan")),
            "lr": ctx["lr"],
        }
        write_header = not self.csv_path.exists()
        append_csv(self.csv_path, row, write_header=write_header)


class FlowMatchingTensorBoardCallback(Callback):
    """TensorBoard logging for Stage 2. No fixed tag list was specified for
    this stage (unlike Stage 1), so tags follow the same train/val/lr
    convention Stage 1 uses for consistency: train/flow_matching_loss,
    val/flow_matching_loss.
    """

    def __init__(self, log_dir: str | Path) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(str(log_dir))

    def log_train_step(self, losses: dict[str, float], step: int) -> None:
        self.writer.add_scalar("train/flow_matching_loss", losses["flow_matching_loss"], step)

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        self.writer.add_scalar("val/flow_matching_loss", ctx["val_loss"], ctx["epoch"])
        self.writer.add_scalar("lr", ctx["lr"], ctx["epoch"])

    def close(self) -> None:
        self.writer.close()


class FlowMatchingCSVLoggerCallback(Callback):
    """Appends one row per epoch to a Stage-2 metrics CSV."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        row = {
            "epoch": ctx["epoch"],
            "train_flow_matching_loss": ctx["train_loss"],
            "val_flow_matching_loss": ctx["val_loss"],
            "lr": ctx["lr"],
        }
        write_header = not self.csv_path.exists()
        append_csv(self.csv_path, row, write_header=write_header)


class CheckpointCallback(Callback):
    """Saves best.pt (on improvement), epoch_N.pt (at configured epochs), and latest.pt (every epoch).

    Stage-agnostic: reads `ctx["val_loss"]` as a flat scalar for model
    selection. The VQ-VAE trainer sets this to validation reconstruction
    loss specifically (comparable across the baseline and physics-
    informed variants, unlike total_loss, whose scale differs because
    only one of them backprops a non-zero physics term); the flow-
    matching trainer sets it to its own validation loss.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        run_name: str,
        checkpoint_epochs: list[int] | None = None,
    ) -> None:
        self.ckpt_dir = Path(checkpoint_dir) / run_name
        self.run_name = run_name
        self.checkpoint_epochs = set(checkpoint_epochs or [])
        self.best_val_loss = float("inf")

    def set_resume_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore the best-loss tracker when resuming, so a resumed run
        cannot overwrite a true best with a worse "new" best."""
        best = checkpoint.get("best_val_loss")
        if best is None:
            best = checkpoint.get("val_loss", float("inf"))
        self.best_val_loss = float(best)

    def on_epoch_end(self, ctx: dict[str, Any]) -> None:
        epoch = ctx["epoch"]
        val_loss = ctx["val_loss"]  # flat scalar -- VQ-VAE trainer sets this to val reconstruction_loss,
        # flow-matching trainer sets it to val flow-matching loss, so this callback is stage-agnostic.
        model, optimiser, scheduler = ctx["model"], ctx["optimiser"], ctx["scheduler"]

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            save_checkpoint(
                model, optimiser, scheduler, epoch, val_loss,
                self.ckpt_dir / "best.pt", best_val_loss=self.best_val_loss,
            )
            logger.info(f"[{self.run_name}] New best val_loss={val_loss:.6f} at epoch {epoch}.")

        if epoch in self.checkpoint_epochs:
            save_checkpoint(
                model, optimiser, scheduler, epoch, val_loss,
                self.ckpt_dir / f"epoch_{epoch}.pt", best_val_loss=self.best_val_loss,
            )

        # Always-current checkpoint: this is what makes "run again and it resumes" work.
        save_checkpoint(
            model, optimiser, scheduler, epoch, val_loss,
            self.ckpt_dir / "latest.pt", best_val_loss=self.best_val_loss,
        )
