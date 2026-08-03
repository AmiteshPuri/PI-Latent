"""CLI: train the latent Flow Matching generator (Stage 2), over a frozen Stage-1 VQ-VAE.

    python scripts/train_vqvae.py --data_config data_synthetic --vqvae_config vqvae_physics
    python scripts/train_flow_matching.py --data_config data_synthetic

Reads configs/flow_matching.yaml's `vqvae_checkpoint`/`vqvae_config` to
locate and rebuild the frozen Stage-1 model. --data_config should
normally match whatever dataset that checkpoint was trained on (codebook
indices are only meaningful for the data distribution the VQ-VAE saw).

Auto-resume: same convention as train_vqvae.py -- re-run the same
command to continue from outputs/checkpoints/<run_name>/latest.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from data.datamodule import build_dataloaders  # noqa: E402
from data.generate_dataset import generate_dataset  # noqa: E402
from models.flow_matching.model import LatentFlowMatcher  # noqa: E402
from models.vqvae.model import TransformerVQVAE  # noqa: E402
from training.callbacks import (  # noqa: E402
    CheckpointCallback,
    FlowMatchingCSVLoggerCallback,
    FlowMatchingTensorBoardCallback,
)
from training.checkpointing import load_frozen_model  # noqa: E402
from training.trainer_flow import FlowMatchingTrainer  # noqa: E402
from utils.config import CONFIG_DIR, config_to_dict, load_config, load_flow_matching_config, resolve_vqvae_arch_config  # noqa: E402
from utils.io import save_json  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the latent Flow Matching generator (Stage 2).")
    parser.add_argument("--data_config", default="data_synthetic")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-path overrides into configs/flow_matching.yaml, e.g. eval.n_ode_steps=25",
    )
    args = parser.parse_args()

    overrides = OmegaConf.from_dotlist(args.override) if args.override else None
    flow_cfg = load_flow_matching_config(overrides=overrides)
    experiment_cfg = load_config(CONFIG_DIR / "experiment.yaml")

    run_name = args.run_name or flow_cfg.run_name
    set_seed(experiment_cfg.seed)
    device = get_device()

    vqvae_ckpt_path = Path(flow_cfg.vqvae_checkpoint)
    if not vqvae_ckpt_path.exists():
        raise FileNotFoundError(
            f"vqvae_checkpoint ({vqvae_ckpt_path}) does not exist -- train Stage 1 first:\n"
            f"  python scripts/train_vqvae.py --vqvae_config {flow_cfg.vqvae_config}\n"
            f"or update configs/flow_matching.yaml's vqvae_checkpoint to an existing run."
        )

    vqvae_arch_cfg = resolve_vqvae_arch_config(vqvae_ckpt_path, flow_cfg.vqvae_config, experiment_cfg.output_dir)
    vqvae = TransformerVQVAE(
        in_channels=vqvae_arch_cfg.in_channels,
        resolution=vqvae_arch_cfg.resolution,
        patch_size=vqvae_arch_cfg.patch_size,
        embed_dim=vqvae_arch_cfg.embed_dim,
        encoder_depth=vqvae_arch_cfg.encoder_depth,
        decoder_depth=vqvae_arch_cfg.decoder_depth,
        n_heads=vqvae_arch_cfg.n_heads,
        mlp_ratio=vqvae_arch_cfg.mlp_ratio,
        num_codes=vqvae_arch_cfg.num_codes,
        code_dim=vqvae_arch_cfg.code_dim,
        commitment_weight=vqvae_arch_cfg.commitment_weight,
        ema_decay=vqvae_arch_cfg.ema_decay,
        reset_after_n_batches=vqvae_arch_cfg.reset_after_n_batches,
        dropout=vqvae_arch_cfg.dropout,
    )
    vqvae = load_frozen_model(vqvae, vqvae_ckpt_path, device=device)
    logger.info(f"Loaded frozen VQ-VAE from {vqvae_ckpt_path} (config={flow_cfg.vqvae_config})")

    data_cfg = load_config(CONFIG_DIR / f"{args.data_config}.yaml")
    generate_dataset(
        output_dir=experiment_cfg.data_root,
        source=data_cfg.source,
        source_cfg=config_to_dict(data_cfg),
        n_windows=config_to_dict(experiment_cfg.n_windows),
        resolution=experiment_cfg.resolution,
        seed=experiment_cfg.seed,
    )
    loaders = build_dataloaders(
        experiment_cfg.data_root, data_cfg.source,
        batch_size=experiment_cfg.batch_size, num_workers=experiment_cfg.num_workers,
    )

    flow_model = LatentFlowMatcher(
        grid_shape=vqvae.grid_shape,
        code_dim=vqvae.codebook.code_dim,
        embed_dim=flow_cfg.embed_dim,
        depth=flow_cfg.depth,
        n_heads=flow_cfg.n_heads,
        mlp_ratio=flow_cfg.mlp_ratio,
        dropout=flow_cfg.dropout,
    )

    output_dir = Path(experiment_cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tensorboard" / run_name
    metrics_csv = output_dir / "metrics" / f"{run_name}.csv"
    metadata_path = output_dir / "metadata" / f"{run_name}.json"

    tb_callback = FlowMatchingTensorBoardCallback(tb_dir)
    callbacks = [
        tb_callback,
        FlowMatchingCSVLoggerCallback(metrics_csv),
        CheckpointCallback(checkpoint_dir, run_name, checkpoint_epochs=list(flow_cfg.checkpoint_epochs)),
    ]

    trainer = FlowMatchingTrainer(
        flow_model=flow_model,
        vqvae=vqvae,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        lr=flow_cfg.lr,
        weight_decay=flow_cfg.weight_decay,
        epochs=flow_cfg.epochs,
        grad_clip=flow_cfg.grad_clip,
        use_amp=flow_cfg.use_amp,
        device=device,
        callbacks=callbacks,
    )

    latest_ckpt = checkpoint_dir / run_name / "latest.pt"
    if latest_ckpt.exists():
        trainer.resume_from_checkpoint(latest_ckpt)
        if trainer.start_epoch > flow_cfg.epochs:
            logger.info(f"'{run_name}' already completed ({trainer.start_epoch - 1}/{flow_cfg.epochs} epochs). Nothing to do.")
            tb_callback.close()
            return
    else:
        logger.info(f"No existing checkpoint for '{run_name}'; starting from epoch 1.")

    save_json(
        {"run_name": run_name, "vqvae_checkpoint": str(vqvae_ckpt_path), "config": config_to_dict(flow_cfg)},
        metadata_path,
    )

    trainer.fit()
    tb_callback.close()
    logger.info(f"Done. Best checkpoint: {checkpoint_dir / run_name / 'best.pt'}")


if __name__ == "__main__":
    main()
