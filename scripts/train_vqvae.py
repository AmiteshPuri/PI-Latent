"""CLI: train the Transformer VQ-VAE (Stage 1).

    python scripts/train_vqvae.py --data_config data_synthetic --vqvae_config vqvae_baseline
    python scripts/train_vqvae.py --data_config data_synthetic --vqvae_config vqvae_physics
    python scripts/train_vqvae.py --data_config data_pdebench --vqvae_config vqvae_physics \
        --override vqvae.num_codes=1024 lr=1e-4

Auto-resume: if outputs/checkpoints/<run_name>/latest.pt already exists,
training continues from the next epoch after it -- just re-run the same
command. A run that already reached the target epoch count is a no-op.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from data.datamodule import build_dataloaders  # noqa: E402
from data.generate_dataset import generate_dataset  # noqa: E402
from models.vqvae.model import TransformerVQVAE  # noqa: E402
from training.callbacks import CheckpointCallback, CSVLoggerCallback, TensorBoardCallback  # noqa: E402
from training.trainer_vqvae import VQVAETrainer  # noqa: E402
from utils.config import config_to_dict, load_experiment_configs  # noqa: E402
from utils.device import device_info, get_device  # noqa: E402
from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Transformer VQ-VAE (Stage 1).")
    parser.add_argument("--data_config", default="data_synthetic")
    parser.add_argument("--vqvae_config", default="vqvae_baseline")
    parser.add_argument("--run_name", default=None, help="Defaults to '<run_name>_<data_config>_<vqvae_config>'.")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-path config overrides, e.g. vqvae.physics_weight=0.2 lr=1e-4",
    )
    args = parser.parse_args()

    overrides = OmegaConf.from_dotlist(args.override) if args.override else None
    cfg = load_experiment_configs(args.data_config, args.vqvae_config, overrides=overrides)

    data_tag = args.data_config.replace("data_", "")
    vqvae_tag = args.vqvae_config.replace("vqvae_", "")
    run_name = args.run_name or f"{cfg.run_name}_{data_tag}_{vqvae_tag}"

    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Run '{run_name}' on device={device} | data={data_tag} vqvae={vqvae_tag}")

    output_dir = Path(cfg.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    tb_dir = output_dir / "tensorboard" / run_name
    metrics_csv = output_dir / "metrics" / f"{run_name}.csv"
    metadata_path = output_dir / "metadata" / f"{run_name}.json"

    generate_dataset(
        output_dir=cfg.data_root,
        source=cfg.data.source,
        source_cfg=config_to_dict(cfg.data),
        n_windows=config_to_dict(cfg.n_windows),
        resolution=cfg.resolution,
        seed=cfg.seed,
    )
    loaders = build_dataloaders(cfg.data_root, cfg.data.source, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    model = TransformerVQVAE(
        in_channels=cfg.vqvae.in_channels,
        resolution=cfg.resolution,
        patch_size=cfg.vqvae.patch_size,
        embed_dim=cfg.vqvae.embed_dim,
        encoder_depth=cfg.vqvae.encoder_depth,
        decoder_depth=cfg.vqvae.decoder_depth,
        n_heads=cfg.vqvae.n_heads,
        mlp_ratio=cfg.vqvae.mlp_ratio,
        num_codes=cfg.vqvae.num_codes,
        code_dim=cfg.vqvae.code_dim,
        commitment_weight=cfg.vqvae.commitment_weight,
        ema_decay=cfg.vqvae.ema_decay,
        reset_after_n_batches=cfg.vqvae.reset_after_n_batches,
        dropout=cfg.vqvae.dropout,
    )

    tb_callback = TensorBoardCallback(tb_dir)
    callbacks = [
        tb_callback,
        CSVLoggerCallback(metrics_csv),
        CheckpointCallback(checkpoint_dir, run_name, checkpoint_epochs=list(cfg.checkpoint_epochs)),
    ]

    trainer = VQVAETrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        physics_weight=cfg.vqvae.physics_weight,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        epochs=cfg.epochs,
        grad_clip=cfg.grad_clip,
        use_amp=cfg.use_amp,
        device=device,
        callbacks=callbacks,
        divergence_weight=cfg.vqvae.divergence_weight,
        residual_weight=cfg.vqvae.residual_weight,
        boundary_margin=cfg.vqvae.boundary_margin,
    )

    latest_ckpt = checkpoint_dir / run_name / "latest.pt"
    if latest_ckpt.exists():
        trainer.resume_from_checkpoint(latest_ckpt)
        if trainer.start_epoch > cfg.epochs:
            logger.info(f"'{run_name}' already completed ({trainer.start_epoch - 1}/{cfg.epochs} epochs). Nothing to do.")
            tb_callback.close()
            return
    else:
        logger.info(f"No existing checkpoint for '{run_name}'; starting from epoch 1.")

    save_json({"run_name": run_name, "config": config_to_dict(cfg), **device_info()}, metadata_path)

    trainer.fit()
    tb_callback.close()
    logger.info(f"Done. Best checkpoint: {checkpoint_dir / run_name / 'best.pt'}")


if __name__ == "__main__":
    main()
