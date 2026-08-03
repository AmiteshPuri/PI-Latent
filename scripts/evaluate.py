"""CLI: evaluate a trained flow-matching generator against its frozen VQ-VAE.

    python scripts/evaluate.py --run_name ns2d_flow_matching --data_config data_synthetic

Saves a JSON metrics summary and the Figure 4/5/7-equivalent PNGs to
outputs/, independent of the notebook -- useful for sweeps and CI, where
opening Jupyter for every combination is impractical. notebooks/
final_analysis.ipynb is the narrative deliverable with all 7 figures;
this script covers the subset that depends only on Stage-2 generation
(not Stage-1 training curves, which come from the CSV logs directly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datamodule import build_dataloaders  # noqa: E402
from evaluation.evaluate_generation import evaluate_generation  # noqa: E402
from models.flow_matching.model import LatentFlowMatcher  # noqa: E402
from models.vqvae.model import TransformerVQVAE  # noqa: E402
from training.checkpointing import load_frozen_model  # noqa: E402
from utils.config import CONFIG_DIR, load_config, load_flow_matching_config, resolve_flow_matching_arch_config, resolve_vqvae_arch_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.io import save_json  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_distribution_comparison,
    plot_energy_spectra,
    plot_vorticity_comparison,
)

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained flow-matching generator.")
    parser.add_argument("--run_name", default=None, help="Flow-matching run to evaluate; defaults to configs/flow_matching.yaml's run_name.")
    parser.add_argument("--data_config", default="data_synthetic")
    args = parser.parse_args()

    experiment_cfg = load_config(CONFIG_DIR / "experiment.yaml")
    flow_cfg = load_flow_matching_config()
    run_name = args.run_name or flow_cfg.run_name
    device = get_device()

    output_dir = Path(experiment_cfg.output_dir)
    flow_ckpt = output_dir / "checkpoints" / run_name / "best.pt"
    if not flow_ckpt.exists():
        raise FileNotFoundError(f"{flow_ckpt} not found -- train Stage 2 first: python scripts/train_flow_matching.py")

    vqvae_arch_cfg = resolve_vqvae_arch_config(flow_cfg.vqvae_checkpoint, flow_cfg.vqvae_config, experiment_cfg.output_dir)
    vqvae = TransformerVQVAE(
        in_channels=vqvae_arch_cfg.in_channels, resolution=vqvae_arch_cfg.resolution,
        patch_size=vqvae_arch_cfg.patch_size, embed_dim=vqvae_arch_cfg.embed_dim,
        encoder_depth=vqvae_arch_cfg.encoder_depth, decoder_depth=vqvae_arch_cfg.decoder_depth,
        n_heads=vqvae_arch_cfg.n_heads, mlp_ratio=vqvae_arch_cfg.mlp_ratio,
        num_codes=vqvae_arch_cfg.num_codes, code_dim=vqvae_arch_cfg.code_dim,
        commitment_weight=vqvae_arch_cfg.commitment_weight, ema_decay=vqvae_arch_cfg.ema_decay,
        reset_after_n_batches=vqvae_arch_cfg.reset_after_n_batches, dropout=vqvae_arch_cfg.dropout,
    )
    vqvae = load_frozen_model(vqvae, flow_cfg.vqvae_checkpoint, device=device)

    flow_arch_cfg = resolve_flow_matching_arch_config(flow_ckpt, experiment_cfg.output_dir)
    flow_model = LatentFlowMatcher(
        grid_shape=vqvae.grid_shape, code_dim=vqvae.codebook.code_dim,
        embed_dim=flow_arch_cfg.embed_dim, depth=flow_arch_cfg.depth,
        n_heads=flow_arch_cfg.n_heads, mlp_ratio=flow_arch_cfg.mlp_ratio, dropout=flow_arch_cfg.dropout,
    )
    flow_model = load_frozen_model(flow_model, flow_ckpt, device=device)

    data_cfg = load_config(CONFIG_DIR / f"{args.data_config}.yaml")
    loaders = build_dataloaders(
        experiment_cfg.data_root, data_cfg.source,
        batch_size=experiment_cfg.batch_size, num_workers=experiment_cfg.num_workers,
    )

    results = evaluate_generation(
        flow_model, vqvae, loaders["val"], device,
        n_samples=flow_cfg.eval.n_samples, n_ode_steps=flow_cfg.eval.n_ode_steps, ode_method=flow_cfg.eval.ode_method,
    )

    summary = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    logger.info(f"Generation evaluation for '{run_name}': {summary}")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics" / f"{run_name}_generation_eval.json"
    save_json({"run_name": run_name, **summary}, metrics_path)

    plot_energy_spectra(results["k_bins"], results["e_gt"], results["e_gen"]).savefig(
        figures_dir / f"{run_name}_energy_spectra.png", dpi=150, bbox_inches="tight"
    )
    plot_vorticity_comparison(results["omega_gt"], results["omega_gen"]).savefig(
        figures_dir / f"{run_name}_vorticity_comparison.png", dpi=150, bbox_inches="tight"
    )
    plot_distribution_comparison(
        results["residual_gt"], results["residual_gen"], results["mmd"], results["sliced_wasserstein_distance"]
    ).savefig(figures_dir / f"{run_name}_distribution_comparison.png", dpi=150, bbox_inches="tight")

    logger.info(f"Saved metrics -> {metrics_path}")
    logger.info(f"Saved figures -> {figures_dir}/{run_name}_*.png")


if __name__ == "__main__":
    main()
