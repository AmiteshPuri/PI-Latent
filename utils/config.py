"""Config loading and merging utilities using OmegaConf.

Configs are split into three composable pieces so datasets, VQ-VAE
variants, and the flow-matching stage can each be swept independently:

    configs/experiment.yaml        <- seed, output_dir, training defaults
    configs/data_<source>.yaml     <- dataset choice: pdebench | the_well | synthetic
    configs/vqvae_<variant>.yaml   <- baseline | physics
    configs/flow_matching.yaml     <- stage-2 generator config
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> DictConfig:
    """Load a YAML config file into an OmegaConf DictConfig."""
    return OmegaConf.load(str(path))


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple DictConfig objects (later ones override earlier ones)."""
    return OmegaConf.merge(*configs)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert DictConfig to a plain Python dict for JSON serialisation."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def load_experiment_configs(
    data_config: str = "data_synthetic",
    vqvae_config: str = "vqvae_baseline",
    config_dir: str | Path = CONFIG_DIR,
    overrides: DictConfig | dict | None = None,
) -> DictConfig:
    """Load and merge experiment + data + VQ-VAE configs.

    Args:
        data_config: Basename (no .yaml) of the dataset config, e.g.
            'data_pdebench', 'data_the_well', 'data_synthetic'.
        vqvae_config: Basename of the VQ-VAE variant config, e.g.
            'vqvae_baseline', 'vqvae_physics'.
        config_dir: Directory containing config YAML files.
        overrides: Optional dict/DictConfig merged in last (highest priority) --
            this is how sweeps inject per-run parameter overrides.

    Returns:
        Merged DictConfig for the full experiment.
    """
    config_dir = Path(config_dir)
    experiment = load_config(config_dir / "experiment.yaml")
    data = load_config(config_dir / f"{data_config}.yaml")
    vqvae = load_config(config_dir / f"{vqvae_config}.yaml")

    merged = OmegaConf.merge(experiment, {"data": data, "vqvae": vqvae})
    if overrides is not None:
        merged = OmegaConf.merge(merged, overrides)
    return merged


def load_flow_matching_config(
    config_dir: str | Path = CONFIG_DIR,
    overrides: DictConfig | dict | None = None,
) -> DictConfig:
    """Load the stage-2 flow-matching config (standalone, since it composes
    with whichever experiment config produced the frozen VQ-VAE checkpoint).
    """
    config_dir = Path(config_dir)
    cfg = load_config(config_dir / "flow_matching.yaml")
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def resolve_vqvae_arch_config(
    vqvae_checkpoint_path: str | Path,
    fallback_config_name: str,
    output_dir: str | Path = "outputs",
    config_dir: str | Path = CONFIG_DIR,
) -> DictConfig:
    """Resolve the exact architecture config a VQ-VAE checkpoint was trained with.

    Reading configs/<fallback_config_name>.yaml (plus experiment.yaml's
    resolution) directly is only correct if that checkpoint's training
    run used no architecture-affecting --override (e.g. vqvae.embed_dim,
    vqvae.num_codes, or resolution itself, which also determines
    grid_shape/pos_embed size) -- which sweeps (configs/sweeps/
    vqvae_sweep.yaml sweeps vqvae.num_codes) routinely do. This instead
    prefers the run's own saved metadata JSON (outputs/metadata/
    <run_name>.json, written by scripts/train_vqvae.py with the fully-
    resolved config, run_name inferred from the checkpoint's parent
    directory), which reflects any overrides that were actually used.
    Falls back to the static YAMLs only if no metadata file is found.

    Args:
        vqvae_checkpoint_path: Path to the checkpoint (e.g. .../best.pt);
            its parent directory name is taken as the run_name.
        fallback_config_name: Basename of configs/vqvae_*.yaml to fall
            back to if no metadata is found.
        output_dir: The experiment's output_dir (for locating metadata/).
        config_dir: Directory containing config YAML files.

    Returns:
        Flat DictConfig with `resolution` plus every vqvae_*.yaml
        architecture field (patch_size, embed_dim, ...) -- exactly the
        fields TransformerVQVAE's constructor needs.
    """
    run_name = Path(vqvae_checkpoint_path).parent.name
    metadata_path = Path(output_dir) / "metadata" / f"{run_name}.json"

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        full_cfg = metadata.get("config", {})
        vqvae_cfg = full_cfg.get("vqvae")
        resolution = full_cfg.get("resolution")
        if vqvae_cfg is not None and resolution is not None:
            return OmegaConf.create({**vqvae_cfg, "resolution": resolution})
        logger.warning(f"{metadata_path} is missing 'config.vqvae' or 'config.resolution'; falling back to static YAML.")
    else:
        logger.warning(
            f"No metadata found at {metadata_path}; falling back to configs/{fallback_config_name}.yaml + "
            f"experiment.yaml's resolution. This is only accurate if that checkpoint's Stage-1 run used no "
            f"architecture-affecting --override."
        )

    fallback_vqvae = load_config(Path(config_dir) / f"{fallback_config_name}.yaml")
    fallback_experiment = load_config(Path(config_dir) / "experiment.yaml")
    merged = dict(config_to_dict(fallback_vqvae))
    merged["resolution"] = fallback_experiment.resolution
    return OmegaConf.create(merged)


def resolve_flow_matching_arch_config(
    flow_checkpoint_path: str | Path,
    output_dir: str | Path = "outputs",
    config_dir: str | Path = CONFIG_DIR,
) -> DictConfig:
    """Resolve the exact architecture config a flow-matching checkpoint was trained with.

    Same rationale as resolve_vqvae_arch_config: reading
    configs/flow_matching.yaml directly is only correct if no
    architecture-affecting --override (embed_dim, depth, n_heads,
    mlp_ratio) was used at training time. Prefers the run's own saved
    metadata JSON (outputs/metadata/<run_name>.json, written by
    scripts/train_flow_matching.py), falling back to the static YAML
    only if no metadata file is found.

    Args:
        flow_checkpoint_path: Path to the checkpoint (e.g. .../best.pt);
            its parent directory name is taken as the run_name.
        output_dir: The experiment's output_dir (for locating metadata/).
        config_dir: Directory containing config YAML files.

    Returns:
        DictConfig with at least embed_dim, depth, n_heads, mlp_ratio, dropout.
    """
    run_name = Path(flow_checkpoint_path).parent.name
    metadata_path = Path(output_dir) / "metadata" / f"{run_name}.json"

    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        cfg = metadata.get("config")
        if cfg is not None:
            return OmegaConf.create(cfg)
        logger.warning(f"{metadata_path} has no 'config' key; falling back to configs/flow_matching.yaml.")
    else:
        logger.warning(
            f"No metadata found at {metadata_path}; falling back to configs/flow_matching.yaml. "
            f"This is only accurate if that checkpoint's Stage-2 run used no architecture-affecting --override."
        )
    return load_config(Path(config_dir) / "flow_matching.yaml")
