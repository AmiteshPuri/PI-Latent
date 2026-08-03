"""CLI: generate train/val/test splits for one dataset source.

    python scripts/prepare_data.py --data_config data_synthetic
    python scripts/prepare_data.py --data_config data_pdebench --force

Thin wrapper only -- see data/generate_dataset.py for the actual logic
and data/synthetic_solver.py, data/pdebench_loader.py,
data/the_well_loader.py for the per-source preparation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf  # noqa: E402

from data.generate_dataset import generate_dataset  # noqa: E402
from utils.config import CONFIG_DIR, config_to_dict, load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.seed import set_seed  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NS2D train/val/test splits for one dataset source.")
    parser.add_argument("--data_config", default="data_synthetic", help="Basename of configs/data_*.yaml.")
    parser.add_argument("--resolution", type=int, default=None, help="Overrides experiment.yaml's resolution.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Regenerate even if a split already exists.")
    parser.add_argument(
        "--override", nargs="*", default=[],
        help="Dot-path overrides, e.g. n_windows.train=200 data.window_stride=2",
    )
    args = parser.parse_args()

    experiment_cfg = load_config(CONFIG_DIR / "experiment.yaml")
    data_cfg = load_config(CONFIG_DIR / f"{args.data_config}.yaml")
    cfg = OmegaConf.merge(experiment_cfg, {"data": data_cfg})
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.override))

    set_seed(args.seed)
    paths = generate_dataset(
        output_dir=cfg.data_root,
        source=cfg.data.source,
        source_cfg=config_to_dict(cfg.data),
        n_windows=config_to_dict(cfg.n_windows),
        resolution=args.resolution or cfg.resolution,
        seed=args.seed,
        force=args.force,
    )
    for split, path in paths.items():
        logger.info(f"{split}: {path}")


if __name__ == "__main__":
    main()
