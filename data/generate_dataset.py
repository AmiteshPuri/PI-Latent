"""Dataset generation orchestrator.

Deliberately separate from both the physics solver (data/synthetic_solver.py,
data/pdebench_loader.py, data/the_well_loader.py) and the CLI entry point
(scripts/prepare_data.py): this module only handles calling the right
source-specific preparer, saving the standardised window format, and
skip-if-exists resumability. It has no CLI parsing and no PDE-solving code
of its own.

Every split is saved as a single .npz with:
    velocity: (N, 3, 2, H, W) float32   -- [prev, center, next] windows
    dt, dx, nu: float32 scalars
    periodic: int32 scalar (1 = spectral derivatives valid, 0 = use
              finite-difference + interior masking; see physics/derivatives.py)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from utils.registry import get_dataset_preparer

logger = logging.getLogger(__name__)

SPLITS = ["train", "val", "test"]


def _split_path(output_dir: Path, source: str, split: str) -> Path:
    return output_dir / f"{source}_{split}.npz"


def generate_split(
    source: str,
    split: str,
    n_windows: int,
    resolution: int,
    seed: int,
    source_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Prepare a single split via the registered source preparer.

    Args:
        source: 'pdebench' | 'the_well' | 'synthetic'.
        split: 'train' | 'val' | 'test'.
        n_windows: Number of (prev, center, next) windows.
        resolution: Spatial resolution NxN.
        seed: RNG seed.
        source_cfg: Source-specific config subtree (paths, dataset names, etc).

    Returns:
        Dict with 'velocity', 'dt', 'dx', 'nu', 'periodic' -- see module docstring.
    """
    preparer = get_dataset_preparer(source)
    logger.info(f"Preparing '{split}' from source='{source}': {n_windows} windows @ {resolution}x{resolution}")
    return preparer(split=split, n_windows=n_windows, resolution=resolution, seed=seed, cfg=source_cfg)


def generate_dataset(
    output_dir: str | Path,
    source: str,
    source_cfg: dict[str, Any],
    n_windows: dict[str, int],
    resolution: int,
    seed: int = 42,
    force: bool = False,
) -> dict[str, Path]:
    """Generate and persist train/val/test splits for one dataset source.

    Skip-aware: a split already on disk is left untouched unless `force`.

    Args:
        output_dir: Directory to write `<source>_<split>.npz` files.
        source: 'pdebench' | 'the_well' | 'synthetic'.
        source_cfg: Source-specific config subtree.
        n_windows: Dict mapping split name -> window count, e.g.
            {'train': 4000, 'val': 500, 'test': 500}.
        resolution: Spatial resolution NxN.
        seed: Base RNG seed; each split derives its own offset internally.
        force: Regenerate even if the split file already exists.

    Returns:
        Dict mapping split name -> path of the saved .npz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split in SPLITS:
        path = _split_path(output_dir, source, split)
        paths[split] = path
        if path.exists() and not force:
            logger.info(f"Found existing {path}, skipping (pass force=True to regenerate).")
            continue

        n = n_windows.get(split)
        if not n:
            logger.info(f"n_windows['{split}'] is 0 or unset; skipping split '{split}'.")
            continue

        result = generate_split(source, split, n, resolution, seed, source_cfg)
        np.savez_compressed(
            str(path),
            velocity=result["velocity"].astype(np.float32),
            dt=np.float32(result["dt"]),
            dx=np.float32(result["dx"]),
            nu=np.float32(result["nu"]),
            periodic=np.int32(1 if result["periodic"] else 0),
        )
        logger.info(f"Saved {path}  velocity={result['velocity'].shape}  nu={result['nu']:.3e}")

    return paths
