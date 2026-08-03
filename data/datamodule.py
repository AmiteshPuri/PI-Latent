"""PyTorch Dataset and DataLoader utilities for the NS2D VQ-VAE experiments.

NS2DDataset wraps a `<source>_<split>.npz` file produced by
data/generate_dataset.py and returns normalised (prev, center, next)
velocity-field triples. The VQ-VAE reconstructs `center`; `prev`/`next`
are ground truth and used only for the physics loss's time-derivative
estimate (see physics/residual.py) -- they are never encoded/decoded.

Normalisation uses per-dataset z-score statistics computed on the
training split and shared with val/test, matching standard practice for
comparing splits on equal footing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class Batch(TypedDict):
    prev: Tensor
    center: Tensor
    next: Tensor


class NS2DDataset(Dataset):
    """Torch Dataset wrapping a pre-generated NS2D window .npz file.

    Each sample is a dict {'prev', 'center', 'next'}, each (2, H, W)
    normalised velocity (float32). Dataset-level physics metadata (dt,
    dx, nu, periodic) is exposed as attributes, not per-sample, since it
    is constant across the split.
    """

    def __init__(
        self,
        npz_path: str | Path,
        vel_mean: float | None = None,
        vel_std: float | None = None,
    ) -> None:
        """
        Args:
            npz_path: Path to a .npz file with 'velocity' (N, 3, 2, H, W)
                and scalar 'dt', 'dx', 'nu', 'periodic' keys.
            vel_mean: Optional pre-computed mean for normalisation
                (shared from the training split).
            vel_std: Optional pre-computed std for normalisation.
        """
        data = np.load(str(npz_path))
        self.velocity: np.ndarray = data["velocity"].astype(np.float32)  # (N, 3, 2, H, W)
        self.dt = float(data["dt"])
        self.dx = float(data["dx"])
        self.nu = float(data["nu"])
        self.periodic = bool(data["periodic"])

        self.vel_mean = vel_mean if vel_mean is not None else float(self.velocity.mean())
        self.vel_std = vel_std if vel_std is not None else float(self.velocity.std() + 1e-8)

        self.resolution = self.velocity.shape[-1]
        logger.debug(
            f"Dataset {npz_path}: {len(self)} windows, {self.resolution}x{self.resolution}, "
            f"dt={self.dt:.4g} dx={self.dx:.4g} nu={self.nu:.3e} periodic={self.periodic}"
        )

    def __len__(self) -> int:
        return len(self.velocity)

    def __getitem__(self, idx: int) -> Batch:
        window = (self.velocity[idx] - self.vel_mean) / self.vel_std  # (3, 2, H, W)
        return {
            "prev": torch.from_numpy(window[0]),
            "center": torch.from_numpy(window[1]),
            "next": torch.from_numpy(window[2]),
        }

    def get_raw(self, idx: int) -> np.ndarray:
        """Return the un-normalised (3, 2, H, W) window, for visualisation/diagnostics."""
        return self.velocity[idx]

    def denormalize(self, x: Tensor) -> Tensor:
        """Invert the z-score normalisation (for plotting/metrics in physical units)."""
        return x * self.vel_std + self.vel_mean


def build_dataloaders(
    data_dir: str | Path,
    source: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build DataLoaders for all available splits of one dataset source.

    Normalisation statistics are computed on the training set and shared
    with val/test for consistency.

    Args:
        data_dir: Directory containing `<source>_<split>.npz` files.
        source: 'pdebench' | 'the_well' | 'synthetic'.
        batch_size: Batch size for the training DataLoader (val/test use
            min(batch_size, split size)).
        num_workers: DataLoader worker processes (0 is the safe default
            on Windows, matching the reference repo's convention).

    Returns:
        Dict mapping split name -> DataLoader. Also attaches the
        underlying NS2DDataset to each loader as `.dataset` (standard
        torch behaviour) so callers can read `.dt/.dx/.nu/.periodic` and
        call `.denormalize`.
    """
    data_dir = Path(data_dir)
    loaders: dict[str, DataLoader] = {}

    train_path = data_dir / f"{source}_train.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run `python scripts/prepare_data.py` first "
            f"(or `python run.py --stage data`)."
        )
    train_ds = NS2DDataset(train_path)
    stats = {"vel_mean": train_ds.vel_mean, "vel_std": train_ds.vel_std}

    loaders["train"] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    for split in ("val", "test"):
        path = data_dir / f"{source}_{split}.npz"
        if not path.exists():
            logger.debug(f"Split '{split}' not found at {path}, skipping.")
            continue
        ds = NS2DDataset(path, **stats)
        bs = min(batch_size, len(ds))
        loaders[split] = DataLoader(
            ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        logger.info(f"Loaded split '{split}': {len(ds)} windows")

    return loaders
