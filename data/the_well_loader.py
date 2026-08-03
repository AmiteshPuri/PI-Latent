"""Loader for Polymathic AI's "The Well" dataset collection.

Uses the documented `WellDataset` API (pip install the_well; see
https://github.com/PolymathicAI/the_well):

    from the_well.data import WellDataset
    ds = WellDataset(well_base_path=..., well_dataset_name=...,
                      well_split_name=..., n_steps_input=2, n_steps_output=1)

Each item is a dict with keys `input_fields`, `output_fields`,
`constant_scalars`, `boundary_conditions`, `space_grid`,
`input_time_grid`, `output_time_grid`. `n_steps_input=2, n_steps_output=1`
is requested specifically so the 2 input steps + 1 output step are 3
consecutive, equally-spaced timesteps, used directly as this pipeline's
(prev, center, next) window.

Recommended dataset: `shear_flow` -- explicitly a 2D-periodic
incompressible Navier-Stokes flow (see The Well paper, Ohana et al. 2024,
appendix C.12), which is why `periodic=True` is returned unconditionally
below. `turbulent_radiative_layer_2D` is the smallest Well dataset and a
faster choice if you just want to exercise the pipeline.

IMPORTANT / stated uncertainty: the exact channel ordering within
`input_fields` (i.e. which channel index is velocity_x vs velocity_z vs
pressure vs tracer) is dataset-specific metadata this environment cannot
verify without downloading real Well data (no network access to
huggingface.co / the_well's hosting from this sandbox). Rather than
hard-code a guessed ordering, `velocity_channels` is a config field
(default [0, 1], the common convention of listing velocity first) that
you should confirm with `inspect_dataset(...)` against your chosen
dataset variant and override in configs/data_the_well.yaml if needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def inspect_dataset(base_path: str, dataset_name: str, split: str = "train") -> dict:
    """Print one sample's field layout (shapes, dtypes) for a Well dataset.

    Run this once for your chosen `dataset_name` and set
    `velocity_channels` in configs/data_the_well.yaml if the default
    [0, 1] is not correct for that dataset's field ordering.
    """
    from the_well.data import WellDataset

    ds = WellDataset(
        well_base_path=base_path,
        well_dataset_name=dataset_name,
        well_split_name=split,
        n_steps_input=2,
        n_steps_output=1,
    )
    sample = ds[0]
    info = {k: (getattr(v, "shape", type(v)) if hasattr(v, "shape") else v) for k, v in sample.items()}
    for k, v in info.items():
        logger.info(f"  {k}: {v}")
    return info


def prepare_the_well_split(
    split: str,
    n_windows: int,
    resolution: int,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Extract (prev, center, next) velocity windows from a Well dataset.

    Args:
        split: 'train' | 'val' | 'test'.
        n_windows: Number of 3-frame windows requested.
        resolution: Target spatial resolution (resized via average
            pooling if the native grid is larger).
        seed: RNG seed for sample selection.
        cfg: Source config subtree. Required: `well_base_path`,
            `dataset_name` (e.g. 'shear_flow'). Optional:
            `velocity_channels` (default [0, 1]), `dt`, `dx`.

    Returns:
        Dict with keys 'velocity' (N, 3, 2, H, W) float32, 'dt', 'dx',
        'nu' (NaN -- Well datasets report viscosity via `constant_scalars`
        per-trajectory rather than a single dataset-wide value; the
        physics loss falls back to the config default, logged clearly),
        'periodic' (True; see module docstring).
    """
    try:
        from the_well.data import WellDataset
    except ImportError as exc:
        raise ImportError(
            "The Well loader requires the `the_well` package: pip install the_well. "
            "Use `source: synthetic` for a no-download fallback."
        ) from exc

    base_path = cfg.get("well_base_path")
    dataset_name = cfg.get("dataset_name", "shear_flow")
    if not base_path:
        raise ValueError(
            "configs/data_the_well.yaml requires `well_base_path`. Download with "
            f"`the-well-download --base-path <path> --dataset {dataset_name} --split {split}` "
            "or point well_base_path at an hf:// URL to stream (see The Well README)."
        )

    well_split = {"train": "train", "val": "valid", "test": "test"}.get(split, split)
    ds = WellDataset(
        well_base_path=base_path,
        well_dataset_name=dataset_name,
        well_split_name=well_split,
        n_steps_input=2,
        n_steps_output=1,
    )
    if len(ds) == 0:
        raise ValueError(f"WellDataset('{dataset_name}', split='{well_split}') has 0 samples.")

    vel_channels = cfg.get("velocity_channels", [0, 1])
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(ds), size=n_windows)

    windows = np.empty((n_windows, 3, 2, resolution, resolution), dtype=np.float32)
    dx_est, dt_est = None, None

    for i, sample_idx in enumerate(sample_indices):
        item = ds[int(sample_idx)]
        input_fields = np.asarray(item["input_fields"])  # expected (n_steps_input, H, W, C) or (n_steps_input, C, H, W)
        output_fields = np.asarray(item["output_fields"])  # (n_steps_output, H, W, C) or channel-first equivalent

        frames = _stack_and_channel_last(input_fields, output_fields)  # -> (3, H, W, C)
        for k in range(3):
            field = frames[k][..., vel_channels]  # (H, W, 2)
            field = _resize_hw(field, resolution)
            windows[i, k, 0] = field[..., 0]
            windows[i, k, 1] = field[..., 1]

        if dx_est is None and "space_grid" in item:
            dx_est = _estimate_spacing(np.asarray(item["space_grid"]))
        if dt_est is None and "input_time_grid" in item and "output_time_grid" in item:
            dt_est = _estimate_dt(np.asarray(item["input_time_grid"]), np.asarray(item["output_time_grid"]))

    dx = float(cfg.get("dx") or dx_est or 1.0 / resolution)
    dt = float(cfg.get("dt") or dt_est or 1.0)
    nu = cfg.get("nu")
    if nu is None:
        nu = float("nan")
        logger.warning(
            "No `nu` set in configs/data_the_well.yaml and it could not be read from "
            "`constant_scalars` generically; the physics loss will use its own config "
            "default. Set `nu` explicitly for a dataset-accurate residual."
        )

    return {
        "velocity": windows,
        "dt": dt,
        "dx": dx,
        "nu": float(nu),
        "periodic": True,
    }


def _stack_and_channel_last(input_fields: np.ndarray, output_fields: np.ndarray) -> np.ndarray:
    """Concatenate input+output timesteps into (3, H, W, C), channel-last."""
    combined = np.concatenate([input_fields, output_fields], axis=0)  # (3, ...)
    if combined.ndim == 4 and combined.shape[1] < combined.shape[-1]:
        # Looks channel-first (3, C, H, W) since C is much smaller than H/W typically -- transpose.
        combined = np.transpose(combined, (0, 2, 3, 1))
    return combined


def _resize_hw(field: np.ndarray, resolution: int) -> np.ndarray:
    """Average-pool a (H, W, C) field down to (resolution, resolution, C); no-op if already there."""
    h, w = field.shape[0], field.shape[1]
    if h == resolution and w == resolution:
        return field
    if h % resolution != 0 or w % resolution != 0:
        # Fall back to nearest-index subsampling when the native grid does not
        # divide evenly (e.g. a non-square crop); documented, not silently wrong.
        ys = np.linspace(0, h - 1, resolution).astype(int)
        xs = np.linspace(0, w - 1, resolution).astype(int)
        return field[np.ix_(ys, xs)]
    fh, fw = h // resolution, w // resolution
    return field.reshape(resolution, fh, resolution, fw, field.shape[-1]).mean(axis=(1, 3))


def _estimate_spacing(space_grid: np.ndarray) -> float | None:
    """Estimate uniform grid spacing from a stored coordinate grid, if possible."""
    try:
        flat = space_grid.reshape(-1)
        diffs = np.diff(np.unique(flat))
        return float(np.median(diffs)) if len(diffs) else None
    except Exception:  # noqa: BLE001 -- best-effort
        return None


def _estimate_dt(input_time_grid: np.ndarray, output_time_grid: np.ndarray) -> float | None:
    """Estimate timestep spacing from stored time grids, if possible."""
    try:
        all_t = np.concatenate([input_time_grid.reshape(-1), output_time_grid.reshape(-1)])
        diffs = np.diff(np.sort(np.unique(all_t)))
        return float(np.median(diffs)) if len(diffs) else None
    except Exception:  # noqa: BLE001 -- best-effort
        return None
