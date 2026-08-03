"""Loader for PDEBench's 2D incompressible Navier-Stokes HDF5 files.

PDEBench stores all PDE families in HDF5 with the documented array
convention [b, t, x1, ..., xd, v] (batch, time, spatial dims, channel) --
see https://github.com/pdebench/PDEBench and the DaRUS dataset record
(doi:10.18419/darus-2986). The incompressible Navier-Stokes subset is
generated with Dirichlet BC (velocity = 0 at the domain edges) via a
differentiable solver (Holl et al., PhiFlow), which matters here because
it is NOT periodic -- see physics/derivatives.py's backend note.

IMPORTANT / stated uncertainty: PDEBench's exact per-file HDF5 key
naming has varied across dataset families and file versions (Darcy flow,
for instance, uses 'nu'/'tensor' rather than a generic velocity key), and
this environment has no network access to PDEBench's host (darus.uni-
stuttgart.de) to inspect an actual incompressible-NS file directly. Hard-
coding a guessed key name would silently produce wrong data if the guess
is stale or wrong for your file version. Instead, this loader:
  1. Uses `cfg.velocity_key` / `cfg.x_key` / `cfg.y_key` if you set them
     (recommended -- run `inspect_file(path)` once to see your file's
     actual layout and set these in configs/data_pdebench.yaml).
  2. Otherwise tries a list of historically-common candidate names.
  3. Raises an error naming every top-level key actually found in the
     file if neither works, rather than guessing silently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_CANDIDATE_VELOCITY_KEYS = ["velocity", "Velocity", "vel", "u"]
_CANDIDATE_VX_KEYS = ["Vx", "vx", "velocity_x", "u"]
_CANDIDATE_VY_KEYS = ["Vy", "vy", "velocity_y", "v"]


def inspect_file(path: str | Path) -> list[str]:
    """Print and return every dataset path inside an HDF5 file.

    Run this once against your downloaded file and set
    configs/data_pdebench.yaml `velocity_key` (or `x_key`/`y_key`)
    accordingly if auto-detection fails.
    """
    import h5py

    paths: list[str] = []

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(f"{name}  shape={obj.shape}  dtype={obj.dtype}")

    with h5py.File(str(path), "r") as f:
        f.visititems(_visit)
    for p in paths:
        logger.info(f"  {p}")
    return paths


def _load_velocity_array(h5path: Path, cfg: dict[str, Any]) -> np.ndarray:
    """Return a (B, T, H, W, 2) velocity array from a PDEBench HDF5 file."""
    import h5py

    with h5py.File(str(h5path), "r") as f:
        top_keys = list(f.keys())

        velocity_key = cfg.get("velocity_key")
        if velocity_key is not None:
            arr = np.asarray(f[velocity_key])
            return _ensure_two_channel_last(arr)

        for key in _CANDIDATE_VELOCITY_KEYS:
            if key in f:
                arr = np.asarray(f[key])
                return _ensure_two_channel_last(arr)

        x_key = cfg.get("x_key")
        y_key = cfg.get("y_key")
        if x_key and y_key and x_key in f and y_key in f:
            return np.stack([np.asarray(f[x_key]), np.asarray(f[y_key])], axis=-1)

        for xk, yk in zip(_CANDIDATE_VX_KEYS, _CANDIDATE_VY_KEYS):
            if xk in f and yk in f and xk != yk:
                return np.stack([np.asarray(f[xk]), np.asarray(f[yk])], axis=-1)

        raise KeyError(
            f"Could not find a velocity field in {h5path}. Top-level keys found: "
            f"{top_keys}. Set `velocity_key` (single [..., 2]-channel array) or "
            f"`x_key`/`y_key` (two separate scalar arrays) in "
            f"configs/data_pdebench.yaml -- run "
            f"`data.pdebench_loader.inspect_file('{h5path}')` to see full dataset "
            f"paths and shapes."
        )


def _ensure_two_channel_last(arr: np.ndarray) -> np.ndarray:
    """Normalise a PDEBench [..., v] array to exactly the 2 velocity channels, last axis."""
    if arr.shape[-1] == 2:
        return arr
    if arr.shape[-1] > 2:
        logger.warning(
            f"Velocity array has {arr.shape[-1]} channels; taking the first 2 as (Vx, Vy)."
        )
        return arr[..., :2]
    raise ValueError(f"Velocity array's last axis has only {arr.shape[-1]} channel(s), need 2.")


def _read_scalar_attr(h5path: Path, names: list[str], default: float) -> float:
    """Try to read a scalar physical parameter (e.g. viscosity) from HDF5 attrs."""
    import h5py

    try:
        with h5py.File(str(h5path), "r") as f:
            for name in names:
                if name in f.attrs:
                    return float(f.attrs[name])
                if name in f:
                    val = np.asarray(f[name])
                    if val.size == 1:
                        return float(val.reshape(-1)[0])
    except Exception as exc:  # noqa: BLE001 -- best-effort metadata read
        logger.debug(f"Could not read attribute from {h5path}: {exc}")
    return default


def _resize_spatial(arr: np.ndarray, resolution: int) -> np.ndarray:
    """Downsample the two spatial axes (H, W) to `resolution` via average pooling.

    Args:
        arr: (..., H, W, C) array.
        resolution: Target H = W.

    Returns:
        (..., resolution, resolution, C) array. No-op if already at target.
    """
    h, w = arr.shape[-3], arr.shape[-2]
    if h == resolution and w == resolution:
        return arr
    if h % resolution != 0 or w % resolution != 0:
        raise ValueError(
            f"Native resolution ({h}x{w}) is not an integer multiple of the "
            f"requested resolution ({resolution}); set `resolution` in "
            f"configs/data_pdebench.yaml to a divisor of the native size."
        )
    fh, fw = h // resolution, w // resolution
    leading_shape = arr.shape[:-3]
    reshaped = arr.reshape(*leading_shape, resolution, fh, resolution, fw, arr.shape[-1])
    return reshaped.mean(axis=(-4, -2))


def prepare_pdebench_split(
    split: str,
    n_windows: int,
    resolution: int,
    seed: int,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Extract (prev, center, next) velocity windows from a PDEBench HDF5 file.

    Args:
        split: 'train' | 'val' | 'test'.
        n_windows: Number of 3-frame windows requested.
        resolution: Target spatial resolution (must divide the native size).
        seed: RNG seed for window sampling.
        cfg: Source config subtree. Required: `file_path` (path to the
            downloaded .h5 file for this split, or one file covering all
            splits with `split_fraction` used to slice trajectories).
            Optional: `velocity_key`, `x_key`, `y_key`, `dx`, `dt`, `nu`,
            `window_stride`, `nu_attr_names`.

    Returns:
        Dict with keys 'velocity' (N, 3, 2, H, W) float32, 'dt', 'dx',
        'nu', 'periodic' (always False -- PDEBench incompressible NS uses
        Dirichlet BC).
    """
    file_path = cfg.get("file_path")
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(
            f"configs/data_pdebench.yaml `file_path` ({file_path}) does not exist. "
            f"Download the PDEBench incompressible Navier-Stokes HDF5 file from "
            f"https://darus.uni-stuttgart.de (dataset doi:10.18419/darus-2986) and "
            f"point `file_path` at it, or use `source: synthetic` for a no-download "
            f"fallback."
        )
    h5path = Path(file_path)

    velocity = _load_velocity_array(h5path, cfg)  # (B, T, H, W, 2), per PDEBench convention
    if velocity.ndim != 5:
        raise ValueError(
            f"Expected a 5D [b,t,x,y,v] array per the PDEBench convention, got shape "
            f"{velocity.shape}. Inspect the file with "
            f"data.pdebench_loader.inspect_file('{h5path}') and adjust the loader "
            f"if this file uses a different layout."
        )

    velocity = _resize_spatial(velocity, resolution)

    stride = int(cfg.get("window_stride", 1))
    n_traj, n_steps = velocity.shape[0], velocity.shape[1]
    if n_steps < 2 * stride + 1:
        raise ValueError(
            f"Trajectory length ({n_steps}) is too short for window_stride={stride}; "
            f"need at least {2 * stride + 1} timesteps."
        )

    # Deterministic, non-overlapping train/val/test split by trajectory index.
    split_fracs = cfg.get("split_fractions", {"train": 0.8, "val": 0.1, "test": 0.1})
    traj_idx = np.arange(n_traj)
    rng = np.random.default_rng(0)  # fixed seed for the split assignment itself, independent of sampling seed
    rng.shuffle(traj_idx)
    n_train = int(n_traj * split_fracs.get("train", 0.8))
    n_val = int(n_traj * split_fracs.get("val", 0.1))
    split_slices = {
        "train": traj_idx[:n_train],
        "val": traj_idx[n_train : n_train + n_val],
        "test": traj_idx[n_train + n_val :],
    }
    split_traj = split_slices.get(split, traj_idx)
    if len(split_traj) == 0:
        raise ValueError(f"Split '{split}' has zero trajectories after the train/val/test split.")

    sample_rng = np.random.default_rng(seed)
    windows = np.empty((n_windows, 3, 2, resolution, resolution), dtype=np.float32)
    for i in range(n_windows):
        traj = sample_rng.choice(split_traj)
        center_t = sample_rng.integers(stride, n_steps - stride)
        for k, t in enumerate((center_t - stride, center_t, center_t + stride)):
            frame = velocity[traj, t]  # (H, W, 2)
            windows[i, k, 0] = frame[..., 0]
            windows[i, k, 1] = frame[..., 1]

    dx = float(cfg.get("dx", 1.0 / resolution))
    dt = float(cfg.get("dt", 1.0)) * stride
    nu = cfg.get("nu")
    if nu is None:
        nu = _read_scalar_attr(h5path, cfg.get("nu_attr_names", ["nu", "viscosity", "eta"]), default=1e-3)
        logger.info(f"Using viscosity nu={nu} (config override not set; read from file attrs or default).")

    return {
        "velocity": windows,
        "dt": dt,
        "dx": dx,
        "nu": float(nu),
        "periodic": False,  # PDEBench incompressible NS uses Dirichlet BC
    }
