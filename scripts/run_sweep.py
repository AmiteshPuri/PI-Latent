"""CLI: run a grid sweep defined in configs/sweeps/*.yaml.

    python scripts/run_sweep.py --sweep configs/sweeps/vqvae_sweep.yaml --stage vqvae
    python scripts/run_sweep.py --sweep configs/sweeps/flow_sweep.yaml --stage flow
    python scripts/run_sweep.py --sweep configs/sweeps/vqvae_sweep.yaml --stage vqvae --dry_run

Each grid combination runs as an isolated subprocess -- a clean CUDA
context per run avoids memory fragmentation across back-to-back runs on
a small GPU, which matters more here than the extra process-start
overhead. Interrupting a sweep and re-running the same command picks up
exactly where it left off, at two levels: already-completed combinations
are skipped (fast no-op, see train_vqvae.py/train_flow_matching.py's own
"already completed" check) and a partially-completed combination resumes
from its own latest.pt rather than restarting.

Grid keys are dot-paths into the target script's config, with two
special-cased keys (`data_config`, `vqvae_config`) that select WHICH
config YAML to load rather than overriding a value inside one -- see
configs/sweeps/vqvae_sweep.yaml.
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)

SPECIAL_AXES = {"data_config", "vqvae_config"}


def _slug(value) -> str:
    """Turn a grid value into a short, filesystem/CLI-safe run-name fragment."""
    s = str(value)
    if "/" in s or "\\" in s:
        # Path-like value (e.g. a checkpoint path) -- use its parent directory name
        # (normally the identifying run name) instead of slugifying the whole path.
        p = Path(s)
        return p.parent.name or p.stem
    return s.replace(".", "p").replace(" ", "")


def _combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    value_lists = [list(grid[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _run_name_for(combo: dict) -> str:
    parts = ["sweep"]
    for key, value in combo.items():
        short_key = key.split(".")[-1]
        parts.append(f"{short_key}{_slug(value)}")
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grid sweep over train_vqvae.py or train_flow_matching.py.")
    parser.add_argument("--sweep", required=True, help="Path to a configs/sweeps/*.yaml file.")
    parser.add_argument("--stage", choices=["vqvae", "flow"], required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    args = parser.parse_args()

    sweep_cfg = load_config(args.sweep)
    grid = dict(sweep_cfg.grid)
    combinations = _combinations(grid)
    logger.info(f"Sweep '{args.sweep}' ({args.stage}): {len(combinations)} combinations.")

    script = "scripts/train_vqvae.py" if args.stage == "vqvae" else "scripts/train_flow_matching.py"
    n_failed = 0

    for i, combo in enumerate(combinations):
        run_name = _run_name_for(combo)
        cmd = [sys.executable, script, "--run_name", run_name]

        if args.stage == "vqvae":
            cmd += ["--data_config", str(combo.get("data_config", sweep_cfg.get("base_data", "data_synthetic")))]
            cmd += ["--vqvae_config", str(combo.get("vqvae_config", sweep_cfg.get("base_vqvae", "vqvae_baseline")))]
        else:
            cmd += ["--data_config", str(combo.get("data_config", "data_synthetic"))]

        override_pairs = [f"{k}={v}" for k, v in combo.items() if k not in SPECIAL_AXES]
        if override_pairs:
            cmd += ["--override", *override_pairs]

        logger.info(f"[{i + 1}/{len(combinations)}] {' '.join(cmd)}")
        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode != 0:
            n_failed += 1
            logger.error(f"Combination {combo} failed (exit code {result.returncode}); continuing with the next one.")

    if not args.dry_run:
        logger.info(f"Sweep finished: {len(combinations) - n_failed}/{len(combinations)} combinations succeeded.")


if __name__ == "__main__":
    main()
