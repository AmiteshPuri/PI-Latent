"""Master pipeline orchestrator.

    python run.py --stage data --data_config data_synthetic
    python run.py --stage train_vqvae --data_config data_synthetic --vqvae_config vqvae_baseline
    python run.py --stage train_vqvae --data_config data_synthetic --vqvae_config vqvae_physics
    python run.py --stage train_flow --data_config data_synthetic
    python run.py --stage evaluate --data_config data_synthetic
    python run.py --stage all --data_config data_synthetic

Every stage delegates to a thin script in scripts/ and is resumable --
interrupting `--stage all` and re-running the same command continues
from wherever it stopped, both across stages (data generation and
already-completed trainings are skipped) and within a stage (partially
trained runs resume from their last checkpoint). See scripts/train_vqvae.py
and scripts/train_flow_matching.py for the resume mechanics.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="NS2D physics-informed latent generative modelling -- master pipeline.")
    parser.add_argument("--stage", choices=["data", "train_vqvae", "train_flow", "evaluate", "all"], default="all")
    parser.add_argument("--data_config", default="data_synthetic")
    parser.add_argument("--vqvae_config", default="vqvae_baseline", help="Only used for --stage train_vqvae.")
    parser.add_argument("--override", nargs="*", default=[], help="Forwarded as --override to the underlying script.")
    args = parser.parse_args()

    py = sys.executable
    override_args = ["--override", *args.override] if args.override else []
    data_tag = args.data_config.replace("data_", "")

    if args.stage == "data":
        run([py, "scripts/prepare_data.py", "--data_config", args.data_config])

    elif args.stage == "train_vqvae":
        run([
            py, "scripts/train_vqvae.py",
            "--data_config", args.data_config, "--vqvae_config", args.vqvae_config,
            *override_args,
        ])

    elif args.stage == "train_flow":
        run([py, "scripts/train_flow_matching.py", "--data_config", args.data_config, *override_args])

    elif args.stage == "evaluate":
        run([py, "scripts/evaluate.py", "--data_config", args.data_config])

    elif args.stage == "all":
        run([py, "scripts/prepare_data.py", "--data_config", args.data_config])
        run([py, "scripts/train_vqvae.py", "--data_config", args.data_config, "--vqvae_config", "vqvae_baseline"])
        run([py, "scripts/train_vqvae.py", "--data_config", args.data_config, "--vqvae_config", "vqvae_physics"])
        # Explicit vqvae_checkpoint/vqvae_config override rather than relying on
        # configs/flow_matching.yaml's static default, so `--stage all` is correct
        # for whichever --data_config was passed, not just the synthetic default.
        vqvae_checkpoint = f"outputs/checkpoints/ns2d_vqvae_{data_tag}_physics/best.pt"
        run([
            py, "scripts/train_flow_matching.py", "--data_config", args.data_config,
            "--override", f"vqvae_checkpoint={vqvae_checkpoint}", "vqvae_config=vqvae_physics",
        ])
        run([py, "scripts/evaluate.py", "--data_config", args.data_config])
        print("\nAll stages complete. Open notebooks/final_analysis.ipynb for the full figure set.")


if __name__ == "__main__":
    main()
