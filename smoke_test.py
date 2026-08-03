"""Fast end-to-end pipeline verification. Runs entirely on CPU with a
tiny grid and tiny models -- no download, no GPU required, seconds not
minutes. Run this before any real training to catch integration bugs
early (shape mismatches, NaNs, checkpoint round-trip failures) rather
than after burning GPU time.

    python smoke_test.py
    python smoke_test.py --keep    # keep outputs/smoke_* for inspection
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.datamodule import build_dataloaders  # noqa: E402
from data.generate_dataset import generate_dataset  # noqa: E402
from evaluation import latent_metrics  # noqa: E402
from evaluation.evaluate_generation import evaluate_generation  # noqa: E402
from models.flow_matching.model import LatentFlowMatcher  # noqa: E402
from models.vqvae.model import TransformerVQVAE  # noqa: E402
from physics.derivatives import curl_2d, divergence  # noqa: E402
from training.callbacks import CheckpointCallback, CSVLoggerCallback  # noqa: E402
from training.checkpointing import load_checkpoint, save_checkpoint  # noqa: E402
from training.losses import FlowMatchingLoss, VQVAELossComputer  # noqa: E402
from training.trainer_flow import FlowMatchingTrainer  # noqa: E402
from training.trainer_vqvae import VQVAETrainer  # noqa: E402
from utils.visualization import (  # noqa: E402
    compute_and_plot_tsne,
    plot_distribution_comparison,
    plot_energy_spectra,
    plot_training_curves,
    plot_validation_fields,
    plot_vorticity_comparison,
)

SMOKE_ROOT = Path("outputs/smoke_test_tmp")
RESOLUTION = 16
PATCH_SIZE = 4


class _Result:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.failures.append(f"{name}: {detail}")
            print(f"  [FAIL] {name}: {detail}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{self.passed}/{total} checks passed.")
        if self.failures:
            print("Failures:")
            for f in self.failures:
                print(f"  - {f}")
        return self.failed == 0


def _run(fn, name: str, result: _Result) -> None:
    try:
        fn()
        result.record(name, True)
    except Exception as exc:  # noqa: BLE001 -- smoke test must catch and report, not crash
        result.record(name, False, f"{type(exc).__name__}: {exc}")


def check_physics_operators(result: _Result) -> None:
    """Divergence/curl of an analytical Taylor-Green vortex against closed-form values."""
    n = 32
    dx = 2 * torch.pi / n
    xs = torch.arange(n) * dx
    x, y = torch.meshgrid(xs, xs, indexing="xy")
    u = torch.cos(x) * torch.sin(y)
    v = -torch.sin(x) * torch.cos(y)

    def _div_check():
        div = divergence(u.unsqueeze(0), v.unsqueeze(0), dx, "spectral")
        assert div.abs().max().item() < 1e-4, f"max |div|={div.abs().max().item()}"

    def _curl_check():
        omega = curl_2d(u.unsqueeze(0), v.unsqueeze(0), dx, "spectral")
        expected = -2 * torch.cos(x) * torch.cos(y)
        err = (omega[0] - expected).abs().max().item()
        assert err < 1e-3, f"max |error|={err}"

    _run(_div_check, "physics: divergence-free field has ~0 divergence", result)
    _run(_curl_check, "physics: curl matches analytical Taylor-Green vorticity", result)


def check_data_generation(result: _Result) -> dict:
    ctx = {}

    def _generate():
        paths = generate_dataset(
            output_dir=SMOKE_ROOT / "data",
            source="synthetic",
            source_cfg={"dt": 0.02, "nu": 1e-2, "window_stride": 2},
            n_windows={"train": 12, "val": 6, "test": 6},
            resolution=RESOLUTION,
            seed=0,
        )
        for split, path in paths.items():
            assert path.exists(), f"{split} split not written to {path}"
        ctx["paths"] = paths

    _run(_generate, "data: synthetic split generation", result)
    return ctx


def check_dataloaders(result: _Result) -> dict:
    ctx = {}

    def _build():
        loaders = build_dataloaders(SMOKE_ROOT / "data", "synthetic", batch_size=4, num_workers=0)
        assert "train" in loaders and "val" in loaders
        batch = next(iter(loaders["train"]))
        assert batch["center"].shape == (4, 2, RESOLUTION, RESOLUTION), batch["center"].shape
        assert batch["prev"].shape == batch["center"].shape
        ctx["loaders"] = loaders

    _run(_build, "data: DataLoader construction and batch shapes", result)
    return ctx


def check_vqvae_model(result: _Result) -> dict:
    ctx = {}

    def _build_and_forward():
        model = TransformerVQVAE(
            in_channels=2, resolution=RESOLUTION, patch_size=PATCH_SIZE,
            embed_dim=32, encoder_depth=2, decoder_depth=2, n_heads=2,
            num_codes=32, reset_after_n_batches=5,
        )
        x = torch.randn(3, 2, RESOLUTION, RESOLUTION)
        out = model(x)
        assert out["reconstruction"].shape == x.shape, out["reconstruction"].shape
        assert out["indices"].shape == (3, model.n_tokens)
        assert torch.isfinite(out["reconstruction"]).all()
        assert torch.isfinite(out["commitment_loss"])
        ctx["model"] = model

    _run(_build_and_forward, "model: TransformerVQVAE forward pass", result)
    return ctx


def check_vqvae_losses(result: _Result, model: torch.nn.Module, loaders: dict) -> None:
    def _baseline_no_nan():
        loss_computer = VQVAELossComputer(physics_weight=0.0)
        batch = next(iter(loaders["train"]))
        out = model(batch["center"])
        losses = loss_computer(out, batch, loaders["train"].dataset)
        for key in ("reconstruction_loss", "vq_loss", "physics_loss", "total_loss"):
            assert torch.isfinite(losses[key]), f"{key} is not finite: {losses[key]}"

    def _physics_no_nan():
        loss_computer = VQVAELossComputer(physics_weight=0.1)
        batch = next(iter(loaders["train"]))
        out = model(batch["center"])
        losses = loss_computer(out, batch, loaders["train"].dataset)
        for key in ("reconstruction_loss", "vq_loss", "physics_loss", "total_loss"):
            assert torch.isfinite(losses[key]), f"{key} is not finite: {losses[key]}"
        losses["total_loss"].backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "non-finite gradient after physics-weighted backward"

    _run(_baseline_no_nan, "losses: baseline (physics_weight=0) losses finite", result)
    _run(_physics_no_nan, "losses: physics-informed losses finite, gradients finite", result)


def check_one_training_epoch(result: _Result, loaders: dict) -> dict:
    ctx = {}

    def _fit_one_epoch():
        model = TransformerVQVAE(
            in_channels=2, resolution=RESOLUTION, patch_size=PATCH_SIZE,
            embed_dim=32, encoder_depth=2, decoder_depth=2, n_heads=2,
            num_codes=32, reset_after_n_batches=5,
        )
        checkpoint_dir = SMOKE_ROOT / "checkpoints"
        callbacks = [
            CSVLoggerCallback(SMOKE_ROOT / "metrics" / "smoke_vqvae.csv"),
            CheckpointCallback(checkpoint_dir, "smoke_vqvae", checkpoint_epochs=[1]),
        ]
        trainer = VQVAETrainer(
            model=model, train_loader=loaders["train"], val_loader=loaders["val"],
            physics_weight=0.1, lr=1e-3, weight_decay=0.0, epochs=1, grad_clip=1.0,
            use_amp=False, device=torch.device("cpu"), callbacks=callbacks,
        )
        trainer.fit()
        assert (checkpoint_dir / "smoke_vqvae" / "latest.pt").exists()
        assert (checkpoint_dir / "smoke_vqvae" / "best.pt").exists()
        ctx["model"] = model
        ctx["trainer"] = trainer

    _run(_fit_one_epoch, "training: one VQ-VAE epoch end-to-end (train+val+checkpoint)", result)
    return ctx


def check_resume(result: _Result, loaders: dict) -> None:
    def _resume():
        model = TransformerVQVAE(
            in_channels=2, resolution=RESOLUTION, patch_size=PATCH_SIZE,
            embed_dim=32, encoder_depth=2, decoder_depth=2, n_heads=2,
            num_codes=32, reset_after_n_batches=5,
        )
        trainer = VQVAETrainer(
            model=model, train_loader=loaders["train"], val_loader=loaders["val"],
            physics_weight=0.1, lr=1e-3, weight_decay=0.0, epochs=2, grad_clip=1.0,
            use_amp=False, device=torch.device("cpu"), callbacks=[],
        )
        ckpt_path = SMOKE_ROOT / "checkpoints" / "smoke_vqvae" / "latest.pt"
        trainer.resume_from_checkpoint(ckpt_path)
        assert trainer.start_epoch == 2, f"expected start_epoch=2, got {trainer.start_epoch}"

    _run(_resume, "training: resume_from_checkpoint sets correct start_epoch", result)


def check_checkpoint_roundtrip(result: _Result) -> None:
    def _roundtrip():
        model = TransformerVQVAE(
            in_channels=2, resolution=RESOLUTION, patch_size=PATCH_SIZE,
            embed_dim=16, encoder_depth=1, decoder_depth=1, n_heads=2, num_codes=16,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        path = SMOKE_ROOT / "checkpoints" / "roundtrip.pt"
        save_checkpoint(model, opt, sched, step=3, val_loss=0.5, path=path)

        model2 = TransformerVQVAE(
            in_channels=2, resolution=RESOLUTION, patch_size=PATCH_SIZE,
            embed_dim=16, encoder_depth=1, decoder_depth=1, n_heads=2, num_codes=16,
        )
        ckpt = load_checkpoint(path, model2)
        assert ckpt["step"] == 3
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "loaded weights do not match saved weights"

    _run(_roundtrip, "checkpointing: save/load round-trip preserves weights", result)


def check_codebook_health_metrics(result: _Result) -> None:
    def _collapsed():
        indices = np.zeros(1000, dtype=np.int64)
        health = latent_metrics.compute_codebook_health(indices, num_codes=64)
        assert abs(health["codebook_perplexity"] - 1.0) < 1e-6, health
        assert abs(health["codebook_utilization"] - 1 / 64) < 1e-6, health

    def _uniform():
        rng = np.random.default_rng(0)
        indices = rng.integers(0, 64, size=100_000)
        health = latent_metrics.compute_codebook_health(indices, num_codes=64)
        assert health["codebook_perplexity"] > 60, health  # close to 64 for near-uniform usage
        assert health["codebook_utilization"] > 0.95, health

    _run(_collapsed, "metrics: codebook health detects a fully-collapsed codebook", result)
    _run(_uniform, "metrics: codebook health reports near-full health for uniform usage", result)


def check_flow_matching(result: _Result, vqvae: torch.nn.Module, loaders: dict) -> dict:
    ctx = {}

    def _forward():
        flow_model = LatentFlowMatcher(grid_shape=vqvae.grid_shape, code_dim=vqvae.codebook.code_dim, embed_dim=32, depth=2, n_heads=2)
        x = torch.randn(3, vqvae.n_tokens, vqvae.codebook.code_dim)
        t = torch.rand(3)
        v = flow_model(x, t)
        assert v.shape == x.shape
        assert torch.isfinite(v).all()
        ctx["flow_model"] = flow_model

    def _loss():
        loss_fn = FlowMatchingLoss()
        x1 = torch.randn(3, vqvae.n_tokens, vqvae.codebook.code_dim)
        loss = loss_fn(ctx["flow_model"], x1)
        assert torch.isfinite(loss)
        loss.backward()

    def _sample():
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad_(False)
        samples = ctx["flow_model"].sample(batch_size=2, device=torch.device("cpu"), n_steps=3, method="euler")
        assert samples.shape == (2, vqvae.n_tokens, vqvae.codebook.code_dim)
        assert torch.isfinite(samples).all()
        heun_samples = ctx["flow_model"].sample(batch_size=2, device=torch.device("cpu"), n_steps=3, method="heun")
        assert torch.isfinite(heun_samples).all()

    def _one_epoch():
        checkpoint_dir = SMOKE_ROOT / "checkpoints"
        trainer = FlowMatchingTrainer(
            flow_model=ctx["flow_model"], vqvae=vqvae,
            train_loader=loaders["train"], val_loader=loaders["val"],
            lr=1e-3, weight_decay=0.0, epochs=1, grad_clip=1.0,
            use_amp=False, device=torch.device("cpu"),
            callbacks=[CheckpointCallback(checkpoint_dir, "smoke_flow", checkpoint_epochs=[])],
        )
        trainer.fit()
        assert (checkpoint_dir / "smoke_flow" / "latest.pt").exists()

    _run(_forward, "flow matching: model forward pass", result)
    _run(_loss, "flow matching: CFM loss finite, backward succeeds", result)
    _run(_sample, "flow matching: Euler and Heun ODE sampling produce finite output", result)
    _run(_one_epoch, "flow matching: one training epoch end-to-end", result)
    return ctx


def check_generation_evaluation(result: _Result, vqvae: torch.nn.Module, flow_model: torch.nn.Module, loaders: dict) -> dict:
    ctx = {}

    def _evaluate():
        results = evaluate_generation(
            flow_model, vqvae, loaders["val"], torch.device("cpu"),
            n_samples=6, n_ode_steps=3, ode_method="euler",
        )
        for key in ("mmd", "sliced_wasserstein_distance", "spectral_energy_error",
                    "pde_residual_norm_generated_mean", "pde_residual_norm_real_mean"):
            assert np.isfinite(results[key]), f"{key}={results[key]}"
        assert results["omega_gt"].shape == results["omega_gen"].shape
        ctx["results"] = results

    _run(_evaluate, "evaluation: evaluate_generation produces finite metrics", result)
    return ctx


def check_visualizations(result: _Result, gen_results: dict) -> None:
    def _validation_fields():
        gt = np.random.randn(2, RESOLUTION, RESOLUTION)
        recon = gt + 0.1 * np.random.randn(2, RESOLUTION, RESOLUTION)
        residual = np.random.randn(RESOLUTION, RESOLUTION)
        divergence_field = np.random.randn(RESOLUTION, RESOLUTION)
        fig = plot_validation_fields(gt, recon, residual, divergence_field)
        fig.savefig(SMOKE_ROOT / "tmp.png")
        import matplotlib.pyplot as plt
        plt.close(fig)

    def _training_curves():
        history = {"train_reconstruction_loss": [1.0, 0.5, 0.3], "val_reconstruction_loss": [1.1, 0.6, 0.35]}
        fig = plot_training_curves(history)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def _energy_spectra():
        fig = plot_energy_spectra(gen_results["k_bins"], gen_results["e_gt"], gen_results["e_gen"])
        import matplotlib.pyplot as plt
        plt.close(fig)

    def _vorticity():
        fig = plot_vorticity_comparison(gen_results["omega_gt"], gen_results["omega_gen"], n_show=2)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def _tsne():
        fig = compute_and_plot_tsne(gen_results["real_latents_pooled"], gen_results["gen_latents_pooled"])
        import matplotlib.pyplot as plt
        plt.close(fig)

    def _distribution():
        fig = plot_distribution_comparison(gen_results["residual_gt"], gen_results["residual_gen"], 0.1, 0.2)
        import matplotlib.pyplot as plt
        plt.close(fig)

    _run(_validation_fields, "viz: plot_validation_fields (Figure 2/3 panels)", result)
    _run(_training_curves, "viz: plot_training_curves (Figure 1)", result)
    _run(_energy_spectra, "viz: plot_energy_spectra (Figure 4)", result)
    _run(_vorticity, "viz: plot_vorticity_comparison (Figure 5)", result)
    _run(_tsne, "viz: compute_and_plot_tsne (Figure 6)", result)
    _run(_distribution, "viz: plot_distribution_comparison (Figure 7)", result)


def check_cli_help(result: _Result) -> None:
    scripts = [
        "run.py", "scripts/prepare_data.py", "scripts/train_vqvae.py",
        "scripts/train_flow_matching.py", "scripts/run_sweep.py", "scripts/evaluate.py",
    ]
    for script in scripts:
        def _help(script=script):
            proc = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True, timeout=30)
            assert proc.returncode == 0, f"exit code {proc.returncode}: {proc.stderr[-500:]}"

        _run(_help, f"cli: {script} --help", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="Keep outputs/smoke_test_tmp for inspection.")
    args = parser.parse_args()

    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    result = _Result()

    print("Physics operators")
    check_physics_operators(result)

    print("\nData pipeline")
    check_data_generation(result)
    loader_ctx = check_dataloaders(result)

    if "loaders" in loader_ctx:
        print("\nVQ-VAE model")
        model_ctx = check_vqvae_model(result)
        if "model" in model_ctx:
            check_vqvae_losses(result, model_ctx["model"], loader_ctx["loaders"])

        print("\nVQ-VAE training")
        epoch_ctx = check_one_training_epoch(result, loader_ctx["loaders"])
        check_resume(result, loader_ctx["loaders"])
        check_checkpoint_roundtrip(result)
        check_codebook_health_metrics(result)

        if "model" in epoch_ctx:
            print("\nFlow matching")
            flow_ctx = check_flow_matching(result, epoch_ctx["model"], loader_ctx["loaders"])

            if "flow_model" in flow_ctx:
                print("\nGeneration evaluation")
                gen_ctx = check_generation_evaluation(result, epoch_ctx["model"], flow_ctx["flow_model"], loader_ctx["loaders"])

                if "results" in gen_ctx:
                    print("\nVisualizations")
                    check_visualizations(result, gen_ctx["results"])

    print("\nCLI entry points")
    check_cli_help(result)

    if not args.keep:
        shutil.rmtree(SMOKE_ROOT, ignore_errors=True)

    ok = result.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
