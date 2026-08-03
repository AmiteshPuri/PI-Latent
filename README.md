# NS2D physics-informed latent generative modelling

**Does physics-informed latent representation learning improve generative surrogate modelling?**

Two-stage pipeline on 2D Navier-Stokes:

1. **Stage 1** -- a Transformer VQ-VAE compresses velocity-field snapshots into a discrete
   latent code. Trained in two variants that are architecturally identical: **baseline**
   (reconstruction + VQ loss) and **physics-informed** (adds a scale-normalised
   divergence-free + vorticity-transport residual penalty).
2. **Stage 2** -- a latent Flow Matching model learns to generate new codebook embeddings
   over the frozen Stage-1 codebook; the frozen decoder turns generated latents back into
   velocity fields.

Structure follows [NO-stresstesting](https://github.com/AmiteshPuri/NO-stresstesting)'s
conventions: OmegaConf configs, a name-based registry, callback-driven training, atomic
checkpoint writes, a `run.py` orchestrator, and a fast CPU `smoke_test.py`.

## Quick start

```bash
pip install -r requirements.txt
python smoke_test.py                 # ~10-20s, no GPU, no download -- verifies the whole pipeline
python run.py --stage all            # generates synthetic data, trains both Stage-1 variants,
                                      # trains Stage 2, evaluates, on the synthetic fallback source
```

`--stage all` uses `data_synthetic` (no download required) by default. For a real dataset:

```bash
python run.py --stage all --data_config data_pdebench   # after setting file_path in configs/data_pdebench.yaml
python run.py --stage all --data_config data_the_well    # after downloading with the-well-download
```

Every stage is resumable: interrupt and re-run the same command, and it continues from the
last checkpoint rather than restarting (see [Resuming](#resuming--checkpoints)).

## Repository layout

```
configs/            experiment.yaml (master) + data_*.yaml + vqvae_*.yaml + flow_matching.yaml + sweeps/
data/                dataset sources (pdebench, the_well, synthetic) -> unified NS2DDataset
models/              TransformerVQVAE (encoder/codebook/decoder), LatentFlowMatcher
physics/             differentiable derivative operators, PDE residuals, spectral energy
training/            losses, checkpointing, callbacks, the two trainers
evaluation/          reconstruction/physics/latent/distribution metrics, generation evaluation
utils/                config, logging, seeding, device, I/O, visualisation
scripts/              thin CLIs: prepare_data, train_vqvae, train_flow_matching, run_sweep, evaluate
notebooks/            final_analysis.ipynb (the 7 figures) + build_notebook.py (regenerates it)
run.py                master orchestrator (--stage data|train_vqvae|train_flow|evaluate|all)
smoke_test.py          fast end-to-end verification, tiny grid, no GPU/download
```

## Datasets

| Source | Config | Boundary condition | Notes |
|---|---|---|---|
| Synthetic (default) | `data_synthetic.yaml` | Periodic | A real pseudospectral 2D NS solver (vorticity form, Lie-split diffusion/advection). No download; used by `smoke_test.py`. Not a substitute for real data in the actual comparison. |
| PDEBench | `data_pdebench.yaml` | **Dirichlet** | Incompressible NS, generated with PhiFlow. Download from [darus.uni-stuttgart.de](https://darus.uni-stuttgart.de) (doi:10.18419/darus-2986). Field key names vary by file version -- run `data.pdebench_loader.inspect_file(path)` and set `velocity_key`/`x_key`/`y_key` if auto-detection fails. |
| The Well | `data_the_well.yaml` | Periodic | `pip install the_well`. Default dataset `shear_flow` is explicitly 2D-periodic incompressible NS (Ohana et al. 2024); `turbulent_radiative_layer_2D` is smaller/faster. Channel ordering is dataset-specific -- run `data.the_well_loader.inspect_dataset(...)` and set `velocity_channels` if the default `[0, 1]` is wrong. |

**The periodic-vs-Dirichlet distinction is not cosmetic.** The physics residual can be computed
two ways (`physics/derivatives.py`): spectral (FFT), exact for periodic domains, or
finite-difference with the boundary ring excluded from the loss/metric, needed for PDEBench's
Dirichlet BC (FFT derivatives silently assume wraparound continuity that does not hold there).
Each dataset source reports its own `periodic` flag; the rest of the pipeline picks the correct
backend automatically.

Every source converges to the same on-disk format: `<source>_<split>.npz` with `velocity`
`(N, 3, 2, H, W)` -- `[prev, center, next]` windows, since the physics loss needs a time
derivative but the VQ-VAE only reconstructs the center frame. Generation is a separate module
(`data/generate_dataset.py`) from both the solvers/loaders and the CLI, and is skip-aware
(existing splits are not regenerated).

## Stage 1: Transformer VQ-VAE

ViT-style patch tokenisation -> Transformer encoder -> EMA vector quantiser (dead-code reset)
-> Transformer decoder. No output activation (inputs are z-score normalised, so a linear head
matches the target range -- an earlier project's tanh-saturation bug is the reason this is
called out explicitly rather than left implicit).

**Baseline vs physics-informed is a single scalar, `physics_weight`** (0.0 vs >0 in
`configs/vqvae_baseline.yaml` / `vqvae_physics.yaml`); the architecture is otherwise identical,
per the project spec. The physics loss (divergence-free + vorticity-transport residual) is
scale-normalised by the RMS of the corresponding leading-order term, so it is O(1) regardless
of dataset/viscosity -- avoiding the multi-order-of-magnitude `lambda` retuning a naive raw-MSE
physics loss would need. For the baseline run, the physics loss is still computed and logged
every epoch (for a fair TensorBoard comparison against the physics-informed run) but under
`torch.no_grad()`, so it costs no autograd memory.

TensorBoard tags (`outputs/tensorboard/<run_name>/`):

```
train/reconstruction_loss   train/vq_loss   train/physics_loss   train/total_loss
val/reconstruction_loss     val/physics_loss
metrics/divergence_error    metrics/residual_norm    metrics/codebook_perplexity
metrics/codebook_utilization (bonus)    validation/fields (GT | reconstruction | error | PDE residual | divergence, logged as a figure each epoch)
```

```bash
python scripts/train_vqvae.py --data_config data_synthetic --vqvae_config vqvae_baseline
python scripts/train_vqvae.py --data_config data_synthetic --vqvae_config vqvae_physics
tensorboard --logdir outputs/tensorboard
```

## Stage 2: latent Flow Matching

Chosen over latent diffusion per the stated research interest. A Transformer vector-field
network (AdaLN-Zero time conditioning, DiT-style) trained with the standard conditional
flow-matching / rectified-flow objective, operating on the frozen VQ-VAE's codebook embeddings
(looked up from indices obtained via the frozen encoder -- not the raw discrete indices
themselves, since flow matching is a continuous-state method). Generation integrates the
learned ODE from Gaussian noise (Euler or Heun, configurable step count = NFE) and snaps the
result to the nearest codebook vector before decoding.

```bash
python scripts/train_flow_matching.py --data_config data_synthetic
python scripts/evaluate.py --data_config data_synthetic   # MMD, Wasserstein, spectral energy error, PDE residual + figures
```

`evaluation/evaluate_generation.py` documents one real modelling choice worth reading before
trusting the generated-sample PDE residual number: an unconditionally generated snapshot has no
true "next timestep", so its residual pairs the generated field with a *randomly drawn* real
(prev, next) context rather than a specific trajectory's. It tests whether generated fields have
plausible lengthscales/gradients for this dynamical system, not closed-loop rollout consistency.

## Sweeps

`scripts/run_sweep.py` runs a Cartesian grid from a YAML file (`configs/sweeps/`) as isolated
subprocesses (a clean CUDA context per run matters more on a small GPU than the process-start
overhead), and is skip/resume-aware at both the combination level and within each combination.

```bash
python scripts/run_sweep.py --sweep configs/sweeps/vqvae_sweep.yaml --stage vqvae --dry_run
python scripts/run_sweep.py --sweep configs/sweeps/vqvae_sweep.yaml --stage vqvae
python scripts/run_sweep.py --sweep configs/sweeps/flow_sweep.yaml --stage flow
```

`vqvae_sweep.yaml` grids over dataset source and `physics_weight` (and, if you extend it,
`num_codes` or anything else in `configs/vqvae_*.yaml`); `flow_sweep.yaml` grids over the
flow-matching NFE and which frozen VQ-VAE checkpoint to build on. Any `configs/vqvae_*.yaml` or
`configs/flow_matching.yaml` field is a valid grid axis via its dot-path.

## Resuming / checkpoints

Every training script auto-detects `outputs/checkpoints/<run_name>/latest.pt` and resumes from
the next epoch -- there is no separate `--resume` flag, just re-run the same command. A run that
already reached its target epoch count is a fast no-op. Checkpoint writes are atomic (write to a
temp file, then rename), so a process killed mid-save cannot leave a corrupt `latest.pt`.

```
outputs/checkpoints/<run_name>/
    latest.pt        always-current, what resume reads
    best.pt           lowest validation loss so far, what Stage 2 / evaluation load
    epoch_N.pt         extra snapshots at configs/experiment.yaml's checkpoint_epochs, for ablation
```

Because `--override` can change the architecture (a sweep varying `vqvae.num_codes`, for
example), reconstructing a frozen model from the checkpoint alone is not safe if you only reread
the static YAML. `scripts/train_flow_matching.py` and `scripts/evaluate.py` instead resolve the
architecture from the run's own `outputs/metadata/<run_name>.json` (written with every launch),
falling back to the plain YAML only when no metadata is found.

## Notebook

`notebooks/final_analysis.ipynb` produces the 7 requested figures (training curves; GT vs
reconstruction; residual maps; energy spectra; vorticity fields; latent t-SNE; MMD/Wasserstein +
residual distributions) plus a final baseline-vs-physics-informed summary table, reading
directly from checkpoints and CSV logs -- no training code in the notebook itself. Regenerate it
with `python build_notebook.py` after editing that script (it is built and validated via
`nbformat` rather than hand-edited as JSON).

## Hardware

Defaults (64x64, patch 8, embed_dim 128, depth 4+4, batch 16, AMP on) target a 4 GB GPU. Drop
`batch_size` to 8 first if you hit OOM, then `embed_dim`/`num_codes`. `num_workers: 0` in
`configs/experiment.yaml` is deliberate (Windows-safe DataLoader default).

## Extending

New dataset source: implement `prepare_<source>_split(split, n_windows, resolution, seed, cfg)`
matching the contract in `data/synthetic_solver.py`, register it in
`utils/registry.py::_lazy_dataset_registry`, add `configs/data_<source>.yaml`.

New metric: add it to the relevant `evaluation/*.py` module; wire it into
`training/callbacks.py` (TensorBoard/CSV) or `evaluation/evaluate_generation.py` (Stage 2 +
notebook) depending on which stage it belongs to.
