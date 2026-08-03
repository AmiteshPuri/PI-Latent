"""Latent Flow Matching generator (Stage 2).

Chosen over latent diffusion per the project's stated research interest:
architecturally the two are close cousins (a transformer regressing a
per-token vector, conditioned on a scalar time/noise-level via AdaLN),
but flow matching's straight-line probability path (Lipman et al., "Flow
Matching for Generative Modeling", 2023; equivalently Liu et al.'s
"Rectified Flow", 2022) gives a deterministic ODE with a well-defined
number-of-function-evaluations (NFE) vs. sample-quality trade-off at a
fixed, small set of integration steps -- the training objective itself
(conditional flow matching regression) lives in training/losses.py; this
module is architecture plus the sampling-time ODE integrator only.

Operates on the CONTINUOUS codebook embedding sequence (B, N, code_dim)
obtained by looking up a sample's codebook indices (from
TransformerVQVAE.encode_to_indices) in the frozen codebook -- not
directly on the discrete indices, since flow matching is a continuous-
state method. At generation time, the ODE's terminal state is snapped to
the nearest codebook vector (VectorQuantizerEMA.nearest_codes) before
decoding, matching the distribution the frozen decoder was trained on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.common.initialization import init_weights
from models.common.transformer_blocks import TransformerStack, sincos_pos_embed_2d, sinusoidal_embedding


class LatentFlowMatcher(nn.Module):
    """Transformer vector field v_theta(x_t, t) over a token sequence.

    Args:
        grid_shape: (Hp, Wp) token grid shape, must match the frozen
            VQ-VAE's grid_shape so positional embeddings align.
        code_dim: Token dimension (must match the frozen VQ-VAE's code_dim).
        embed_dim: Internal transformer width.
        depth: Number of AdaLN-conditioned Transformer blocks.
        n_heads: Attention heads.
        mlp_ratio: MLP hidden-dim multiplier.
        dropout: Attention dropout.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int] = (8, 8),
        code_dim: int = 128,
        embed_dim: int = 128,
        depth: int = 6,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_shape = grid_shape
        self.n_tokens = grid_shape[0] * grid_shape[1]
        self.code_dim = code_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Linear(code_dim, embed_dim)
        pos_embed = sincos_pos_embed_2d(embed_dim, *grid_shape, device=torch.device("cpu"))
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )
        self.transformer = TransformerStack(embed_dim, depth, n_heads, mlp_ratio, dropout, cond_dim=embed_dim)
        self.output_proj = nn.Linear(embed_dim, code_dim)

        init_weights(self)
        # Zero-init the final layer AFTER init_weights (which would otherwise overwrite
        # this): a flow-matching/diffusion vector-field head that starts at exactly zero
        # is a standard stabilisation trick (as in DiT) -- early training starts from a
        # well-defined "predict nothing" state rather than a random velocity.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x_t: (B, N, code_dim) latent tokens at interpolation time t.
            t: (B,) tensor in [0, 1].

        Returns:
            (B, N, code_dim) predicted velocity dx/dt.
        """
        h = self.input_proj(x_t) + self.pos_embed
        cond = self.time_mlp(sinusoidal_embedding(t, self.embed_dim))
        h = self.transformer(h, cond)
        return self.output_proj(h)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        n_steps: int = 50,
        method: str = "euler",
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate the learned ODE dx/dt = v_theta(x, t) from t=0 (Gaussian noise) to t=1.

        Args:
            batch_size: Number of samples to generate.
            device: Target device.
            n_steps: Number of integration steps (NFE = n_steps for
                'euler', 2*n_steps for 'heun'). This is the flow-matching
                NFE-vs-quality knob referenced throughout evaluation/.
            method: 'euler' (1st order, cheapest) or 'heun' (2nd order
                Runge-Kutta, 2x cost per step).
            generator: Optional torch.Generator for reproducible sampling.

        Returns:
            (batch_size, N, code_dim) continuous terminal latent tokens
            (NOT yet snapped to the codebook -- callers should apply
            VectorQuantizerEMA.nearest_codes before decoding).
        """
        shape = (batch_size, self.n_tokens, self.code_dim)
        x = torch.randn(shape, device=device, generator=generator)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            if method == "euler":
                x = x + dt * self.forward(x, t)
            elif method == "heun":
                v1 = self.forward(x, t)
                x_pred = x + dt * v1
                t_next = torch.full((batch_size,), (i + 1) * dt, device=device)
                v2 = self.forward(x_pred, t_next)
                x = x + dt * 0.5 * (v1 + v2)
            else:
                raise ValueError(f"Unknown ODE method '{method}', expected 'euler' or 'heun'.")
        return x
