"""Pre-LN Transformer blocks and positional/time embeddings.

TransformerBlock supports an optional AdaLN-Zero conditioning path (DiT
style: Peebles & Xie, "Scalable Diffusion Models with Transformers",
2023). Unconditional mode (cond_dim=None) is used by the VQ-VAE
encoder/decoder; conditional mode (cond_dim=embed_dim, driven by a
sinusoidal time embedding) is used by the flow-matching vector-field
network, which needs every block to see the integration time t. Sharing
one block implementation keeps the two models architecturally consistent
and halves the surface area for bugs.

The final AdaLN projection is zero-initialised (the "-Zero" in AdaLN-
Zero) so every conditioned block starts as an identity map -- a standard
trick for stabilising Transformer training from step 0, relevant given
this project's history of NaNs from under-controlled initial dynamics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_embedding(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
    """Transformer/diffusion-style sinusoidal embedding of a scalar per batch element.

    Args:
        t: (B,) tensor of scalars (e.g. flow-matching integration time in [0, 1]).
        dim: Output embedding dimension (must be even).
        max_period: Controls the lowest frequency.

    Returns:
        (B, dim) embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def sincos_pos_embed_2d(embed_dim: int, grid_h: int, grid_w: int, device: torch.device) -> Tensor:
    """Fixed (non-learned) 2D sin-cos positional embedding, as in MAE/ViT.

    Args:
        embed_dim: Total embedding dimension; must be divisible by 4.
        grid_h, grid_w: Token grid shape (from PatchEmbed).
        device: Target device.

    Returns:
        (grid_h * grid_w, embed_dim) embedding.
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D sincos, got {embed_dim}")
    dim_q = embed_dim // 4

    def embed_1d(pos: Tensor) -> Tensor:
        omega = torch.arange(dim_q, device=device, dtype=torch.float32) / dim_q
        omega = 1.0 / (10000**omega)
        out = pos[:, None] * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # (len(pos), 2*dim_q)

    hh = torch.arange(grid_h, device=device, dtype=torch.float32)
    ww = torch.arange(grid_w, device=device, dtype=torch.float32)
    mesh_h, mesh_w = torch.meshgrid(hh, ww, indexing="ij")

    emb_h = embed_1d(mesh_h.reshape(-1))
    emb_w = embed_1d(mesh_w.reshape(-1))
    return torch.cat([emb_h, emb_w], dim=1)  # (grid_h*grid_w, embed_dim)


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Mlp(nn.Module):
    """Standard transformer MLP: Linear -> GELU -> Linear."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-LN self-attention block, with optional AdaLN-Zero time conditioning.

    Args:
        dim: Token embedding dimension.
        n_heads: Number of attention heads.
        mlp_ratio: MLP hidden-dim multiplier.
        dropout: Attention dropout probability.
        cond_dim: If set, enables AdaLN-Zero conditioning driven by a
            (B, cond_dim) vector passed to forward(). If None, this is a
            plain unconditional Pre-LN block.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=cond_dim is None)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=cond_dim is None)
        self.mlp = Mlp(dim, mlp_ratio)

        if cond_dim is not None:
            self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
            nn.init.zeros_(self.ada_ln[-1].weight)
            nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """
        Args:
            x: (B, N, D) token sequence.
            cond: (B, cond_dim) conditioning vector; required if cond_dim was set.

        Returns:
            (B, N, D).
        """
        if self.cond_dim is not None:
            if cond is None:
                raise ValueError("This block was built with cond_dim set; forward() needs `cond`.")
            shift1, scale1, gate1, shift2, scale2, gate2 = self.ada_ln(cond).chunk(6, dim=-1)
            h = _modulate(self.norm1(x), shift1, scale1)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + gate1.unsqueeze(1) * attn_out
            h2 = _modulate(self.norm2(x), shift2, scale2)
            x = x + gate2.unsqueeze(1) * self.mlp(h2)
        else:
            h = self.norm1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    """A stack of TransformerBlocks with a final LayerNorm."""

    def __init__(
        self,
        dim: int,
        depth: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, mlp_ratio, dropout, cond_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return self.norm(x)
