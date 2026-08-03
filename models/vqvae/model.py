"""Transformer VQ-VAE for 2D velocity fields.

Architecture (identical for the baseline and physics-informed variants --
the two differ only in which loss terms training/losses.py backprops
through, not in architecture, per the project spec):

    Input (B, 2, H, W) velocity field
    -> PatchEmbed              (2 -> embed_dim, ViT-style patch tokens)
    -> + 2D sin-cos pos embed
    -> Transformer encoder     (N_enc pre-LN blocks)
    -> EMA vector quantizer    (codebook lookup, straight-through grad)
    -> Transformer decoder     (N_dec pre-LN blocks)
    -> PatchUnembed             (embed_dim -> 2)
    -> Output (B, 2, H, W) reconstructed velocity field

No output activation (no tanh/sigmoid clamp): inputs are z-score
normalised (data/datamodule.py), so targets are roughly N(0, 1) and a
linear output head is the correct match -- a saturating activation here
previously caused a severe reconstruction-quality bug in a sibling
project (darcy_flow_pinn_vae) when the target range didn't match the
activation's range.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.common.initialization import init_weights
from models.common.patchify import PatchEmbed, PatchUnembed
from models.common.transformer_blocks import TransformerStack, sincos_pos_embed_2d
from models.vqvae.codebook import VectorQuantizerEMA


class TransformerVQVAE(nn.Module):
    """VQ-VAE with a Transformer encoder and decoder over patch tokens.

    Args:
        in_channels: Input/output channel count (2 for velocity: Vx, Vy).
        resolution: Spatial resolution H = W (must be divisible by patch_size).
        patch_size: Side length of each square patch.
        embed_dim: Transformer token dimension.
        encoder_depth: Number of encoder Transformer blocks.
        decoder_depth: Number of decoder Transformer blocks.
        n_heads: Attention heads.
        mlp_ratio: MLP hidden-dim multiplier.
        num_codes: Codebook size K.
        code_dim: Codebook vector dimension (defaults to embed_dim).
        commitment_weight: VQ commitment loss weight.
        ema_decay: Codebook EMA decay rate.
        reset_after_n_batches: Dead-code reset threshold (see codebook.py).
        dropout: Attention dropout.
    """

    def __init__(
        self,
        in_channels: int = 2,
        resolution: int = 64,
        patch_size: int = 8,
        embed_dim: int = 128,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_codes: int = 512,
        code_dim: int | None = None,
        commitment_weight: float = 0.25,
        ema_decay: float = 0.99,
        reset_after_n_batches: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if resolution % patch_size != 0:
            raise ValueError(f"resolution ({resolution}) must be divisible by patch_size ({patch_size})")

        code_dim = code_dim or embed_dim
        self.grid_shape = (resolution // patch_size, resolution // patch_size)
        self.n_tokens = self.grid_shape[0] * self.grid_shape[1]

        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        pos_embed = sincos_pos_embed_2d(embed_dim, *self.grid_shape, device=torch.device("cpu"))
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # (1, N, D), fixed (not learned)

        self.encoder = TransformerStack(embed_dim, encoder_depth, n_heads, mlp_ratio, dropout)
        self.pre_quant_proj = nn.Linear(embed_dim, code_dim) if code_dim != embed_dim else nn.Identity()
        self.post_quant_proj = nn.Linear(code_dim, embed_dim) if code_dim != embed_dim else nn.Identity()

        self.codebook = VectorQuantizerEMA(
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            decay=ema_decay,
            reset_after_n_batches=reset_after_n_batches,
        )

        self.decoder = TransformerStack(embed_dim, decoder_depth, n_heads, mlp_ratio, dropout)
        self.patch_unembed = PatchUnembed(in_channels, patch_size, embed_dim)

        init_weights(self)  # safe w.r.t. the codebook: it has no Linear/LayerNorm/Embedding/Conv2d submodules

    def encode(self, x: Tensor) -> Tensor:
        """Field -> pre-quantization continuous latent tokens (B, N, code_dim)."""
        tokens, grid_shape = self.patch_embed(x)
        assert grid_shape == self.grid_shape, f"grid_shape mismatch: {grid_shape} vs {self.grid_shape}"
        tokens = tokens + self.pos_embed
        encoded = self.encoder(tokens)
        return self.pre_quant_proj(encoded)

    def decode(self, quantized: Tensor) -> Tensor:
        """Quantized latent tokens (B, N, code_dim) -> reconstructed field (B, C, H, W)."""
        h = self.post_quant_proj(quantized)
        h = h + self.pos_embed
        decoded = self.decoder(h)
        return self.patch_unembed(decoded, self.grid_shape)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: (B, C, H, W) normalised velocity field.

        Returns:
            Dict with 'reconstruction' (B, C, H, W), 'indices' (B, N),
            'commitment_loss' (scalar), 'perplexity' (scalar).
        """
        z = self.encode(x)
        vq_out = self.codebook(z)
        reconstruction = self.decode(vq_out["quantized"])
        return {
            "reconstruction": reconstruction,
            "indices": vq_out["indices"],
            "commitment_loss": vq_out["commitment_loss"],
            "perplexity": vq_out["perplexity"],
        }

    @torch.no_grad()
    def encode_to_indices(self, x: Tensor) -> Tensor:
        """Field -> codebook indices (B, N), for building the Stage-2 latent dataset.

        Call with the model in eval() mode (frozen VQ-VAE) so this does
        not perturb codebook EMA statistics.
        """
        z = self.encode(x)
        return self.codebook(z)["indices"]

    @torch.no_grad()
    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Codebook indices (B, N) -> reconstructed field (B, C, H, W), for Stage-2 generation."""
        quantized = self.codebook.lookup(indices)
        return self.decode(quantized)
