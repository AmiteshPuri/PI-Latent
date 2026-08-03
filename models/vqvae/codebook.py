"""Vector quantizer with EMA codebook updates and dead-code reset.

EMA updates (Razavi et al., "Generating Diverse High-Fidelity Images with
VQ-VAE-2", 2019) rather than a gradient-based codebook loss: the codebook
targets are a moving average of assigned encoder outputs, computed
without backprop, which is more stable than letting a codebook-loss
gradient term fight the commitment-loss gradient term directly -- the
kind of instability that has previously produced NaNs and lambda-scaling
problems in this codebase's sibling physics-informed VAE projects.

Dead-code reset (Kaiser & Roy, "Theory and Experiments on Vector
Quantized Autoencoders", 2018 -- also used in Jukebox, Dhariwal et al.
2020): a codebook entry unused for `reset_after_n_batches` consecutive
batches is respawned at a random encoder output from the current batch.
Without this, VQ-VAEs commonly collapse to using a small fraction of the
codebook, which both wastes capacity and is exactly what the
`codebook_utilization`/`perplexity` metrics this project tracks are
designed to catch -- this module is what keeps them healthy in the first
place, not just what gets measured after the fact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VectorQuantizerEMA(nn.Module):
    """EMA-updated vector quantizer with straight-through gradients.

    Args:
        num_codes: Codebook size K.
        code_dim: Dimension D of each code vector (must match the
            encoder's output token dimension).
        commitment_weight: Weight on the commitment loss term
            (encourages encoder outputs to stay close to their assigned
            code; standard default 0.25 from the original VQ-VAE paper).
        decay: EMA decay rate for codebook updates.
        epsilon: Laplace smoothing constant for cluster-size normalisation.
        reset_after_n_batches: Consecutive unused batches before a code
            is respawned. Set to 0 to disable dead-code reset.
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 128,
        commitment_weight: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        reset_after_n_batches: int = 50,
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.epsilon = epsilon
        self.reset_after_n_batches = reset_after_n_batches

        embedding = torch.randn(num_codes, code_dim) * 0.02
        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("embed_avg", embedding.clone())
        self.register_buffer("stale_batches", torch.zeros(num_codes))

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """
        Args:
            z: (B, N, D) continuous encoder output (pre-quantization).

        Returns:
            Dict with:
                quantized: (B, N, D) quantized tokens, straight-through
                    gradient (backward pass acts as identity w.r.t. z).
                indices: (B, N) long tensor of codebook indices.
                commitment_loss: Scalar tensor.
                perplexity: Scalar tensor, exp(entropy of code usage in
                    this batch). Ranges from 1 (single code used) to
                    num_codes (perfectly uniform usage).
        """
        B, N, D = z.shape
        flat = z.reshape(-1, D)

        distances = (
            (flat**2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + (self.embedding**2).sum(1)
        )
        indices = distances.argmin(dim=1)  # (B*N,)
        one_hot = F.one_hot(indices, self.num_codes).to(flat.dtype)  # (B*N, K)
        quantized_flat = one_hot @ self.embedding
        quantized = quantized_flat.reshape(B, N, D)

        if self.training:
            self._ema_update(flat, one_hot)

        commitment_loss = self.commitment_weight * F.mse_loss(z, quantized.detach())
        quantized_st = z + (quantized - z).detach()  # straight-through estimator

        avg_probs = one_hot.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return {
            "quantized": quantized_st,
            "indices": indices.reshape(B, N),
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
        }

    @torch.no_grad()
    def _ema_update(self, flat: Tensor, one_hot: Tensor) -> None:
        cluster_size_batch = one_hot.sum(dim=0)  # (K,)
        embed_sum_batch = one_hot.t() @ flat  # (K, D)

        self.cluster_size.mul_(self.decay).add_(cluster_size_batch, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum_batch, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        cluster_size_norm = (self.cluster_size + self.epsilon) / (n + self.num_codes * self.epsilon) * n
        self.embedding.copy_(self.embed_avg / cluster_size_norm.unsqueeze(1))

        if self.reset_after_n_batches > 0:
            used = cluster_size_batch > 0
            self.stale_batches[used] = 0
            self.stale_batches[~used] += 1

            dead = self.stale_batches >= self.reset_after_n_batches
            n_dead = int(dead.sum().item())
            if n_dead > 0:
                replacement_idx = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
                replacement = flat[replacement_idx].detach()
                self.embedding[dead] = replacement
                self.embed_avg[dead] = replacement
                self.cluster_size[dead] = 1.0
                self.stale_batches[dead] = 0

    @torch.no_grad()
    def lookup(self, indices: Tensor) -> Tensor:
        """Embed a tensor of codebook indices back into continuous vectors.

        Args:
            indices: (...,) long tensor of indices in [0, num_codes).

        Returns:
            (..., code_dim) embedded tensor.
        """
        return self.embedding[indices]

    @torch.no_grad()
    def nearest_codes(self, z: Tensor) -> Tensor:
        """Snap arbitrary continuous vectors to their nearest codebook entry.

        Used by the flow-matching sampler to project generated continuous
        latents back onto the codebook before decoding, matching the
        distribution the decoder was actually trained on.

        Args:
            z: (..., code_dim) continuous vectors.

        Returns:
            (..., code_dim) quantized vectors (same shape as z).
        """
        shape = z.shape
        flat = z.reshape(-1, self.code_dim)
        distances = (
            (flat**2).sum(1, keepdim=True)
            - 2 * flat @ self.embedding.t()
            + (self.embedding**2).sum(1)
        )
        indices = distances.argmin(dim=1)
        return self.embedding[indices].reshape(shape)
