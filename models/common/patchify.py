"""Patch embedding: (B, C, H, W) field <-> (B, N, D) token sequence.

Non-overlapping square patches, standard ViT tokenization. A Conv2d with
kernel_size=stride=patch_size implements the patch-and-linear-project step
in one op; the inverse (PatchUnembed) is a Linear projection followed by
a reshape/permute back to a spatial grid.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class PatchEmbed(nn.Module):
    """Field -> token sequence.

    Args:
        in_channels: Input channel count (2 for velocity: Vx, Vy).
        patch_size: Side length of each square patch.
        embed_dim: Output token embedding dimension.
    """

    def __init__(self, in_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """
        Args:
            x: (B, C, H, W), with H and W divisible by patch_size.

        Returns:
            tokens: (B, N, D) where N = (H/patch_size) * (W/patch_size).
            grid_shape: (H/patch_size, W/patch_size), needed by PatchUnembed.
        """
        x = self.proj(x)  # (B, D, Hp, Wp)
        B, D, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens, (Hp, Wp)


class PatchUnembed(nn.Module):
    """Token sequence -> field (inverse of PatchEmbed).

    Args:
        out_channels: Output channel count.
        patch_size: Side length of each square patch (must match PatchEmbed).
        embed_dim: Input token embedding dimension.
    """

    def __init__(self, out_channels: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, tokens: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        """
        Args:
            tokens: (B, N, D).
            grid_shape: (Hp, Wp) as returned by PatchEmbed, with N = Hp * Wp.

        Returns:
            (B, out_channels, Hp*patch_size, Wp*patch_size) field.
        """
        Hp, Wp = grid_shape
        B, N, D = tokens.shape
        P, C = self.patch_size, self.out_channels

        x = self.proj(tokens)  # (B, N, C*P*P)
        x = x.reshape(B, Hp, Wp, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, Hp, P, Wp, P)
        x = x.reshape(B, C, Hp * P, Wp * P)
        return x
