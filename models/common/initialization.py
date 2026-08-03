"""Weight initialisation, GPT/ViT-style: truncated-normal Linear/Embedding
weights (std=0.02), zero biases, unit-scale LayerNorm. Applied once at the
end of each model's __init__.
"""

from __future__ import annotations

import torch.nn as nn


def init_weights(module: nn.Module) -> None:
    """Apply GPT/ViT-style initialisation to every submodule in-place."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
