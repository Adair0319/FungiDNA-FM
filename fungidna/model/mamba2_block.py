"""Mamba2 block: RMSNorm → Mamba2 → residual (with gradient checkpointing)."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt
from mamba_ssm import Mamba2


class Mamba2Block(nn.Module):
    """Pre-norm Mamba2 block with residual connection.

    Gradient checkpointing discards intermediate activations during forward
    and recomputes them during backward — trading ~25% slower training for
    ~60% less activation memory on Mamba's scan states.
    """

    def __init__(self, d_model: int = 768, d_state: int = 128, d_conv: int = 4, expand: int = 2, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=eps)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + ckpt(self._forward, x, use_reentrant=False)
