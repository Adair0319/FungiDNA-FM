"""Self-attention block: RMSNorm → FlashAttention-2 (RoPE) → residual → RMSNorm → SwiGLU MLP → residual.

Gradient checkpointing applied to both attention and MLP sub-blocks to
reduce activation memory for long sequences.
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as ckpt
from flash_attn import flash_attn_func
from fungidna.model.embedding import RotaryPositionEmbedding


class AttentionBlock(nn.Module):
    """Pre-norm self-attention + SwiGLU MLP, both with residual connections."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, head_dim: int = 64,
                 mlp_expand: int = 4, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        assert n_heads * head_dim == d_model, f"{n_heads}×{head_dim} != {d_model}"

        self.norm1 = nn.RMSNorm(d_model, eps=eps)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        swiglu_dim = int(d_model * mlp_expand * 2 / 3)
        swiglu_dim = ((swiglu_dim + 255) // 256) * 256
        self.swiglu_dim = swiglu_dim
        self.norm2 = nn.RMSNorm(d_model, eps=eps)
        self.gate_proj = nn.Linear(d_model, swiglu_dim, bias=False)
        self.up_proj = nn.Linear(d_model, swiglu_dim, bias=False)
        self.down_proj = nn.Linear(swiglu_dim, d_model, bias=False)

    def _attn_forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      batch: int, seq_len: int) -> torch.Tensor:
        """Checkpointed attention sub-block: norm → QKV → FlashAttention → output proj."""
        x = self.norm1(x)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        q = RotaryPositionEmbedding.apply_rotary(q, cos, sin)
        k = RotaryPositionEmbedding.apply_rotary(k, cos, sin)
        attn_out = flash_attn_func(q, k, v, causal=False)
        return self.o_proj(attn_out.reshape(batch, seq_len, self.d_model))

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Checkpointed MLP sub-block: norm → SwiGLU → down proj."""
        x = self.norm2(x)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        x = x + ckpt(self._attn_forward, x, cos, sin, batch, seq_len, use_reentrant=False)
        x = x + ckpt(self._mlp_forward, x, use_reentrant=False)
        return x
