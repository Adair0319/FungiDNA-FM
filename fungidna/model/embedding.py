"""Dual-token embedding with RoPE for FungiDNA."""
import torch
import torch.nn as nn
from fungidna.model.config import FungiDNAConfig


class RotaryPositionEmbedding(nn.Module):
    """RoPE — shared between Mamba2 and Attention layers."""

    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 131072):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().to(dtype=torch.bfloat16)
        sin = freqs.sin().to(dtype=torch.bfloat16)
        return cos, sin

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE. x: (..., seq_len, dim). dim must be even."""
        half = x.shape[-1] // 2
        x_rot = x[..., :half]
        x_pass = x[..., half:]
        c = cos[:x.shape[-2], :half].unsqueeze(0)
        s = sin[:x.shape[-2], :half].unsqueeze(0)
        out_rot = x_rot * c - x_pass * s
        out_pass = x_rot * s + x_pass * c
        return torch.cat([out_rot, out_pass], dim=-1)


class DualTokenEmbedding(nn.Module):
    """Dual embedding: BPE (vocab=4096) + single-nt (vocab=5), plus token-type and [CLS]."""

    def __init__(self, config: FungiDNAConfig):
        super().__init__()
        self.config = config
        self.bpe_embed = nn.Embedding(config.bpe_vocab_size, config.hidden_size, padding_idx=0)
        self.nt_embed = nn.Embedding(config.nt_vocab_size, config.hidden_size, padding_idx=4)
        self.token_type_embed = nn.Embedding(2, config.hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.rope = RotaryPositionEmbedding(
            config.hidden_size, config.rope_theta, config.max_position_embeddings
        )

    def forward(self, input_ids: torch.Tensor, token_type: int = 0, add_cls: bool = True):
        """Returns (hidden_states, cos, sin). When add_cls=True, [CLS] is prepended (pos 0)."""
        batch_size, seq_len = input_ids.shape
        if token_type == 0:
            x = self.bpe_embed(input_ids)
        else:
            x = self.nt_embed(input_ids)

        tt_ids = torch.full((batch_size, seq_len), token_type, device=input_ids.device, dtype=torch.long)
        x = x + self.token_type_embed(tt_ids)

        if add_cls:
            cls = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.layer_norm(x)
        x = self.dropout(x)
        cos, sin = self.rope(x.shape[1], x.device)
        return x, cos, sin
