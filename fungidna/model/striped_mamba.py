"""StripedMamba backbone: 3×Mamba2 → 1×Attention, repeated 6 times (24 layers total)."""
import torch
import torch.nn as nn
from fungidna.model.config import FungiDNAConfig
from fungidna.model.embedding import DualTokenEmbedding
from fungidna.model.mamba2_block import Mamba2Block
from fungidna.model.attention_block import AttentionBlock


class StripedMambaBackbone(nn.Module):
    """StripedMamba backbone with dual-token embedding.

    Layers: [M, M, M, A] × 6 → 24 total (18 Mamba2 + 6 Attention).
    """

    def __init__(self, config: FungiDNAConfig):
        super().__init__()
        self.config = config
        self.embedding = DualTokenEmbedding(config)

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            is_attention = (i + 1) % (config.mamba_per_attention + 1) == 0
            if is_attention:
                self.layers.append(AttentionBlock(
                    d_model=config.hidden_size,
                    n_heads=config.num_attention_heads,
                    head_dim=config.attention_head_dim,
                    mlp_expand=config.mlp_expand,
                    eps=config.layer_norm_eps,
                ))
            else:
                self.layers.append(Mamba2Block(
                    d_model=config.hidden_size,
                    d_state=config.mamba_d_state,
                    d_conv=config.mamba_d_conv,
                    expand=config.mamba_expand,
                    eps=config.layer_norm_eps,
                ))

        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor, token_type: int = 0, add_cls: bool = True) -> torch.Tensor:
        """Returns last_hidden_state. When add_cls=True, [CLS] is at position 0."""
        x, cos, sin = self.embedding(input_ids, token_type=token_type, add_cls=add_cls)
        for i, layer in enumerate(self.layers):
            is_attn = (i + 1) % (self.config.mamba_per_attention + 1) == 0
            if is_attn:
                x = layer(x, cos, sin)
            else:
                x = layer(x)
        return self.final_norm(x)
