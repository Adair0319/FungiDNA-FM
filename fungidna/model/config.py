"""FungiDNA model configuration."""
from dataclasses import dataclass, field


@dataclass
class FungiDNAConfig:
    # ── Embedding ──
    bpe_vocab_size: int = 4096
    nt_vocab_size: int = 5
    hidden_size: int = 768
    max_position_embeddings: int = 131072

    # ── StripedMamba backbone ──
    num_layers: int = 24
    mamba_per_attention: int = 3   # Mamba2 layers per attention layer
    mamba_d_state: int = 128
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    num_attention_heads: int = 12
    attention_head_dim: int = 64
    mlp_expand: int = 4             # SwiGLU intermediate multiplier (× 2/3)
    layer_norm_eps: float = 1e-5
    dropout: float = 0.0

    # ── Token configuration ──
    use_cls: bool = True  # False = no [CLS] token, use mean-pool for sequence representation

    # ── RoPE ──
    rope_theta: float = 10000.0

    # ── Downstream defaults ──
    num_classes: int = 3
    num_bgc_enzyme_classes: int = 6
    num_go_terms: dict = field(default_factory=lambda: {"MF": 50, "BP": 50, "CC": 50})
    num_kegg_labels: int = 100

    # ── LoRA defaults ──
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    @property
    def num_mamba_layers(self) -> int:
        blocks = self.num_layers // (self.mamba_per_attention + 1)
        return blocks * self.mamba_per_attention

    @property
    def num_attention_layers(self) -> int:
        return self.num_layers // (self.mamba_per_attention + 1)

    @property
    def swiglu_intermediate(self) -> int:
        dim = int(self.hidden_size * self.mlp_expand * 2 / 3)
        return ((dim + 255) // 256) * 256  # round to 256x for tensor cores
