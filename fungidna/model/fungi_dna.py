"""Full FungiDNA model classes: MLM pretraining, Phase 2 contrastive, downstream classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone
from fungidna.model.heads import ClassificationHead, DualBinaryHead


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """Mean-pool hidden states, excluding padding positions.

    Args:
        hidden: [batch, seq_len, dim]
        attention_mask: [batch, seq_len] — 1 for real tokens, 0 for padding.
                        If None, mean-pools all positions.
    Returns:
        [batch, dim]
    """
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)  # [batch, seq_len, 1]
    sum_hidden = (hidden * mask).sum(dim=1)                      # [batch, dim]
    valid_len = mask.sum(dim=1).clamp(min=1e-9)                  # [batch, 1]
    return sum_hidden / valid_len


# ═══════════════════════════════════════════════════
# Phase 1: MLM Pretraining
# ═══════════════════════════════════════════════════

class FungiDNAForMLM(nn.Module):
    """Phase 1: Masked Language Modeling on BPE tokens."""

    def __init__(self, config: FungiDNAConfig):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.bpe_vocab_size, bias=False)
        # Tie weights: lm_head shares weights with BPE embedding
        self.lm_head.weight = self.backbone.embedding.bpe_embed.weight

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, add_cls: bool = True):
        hidden = self.backbone(input_ids, token_type=0, add_cls=add_cls)
        if add_cls:
            logits = self.lm_head(hidden[:, 1:, :])  # exclude [CLS]
        else:
            logits = self.lm_head(hidden)             # all positions are DNA tokens
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.bpe_vocab_size),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits} if loss is not None else logits


# ═══════════════════════════════════════════════════
# Phase 2: Contrastive + NSP Pretraining
# ═══════════════════════════════════════════════════

class FungiDNAForContrastive(nn.Module):
    """Phase 2: Contrastive learning + NSP + Species-ID, single-nt tokens."""

    def __init__(self, config: FungiDNAConfig, num_species: int = 735):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)
        # Projection head for contrastive
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.nsp_head = nn.Linear(config.hidden_size, 2)
        self.species_head = nn.Linear(config.hidden_size, num_species)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode a single-nt sequence and return L2-normalized projection."""
        hidden = self.backbone(input_ids, token_type=1)
        pooled = hidden[:, 1:, :].mean(dim=1)  # mean-pool excluding [CLS]
        return F.normalize(self.projection(pooled), dim=-1)

    def forward(self, batch: dict):
        device = next(self.parameters()).device
        # Contrastive
        anchor = self.encode(batch["anchor_ids"].to(device))
        positive = self.encode(batch["positive_ids"].to(device))
        negative = self.encode(batch["negative_ids"].to(device))

        # InfoNCE contrastive loss (simplified)
        pos_sim = (anchor * positive).sum(dim=-1)
        neg_sim = (anchor * negative).sum(dim=-1)
        temperature = 0.07
        logits = torch.stack([pos_sim, neg_sim], dim=1) / temperature
        contrastive_loss = F.cross_entropy(logits, torch.zeros(anchor.shape[0], device=device, dtype=torch.long))

        # NSP
        nsp_hidden = self.backbone(batch["nsp_seg1"].to(device), token_type=1)
        nsp_pooled = nsp_hidden[:, 1:, :].mean(dim=1)
        nsp_logits = self.nsp_head(nsp_pooled)
        nsp_loss = F.cross_entropy(nsp_logits, batch["nsp_label"].to(device))

        # Species-ID: use raw backbone pooled output (768-dim), not projected (128-dim)
        anchor_hidden = self.backbone(batch["anchor_ids"].to(device), token_type=1)
        anchor_pooled = anchor_hidden[:, 1:, :].mean(dim=1)
        species_logits = self.species_head(anchor_pooled)
        species_loss = F.cross_entropy(species_logits, batch["species_id"].to(device))

        loss = contrastive_loss + 0.3 * nsp_loss + 0.1 * species_loss
        return {
            "loss": loss,
            "contrastive_loss": contrastive_loss.detach(),
            "nsp_loss": nsp_loss.detach(),
            "species_loss": species_loss.detach(),
        }


# ═══════════════════════════════════════════════════
# Phase 2: Joint MLM + Contrastive Training
# ═══════════════════════════════════════════════════

class FungiDNAForJointTraining(nn.Module):
    """Phase 2: Joint MLM + Contrastive training (no [CLS], BPE tokens).

    Single backbone forward — masked input goes through backbone once,
    hidden states feed both MLM head and CL projection head.
    Loss = Loss_MLM + lambda(t) * Loss_CL

    LM head shares weights with BPE embedding (tied, from Phase 1).
    Projection head is randomly initialized.
    """

    def __init__(self, config: FungiDNAConfig, num_species: int = 735):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)

        # LM head (tied with BPE embedding)
        self.lm_head = nn.Linear(config.hidden_size, config.bpe_vocab_size, bias=False)
        self.lm_head.weight = self.backbone.embedding.bpe_embed.weight

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """Encode clean sequences -> L2-normalized embeddings (for Snapshot Matrix)."""
        hidden = self.backbone(input_ids, token_type=0, add_cls=False)
        pooled = mean_pool(hidden, attention_mask)
        z = self.projection(pooled)
        return F.normalize(z, p=2, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
        species_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        lambda_val: float,
    ) -> dict:
        """Single backbone forward → both MLM and CL heads.

        The input is MASKED BPE tokens — the same hidden representation
        serves both MLM (predict masked tokens) and CL (species clustering).

        Args:
            input_ids: [batch, seq_len] masked BPE tokens
            mlm_labels: [batch, seq_len] -100 for unmasked positions
            species_labels: [batch] integer species indices
            attention_mask: [batch, seq_len] 1=real, 0=pad
            lambda_val: current lambda(t) weight for CL loss

        Returns:
            {"loss": Tensor, "mlm_loss": Tensor, "cl_loss": Tensor}
        """
        # Single backbone forward
        hidden = self.backbone(input_ids, token_type=0, add_cls=False)

        # MLM head
        logits = self.lm_head(hidden)  # [batch, seq_len, vocab]
        loss_mlm = F.cross_entropy(
            logits.reshape(-1, self.config.bpe_vocab_size),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )

        # CL head — same hidden states, mean-pooled + projected
        pooled = mean_pool(hidden, attention_mask)
        z = F.normalize(self.projection(pooled), p=2, dim=-1)

        from fungidna.training.losses import supcon_loss
        loss_cl = supcon_loss(z, species_labels, temperature=0.1)

        total_loss = loss_mlm + lambda_val * loss_cl

        return {
            "loss": total_loss,
            "mlm_loss": loss_mlm.detach(),
            "cl_loss": loss_cl.detach(),
        }

    def forward_weighted(
        self,
        input_ids: torch.Tensor,
        mlm_labels: torch.Tensor,
        species_labels: torch.Tensor,
        attention_mask: torch.Tensor,
        lambda_val: float,
        batch_weight_matrix: torch.Tensor,
    ) -> dict:
        """Same as forward() but uses weighted_supcon_loss with Snapshot Matrix weights."""
        hidden = self.backbone(input_ids, token_type=0, add_cls=False)

        logits = self.lm_head(hidden)
        loss_mlm = F.cross_entropy(
            logits.reshape(-1, self.config.bpe_vocab_size),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )

        pooled = mean_pool(hidden, attention_mask)
        z = F.normalize(self.projection(pooled), p=2, dim=-1)

        from fungidna.training.losses import weighted_supcon_loss
        loss_cl = weighted_supcon_loss(z, species_labels, batch_weight_matrix, temperature=0.1)

        total_loss = loss_mlm + lambda_val * loss_cl

        return {
            "loss": total_loss,
            "mlm_loss": loss_mlm.detach(),
            "cl_loss": loss_cl.detach(),
        }


# ═══════════════════════════════════════════════════
# Downstream Task Models
# ═══════════════════════════════════════════════════

class FungiDNAForSequenceClassification(nn.Module):
    """Downstream: [CLS]-token classification (Tasks 1, 3, 4)."""
    def __init__(self, config: FungiDNAConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)
        self.head = ClassificationHead(config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids, token_type=0)
        return self.head(hidden)


class FungiDNAForSpliceSite(nn.Module):
    """Downstream: dual binary classification for donor and acceptor (Task 2)."""
    def __init__(self, config: FungiDNAConfig):
        super().__init__()
        self.config = config
        self.backbone = StripedMambaBackbone(config)
        self.head = DualBinaryHead(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids, token_type=0)
        return self.head(hidden)
