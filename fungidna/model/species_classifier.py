"""Species classification with independent heads + Incertae sedis masking.

Each rank has its own independent classification head. Loss for samples
labeled "Incertae sedis" (uncertain placement) is ignored for that rank.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fungidna.model.config import FungiDNAConfig
from fungidna.model.striped_mamba import StripedMambaBackbone

RANK_ORDER = ["phylum", "subphylum", "class", "order", "family"]


class FungiDNAForSpeciesClassification(nn.Module):
    """Independent-head classifier for 5-rank taxonomy prediction.

    Each head: [CLS] → LayerNorm → Linear(768→256) → GELU → Dropout → Linear(256→N)
    """

    def __init__(
        self,
        config: FungiDNAConfig,
        num_classes: dict[str, int],
        incertae_sedis_ids: dict[str, int] | None = None,
        freeze_backbone: bool = True,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.incertae_sedis_ids = incertae_sedis_ids or {}

        self.backbone = StripedMambaBackbone(config)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.heads = nn.ModuleDict()
        for rank in RANK_ORDER:
            self.heads[rank] = _ClassificationHead(
                config.hidden_size, num_classes[rank],
                hidden_dim=head_hidden, dropout=dropout,
            )

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns logits for each taxonomic rank."""
        hidden = self.backbone(input_ids, token_type=0)
        return {rank: head(hidden) for rank, head in self.heads.items()}

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        labels: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward + per-rank CE loss, ignoring Incertae sedis labels."""
        logits = self.forward(input_ids)
        losses = {}
        for rank in RANK_ORDER:
            ignore_idx = self.incertae_sedis_ids.get(rank)
            loss = F.cross_entropy(
                logits[rank], labels[rank],
                ignore_index=ignore_idx if ignore_idx is not None else -100,
            )
            # When all labels in a batch are ignored, cross_entropy returns NaN.
            # Replace with 0 so training continues.
            losses[rank] = torch.nan_to_num(loss, nan=0.0)
        return sum(losses.values()), losses


class _ClassificationHead(nn.Module):
    """LayerNorm → Linear → GELU → Dropout → Linear."""

    def __init__(self, hidden_size: int, num_classes: int, hidden_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls = hidden_states[:, 0, :]
        return self.fc2(self.dropout(F.gelu(self.fc1(self.norm(cls)))))


def class_balanced_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_samples_per_class: torch.Tensor,
    beta: float = 0.9999,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Class-balanced focal loss (Cui et al., CVPR 2019).

    Args:
        logits: (N, C) raw logits
        labels: (N,) integer labels
        num_samples_per_class: (C,) tensor of sample counts per class
        beta: re-weighting hyperparameter (0.999 = mild, 0.9999 = strong)
        gamma: focusing parameter (0 = plain CE, 2 = standard focal)
        ignore_index: label value to ignore in loss
    """
    effective_num = 1.0 - beta ** num_samples_per_class.float()
    class_weights = (1.0 - beta) / effective_num.clamp(min=1e-8)
    class_weights = class_weights / class_weights.mean()

    ce = F.cross_entropy(logits, labels, reduction="none", ignore_index=ignore_index)
    valid_mask = labels != ignore_index

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)

    pt = torch.exp(-ce)
    focal_weight = (1.0 - pt) ** gamma

    alpha = class_weights.to(logits.device)
    alpha_per_sample = torch.ones_like(labels, dtype=torch.float32).to(logits.device)
    alpha_per_sample[valid_mask] = alpha[labels[valid_mask].clamp(min=0)]

    loss = alpha_per_sample * focal_weight * ce
    return loss[valid_mask].mean()


class _HierarchicalClassificationHead(nn.Module):
    """Shared FC + 5 conditional heads with parent-label input.

    Shared FC: LayerNorm(768) -> Linear(768->512) -> GELU -> Dropout
    Each rank head: Linear(input_dim -> num_classes)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        shared_dim: int = 512,
        num_classes: dict | None = None,
        dropout: float = 0.1,
        conditional_heads: bool = True,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.conditional_heads = conditional_heads
        self.num_classes = num_classes or {}

        self.norm = nn.LayerNorm(hidden_size)
        self.shared_fc = nn.Linear(hidden_size, shared_dim)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleDict()
        prev_dim = shared_dim
        for rank in RANK_ORDER:
            n_class = self.num_classes.get(rank, 2)
            self.heads[rank] = nn.Linear(prev_dim, n_class)
            if conditional_heads:
                prev_dim = shared_dim + n_class
            else:
                prev_dim = shared_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        parent_labels: dict | None = None,
    ) -> dict:
        cls = hidden_states[:, 0, :]
        shared = self.dropout(F.gelu(self.shared_fc(self.norm(cls))))

        logits = {}
        input_feat = shared

        for rank in RANK_ORDER:
            pred = self.heads[rank](input_feat)
            logits[rank] = pred

            if self.conditional_heads:
                if parent_labels is not None and rank in parent_labels:
                    pl = parent_labels[rank]
                    parent = F.one_hot(
                        pl.clamp(min=0),
                        num_classes=self.num_classes[rank],
                    ).to(dtype=shared.dtype)
                    # Zero out parent signal for ignored labels (e.g. Incertae sedis)
                    ignore_mask = (pl < 0).unsqueeze(-1).to(dtype=shared.dtype)
                    parent = parent * (1.0 - ignore_mask)
                else:
                    parent = F.one_hot(
                        pred.argmax(dim=-1),
                        num_classes=self.num_classes[rank],
                    ).to(dtype=shared.dtype)
                input_feat = torch.cat([shared, parent], dim=-1)

        return logits


class FungiDNAForHierarchicalSpeciesClassification(nn.Module):
    """Hierarchical classifier with shared FC + conditional heads.

    Supports phased backbone unfreezing and weighted CE loss.
    """

    def __init__(
        self,
        config: FungiDNAConfig,
        num_classes: dict,
        incertae_sedis_ids: dict | None = None,
        freeze_backbone: bool = True,
        shared_dim: int = 512,
        head_dropout: float = 0.1,
        conditional_heads: bool = True,
        loss_weights: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.incertae_sedis_ids = incertae_sedis_ids or {}
        self.loss_weights = loss_weights or {
            "phylum": 1.5, "subphylum": 1.5, "class": 1.0,
            "order": 1.0, "family": 2.0,
        }

        self.backbone = StripedMambaBackbone(config)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = _HierarchicalClassificationHead(
            hidden_size=config.hidden_size,
            shared_dim=shared_dim,
            num_classes=num_classes,
            dropout=head_dropout,
            conditional_heads=conditional_heads,
        )

    def forward(self, input_ids: torch.Tensor) -> dict:
        hidden = self.backbone(input_ids, token_type=0)
        return self.head(hidden, parent_labels=None)

    def forward_with_loss(
        self,
        input_ids: torch.Tensor,
        labels: dict,
        use_focal_for_family: bool = False,
        family_sample_counts: torch.Tensor | None = None,
        focal_beta: float = 0.9999,
        focal_gamma: float = 2.0,
    ) -> tuple:
        hidden = self.backbone(input_ids, token_type=0)
        logits = self.head(hidden, parent_labels=labels)

        losses = {}
        for rank in RANK_ORDER:
            ignore_idx = self.incertae_sedis_ids.get(rank)
            weight = self.loss_weights.get(rank, 1.0)

            if use_focal_for_family and rank == "family" and family_sample_counts is not None:
                loss = class_balanced_focal_loss(
                    logits[rank], labels[rank],
                    num_samples_per_class=family_sample_counts,
                    beta=focal_beta, gamma=focal_gamma,
                    ignore_index=ignore_idx if ignore_idx is not None else -100,
                )
            else:
                loss = F.cross_entropy(
                    logits[rank], labels[rank],
                    ignore_index=ignore_idx if ignore_idx is not None else -100,
                )
            losses[rank] = weight * torch.nan_to_num(loss, nan=0.0)

        return sum(losses.values()), losses

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze the last n layers of the backbone."""
        total = len(self.backbone.layers)
        for i, layer in enumerate(self.backbone.layers):
            if i >= total - n:
                for p in layer.parameters():
                    p.requires_grad = True

    def freeze_all_layers(self):
        """Freeze all backbone layers."""
        for layer in self.backbone.layers:
            for p in layer.parameters():
                p.requires_grad = False
