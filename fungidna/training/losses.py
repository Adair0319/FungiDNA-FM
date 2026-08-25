"""Contrastive loss functions: standard SupCon and weighted variant."""
import torch
import torch.nn.functional as F


def supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor, positives are samples with the same label.
    Negatives are all other samples in the batch.

    Args:
        z: L2-normalized embeddings [N, dim]
        labels: integer class labels [N]
        temperature: softmax temperature (default 0.1)

    Returns:
        scalar loss
    """
    N = z.shape[0]
    device = z.device

    # Cosine similarity matrix [N, N]
    sim = torch.matmul(z, z.T) / temperature

    # Mask: same-label pairs (excluding self)
    labels = labels.reshape(-1, 1)
    pos_mask = (labels == labels.T).float()  # [N, N]
    pos_mask.fill_diagonal_(0.0)

    # For numerical stability: subtract max per row
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    # exp of similarities
    exp_sim = torch.exp(sim)

    # Denominator: sum over all except self
    denom = (exp_sim * (1.0 - torch.eye(N, device=device))).sum(dim=1, keepdim=True)

    # Numerator: sum over positives
    num_pos = pos_mask.sum(dim=1)  # [N]
    # Guard against samples with no positives
    valid_mask = num_pos > 0

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Log-probabilities for positive pairs
    log_prob = torch.log(exp_sim / denom + 1e-8)  # [N, N]

    # Mean over positives
    pos_log_prob = (pos_mask * log_prob).sum(dim=1)  # [N]
    loss = -(pos_log_prob[valid_mask] / num_pos[valid_mask]).mean()

    return loss


def weighted_supcon_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    weight_matrix: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Weighted Supervised Contrastive Loss.

    Extends supcon_loss by scaling the repulsion between negative pairs
    using a pre-computed similarity weight matrix.

    For anchor i and negative j: repulsion ∝ (1 - W_ij) in the denominator,
    so more similar species are pushed apart less aggressively.

    Args:
        z: L2-normalized embeddings [N, dim]
        labels: integer class labels [N]
        weight_matrix: [num_classes, num_classes] pre-computed similarity matrix W.
                       W[i,j] ∈ [0,1]; higher = more similar = less repulsion.
        temperature: softmax temperature (default 0.1)

    Returns:
        scalar loss
    """
    N = z.shape[0]
    device = z.device

    # Cosine similarity [N, N]
    sim = torch.matmul(z, z.T) / temperature

    # Positive mask
    labels = labels.reshape(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0.0)

    # Build batch-level weight matrix from species labels
    # W_batch[i,j] = weight_matrix[label[i], label[j]]
    batch_W = weight_matrix[labels.squeeze()][:, labels.squeeze()]  # [N, N]

    # Numerical stability
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    exp_sim = torch.exp(sim)

    # Weighted denominator: sum over all except self, weighted by (1 - W)
    neg_mask = 1.0 - torch.eye(N, device=device) - pos_mask  # all non-self, non-positive
    neg_weights = 1.0 - batch_W  # [N, N]: high W → low repulsion weight
    weighted_exp = exp_sim * (pos_mask + neg_mask * neg_weights)

    denom = weighted_exp.sum(dim=1, keepdim=True)

    num_pos = pos_mask.sum(dim=1)
    valid_mask = num_pos > 0

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    log_prob = torch.log(exp_sim / (denom + 1e-8))
    pos_log_prob = (pos_mask * log_prob).sum(dim=1)
    loss = -(pos_log_prob[valid_mask] / num_pos[valid_mask]).mean()

    return loss
