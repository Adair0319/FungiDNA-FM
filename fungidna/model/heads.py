"""Task-specific classification/regression heads for downstream tasks."""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """[CLS]-token → MLP → logits (for tasks 1, 2, 3)."""
    def __init__(self, hidden_size: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls = hidden_states[:, 0, :]
        x = self.norm(cls)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


class DualBinaryHead(nn.Module):
    """Two independent binary heads for Task 2 (donor + acceptor)."""
    def __init__(self, hidden_size: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.donor_head = nn.Linear(hidden_dim, 2)
        self.acceptor_head = nn.Linear(hidden_dim, 2)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = hidden_states[:, 0, :]
        x = self.norm(cls)
        x = self.dropout(self.act(self.fc1(x)))
        return self.donor_head(x), self.acceptor_head(x)


