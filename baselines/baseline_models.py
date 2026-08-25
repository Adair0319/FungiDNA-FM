"""CNN and Transformer baseline models for species classification."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBaseline(nn.Module):
    """3-layer 1D-CNN for end-to-end DNA sequence classification."""

    def __init__(self, vocab_size=4096, embedding_dim=128, num_classes=6, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=11, stride=5, padding=5)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=3)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.layer_norm = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (B, L, E)
        x = x.transpose(1, 2)  # (B, E, L)

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))

        x = self.pool(x).squeeze(-1)  # (B, 256)
        x = self.layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        return self.classifier(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerBaseline(nn.Module):
    """1-layer Transformer Encoder for end-to-end DNA sequence classification."""

    def __init__(self, vocab_size=4096, d_model=256, nhead=4, dim_feedforward=1024,
                 num_classes=6, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=(input_ids == 0))
        x = x.mean(dim=1)  # mean pool
        return self.classifier(x)


def create_baseline_model(model_type, num_classes, vocab_size=4096):
    """Factory for baseline models."""
    if model_type == "cnn":
        return CNNBaseline(vocab_size=vocab_size, num_classes=num_classes)
    elif model_type == "transformer":
        return TransformerBaseline(vocab_size=vocab_size, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
