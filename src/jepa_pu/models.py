from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class JEPAPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, repr_dim: int, dropout: float):
        super().__init__()
        self.repr_dim = repr_dim
        self.predictor = MLP(input_dim, hidden_dim, 3 * repr_dim, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        prediction = self.predictor(features)
        chunks = prediction.reshape(*prediction.shape[:-1], 3, self.repr_dim)
        return F.layer_norm(chunks, (self.repr_dim,)).flatten(-2)


class RiskHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.head = MLP(input_dim, hidden_dim, 1, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


def representation_loss(
    prediction: torch.Tensor, target: torch.Tensor, repr_dim: int, variance_weight: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred = prediction.reshape(-1, 3, repr_dim)
    truth = target.reshape(-1, 3, repr_dim).detach()
    cosine = 1.0 - F.cosine_similarity(pred, truth, dim=-1).mean()
    pred_std = torch.sqrt(pred.var(dim=0, unbiased=False) + 1e-4)
    variance = F.relu(1.0 - pred_std).mean()
    loss = cosine + variance_weight * variance
    return loss, {"cosine_loss": cosine.detach(), "variance_loss": variance.detach()}
