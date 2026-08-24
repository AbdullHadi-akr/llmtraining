"""Point-wise MLP with learnable Swish activations."""

from __future__ import annotations

import torch
from torch import nn


class LearnableSwish(nn.Module):
    """Swish activation with its own trainable beta parameter."""

    def __init__(self, init_beta: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class PointwiseMLP(nn.Module):
    """Point-wise MLP that predicts [T, bc_V] from 11 input features."""

    def __init__(
        self,
        n_features: int,
        n_hidden_layers: int,
        hidden_size: int,
        swish_beta_init: float = 1.0,
        swish_beta_learnable: bool = True,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 1:
            raise ValueError("n_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            activation = LearnableSwish(swish_beta_init)
            if not swish_beta_learnable:
                activation.beta.requires_grad_(False)
            layers.append(activation)
            in_dim = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    @property
    def betas(self) -> list[float]:
        return [m.beta.detach().item() for m in self.modules() if isinstance(m, LearnableSwish)]

    @property
    def n_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
