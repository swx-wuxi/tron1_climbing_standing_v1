"""Small MLP used by the first TRON1 wheel actuator-network experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class ActuatorMLP(nn.Module):
    """Map a fixed wheel command/state history to one actuator response."""

    def __init__(self, input_dim: int, hidden_sizes: Sequence[int] = (64, 64)):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_sizes:
            layers.extend((nn.Linear(previous, int(width)), nn.Softsign()))
            previous = int(width)
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def make_model(input_dim: int, hidden_sizes: Sequence[int]) -> ActuatorMLP:
    model = ActuatorMLP(input_dim=input_dim, hidden_sizes=hidden_sizes)
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    return model

