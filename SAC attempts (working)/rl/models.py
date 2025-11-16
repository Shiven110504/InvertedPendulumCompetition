import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP used by both actor and critic networks."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: nn.Module = nn.ReLU):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        layers = []
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(din, dout))
            layers.append(activation())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            bound = 1.0 / math.sqrt(module.weight.shape[0])
            nn.init.uniform_(module.weight, -bound, bound)
            nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_hidden_dims(value: Iterable[int] | None) -> Sequence[int]:
    if value is None:
        return (256, 256)
    return tuple(int(v) for v in value)


class ResidualPolicy(nn.Module):
    """Deterministic policy wrapper used for deployment in YourControlCode."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dims: Sequence[int] = (256, 256)):
        super().__init__()
        self.backbone = MLP(input_dim, hidden_dims, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.backbone(obs))
