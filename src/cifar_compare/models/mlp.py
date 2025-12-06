from __future__ import annotations

import torch.nn as nn


class MLPClassifier(nn.Module):
    """Simple MLP baseline ignoring spatial structure."""

    def __init__(self, input_dim: int = 32 * 32 * 3, hidden_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

