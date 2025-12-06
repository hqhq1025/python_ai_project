from __future__ import annotations

import torch.nn as nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit_b16(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("heads."):
                param.requires_grad = False
    return model

