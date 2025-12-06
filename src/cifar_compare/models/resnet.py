from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = False,
    cifar_stem: bool = True,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

