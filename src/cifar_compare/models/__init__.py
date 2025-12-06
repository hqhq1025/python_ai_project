from __future__ import annotations

from typing import Callable, Dict

from .cnn_small import SmallCNN
from .mlp import MLPClassifier
from .deep_cnn import DeepCNNLarge
from .resnet import build_resnet18
from .vit import build_vit_b16


MODEL_REGISTRY: Dict[str, Callable] = {
    "mlp": MLPClassifier,
    "small_cnn": SmallCNN,
    "deep_cnn": DeepCNNLarge,
    "resnet18": build_resnet18,
    "vit_b16": build_vit_b16,
    "vit": build_vit_b16,
}


def build_model(name: str, **kwargs):
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model name: {name}")
    return MODEL_REGISTRY[key](**kwargs)
