from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def get_device(device: str = "auto", prefer_gpu: bool = True) -> torch.device:
    """
    Select device.
    device: "auto" | "cpu" | "cuda" | "mps".
    prefer_gpu keeps backward compatibility for older callers.
    """
    device = device.lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # auto
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path | str) -> Path:
    """Create directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_checkpoint(state: Dict[str, Any], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(state, path)


def load_checkpoint(path: Path | str, map_location: torch.device | str = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def epoch_timer(start_time: float) -> float:
    """Return elapsed seconds since start_time."""
    return time.time() - start_time


def dump_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
