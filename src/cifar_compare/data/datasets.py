from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(model_type: str, train: bool = True) -> transforms.Compose:
    model_type = model_type.lower()
    if model_type.startswith("vit"):
        if train:
            return transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )


def _build_train_val_datasets(
    data_dir: Path,
    model_type: str,
    val_split: int,
    seed: int,
) -> Tuple[Subset, Subset]:
    train_transform = get_transforms(model_type, train=True)
    eval_transform = get_transforms(model_type, train=False)

    full_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    eval_train = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )

    train_size = len(full_train) - val_split
    val_size = val_split
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(len(full_train)),
        [train_size, val_size],
        generator=generator,
    )
    train_dataset = Subset(full_train, train_indices.indices)
    val_dataset = Subset(eval_train, val_indices.indices)
    return train_dataset, val_dataset


def get_dataloaders(
    model_type: str,
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: int = 5000,
    seed: int = 42,
    data_dir: str | Path | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train/val/test dataloaders for CIFAR-10.

    model_type: used to select transforms (cnn/resnet vs vit).
    """
    if data_dir:
        root = Path(data_dir)
    else:
        repo_local = Path(__file__).resolve().parent / "cifar-10-batches-py"
        root = repo_local.parent if repo_local.exists() else Path.home() / ".torch" / "datasets"
    train_dataset, val_dataset = _build_train_val_datasets(root, model_type, val_split, seed)

    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=get_transforms(model_type, train=False),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
