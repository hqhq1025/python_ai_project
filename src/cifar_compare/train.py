from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn, optim

from cifar_compare.data.datasets import get_dataloaders
from cifar_compare.models import build_model
from cifar_compare.utils import (
    count_parameters,
    dump_json,
    ensure_dir,
    get_device,
    set_seed,
)


def setup_wandb(args: argparse.Namespace, config: Dict) -> object | None:
    if not args.wandb:
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        print("wandb 未安装，跳过日志。可通过 `python -m pip install wandb` 安装或使用 `pip install -e .[logging]`。")
        return None

    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.model}",
        config=config,
    )
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 模型对比训练脚本")
    parser.add_argument("--model", type=str, default="small_cnn", help="模型名称: mlp | small_cnn | resnet18 | vit_b16")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=int, default=5000)
    parser.add_argument("--data-dir", type=str, default=None, help="数据存放路径（默认 ~/.torch/datasets）")
    parser.add_argument("--output-dir", type=str, default="outputs", help="日志和模型输出路径")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="选择设备：auto 优先 cuda/mps，其次 cpu",
    )
    parser.add_argument("--use-gpu", action="store_true", help="若可用则使用 GPU（兼容旧参数）")
    parser.add_argument("--pretrained", action="store_true", help="ResNet/ViT 是否加载预训练权重")
    parser.add_argument("--freeze-backbone", action="store_true", help="ViT 是否冻结骨干网络")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb 日志（需安装 wandb）")
    parser.add_argument("--wandb-project", type=str, default="cifar-compare", help="wandb 项目名")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="wandb 运行名称")
    parser.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练（支持 best/last）")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


def save_history(history: List[Dict], path: Path) -> None:
    dump_json(history, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(device=args.device, prefer_gpu=args.use_gpu)
    print(f"Using device: {device}")

    data_model_type = "vit" if args.model.lower().startswith("vit") else "cnn"
    train_loader, val_loader, test_loader = get_dataloaders(
        model_type=data_model_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        data_dir=args.data_dir,
    )

    model_kwargs: Dict = {}
    model_name = args.model.lower()
    if model_name == "resnet18":
        model_kwargs["pretrained"] = args.pretrained
    if model_name.startswith("vit"):
        model_kwargs["pretrained"] = args.pretrained
        model_kwargs["freeze_backbone"] = args.freeze_backbone or args.pretrained

    model = build_model(args.model, **model_kwargs).to(device)
    n_params = count_parameters(model)
    print(f"Model: {args.model}, trainable params: {n_params:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    run_config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "val_split": args.val_split,
    }
    wandb_run = setup_wandb(args, run_config)

    output_dir = Path(args.output_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")
    log_dir = ensure_dir(output_dir / "logs")

    best_val = 0.0
    start_epoch = 1
    history_path = log_dir / f"{args.model}_history.json"
    history: List[Dict] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                best_val = max(h.get("best_val", 0.0) for h in history)
                start_epoch = history[-1].get("epoch", 0) + 1
        except Exception:
            history = []

    # resume from checkpoint if provided
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        best_val = ckpt.get("best_val", best_val)
        start_epoch = max(start_epoch, ckpt.get("epoch", 0) + 1)
        print(f"Resumed from {args.resume}, start epoch = {start_epoch}, best_val = {best_val:.4f}")

    if start_epoch > args.epochs:
        print(f"Start epoch {start_epoch} > total epochs {args.epochs}, nothing to do.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate_accuracy(model, val_loader, device)
        elapsed = time.time() - start

        is_best = val_acc > best_val
        if is_best:
            best_val = val_acc
            save_path = ckpt_dir / f"{args.model}_best.pth"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "params": n_params,
                    "model": args.model,
                    "best_val": best_val,
                },
                save_path,
            )

        # always save last checkpoint for resume
        last_path = ckpt_dir / f"{args.model}_last.pth"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "epoch": epoch,
                "params": n_params,
                "model": args.model,
                "best_val": best_val,
            },
            last_path,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "best_val": best_val,
            "elapsed_sec": elapsed,
        }
        history.append(epoch_record)

        print(
            f"[{epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"best_val={best_val:.4f} time={elapsed:.1f}s"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "best_val": best_val,
                    "elapsed_sec": elapsed,
                }
            )

    history_path = log_dir / f"{args.model}_history.json"
    save_history(history, history_path)

    # Load best checkpoint for final test evaluation
    best_ckpt = ckpt_dir / f"{args.model}_best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded best checkpoint from epoch {state['epoch']} with val_acc={state['val_acc']:.4f}")

    test_acc = evaluate_accuracy(model, test_loader, device)
    summary = {
        "model": args.model,
        "params": n_params,
        "best_val": best_val,
        "test_acc": test_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    dump_json(summary, log_dir / f"{args.model}_summary.json")
    print(f"Test accuracy: {test_acc:.4f}")
    if wandb_run:
        wandb_run.log({"test_acc": test_acc, "best_val": best_val, "params": n_params})
        wandb_run.finish()


if __name__ == "__main__":
    main()
