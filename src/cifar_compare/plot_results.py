from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_history(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_curves(model_histories: Dict[str, List[Dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, hist in model_histories.items():
        epochs = [e["epoch"] for e in hist]
        losses = [e["train_loss"] for e in hist]
        plt.plot(epochs, losses, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Train Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, hist in model_histories.items():
        epochs = [e["epoch"] for e in hist]
        accs = [e["val_acc"] for e in hist]
        plt.plot(epochs, accs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.title("Val Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_acc.png", dpi=200)
    plt.close()


def write_markdown_table(summaries: Dict[str, Dict], path: Path) -> None:
    headers = ["模型", "参数量", "最佳 Val Acc", "Test Acc", "Epochs", "Batch Size", "LR"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for name, s in summaries.items():
        line = " | ".join(
            [
                name,
                f"{s.get('params', 0):,}",
                f"{s.get('best_val', 0):.4f}",
                f"{s.get('test_acc', 0):.4f}",
                str(s.get("epochs", "")),
                str(s.get("batch_size", "")),
                str(s.get("lr", "")),
            ]
        )
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="汇总并绘制训练结果")
    parser.add_argument("--log-dir", type=str, default="outputs/logs", help="历史与 summary JSON 所在目录")
    parser.add_argument("--figure-dir", type=str, default="outputs/figures", help="输出曲线图目录")
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="指定模型名称过滤（默认读取目录下所有 *_history.json）",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    figure_dir = Path(args.figure_dir)

    histories: Dict[str, List[Dict]] = {}
    summaries: Dict[str, Dict] = {}

    if not log_dir.exists():
        raise FileNotFoundError(f"Log dir not found: {log_dir}")

    history_files = sorted(log_dir.glob("*_history.json"))
    for hf in history_files:
        name = hf.stem.replace("_history", "")
        if args.models and name not in args.models:
            continue
        histories[name] = load_history(hf)
        summary_path = log_dir / f"{name}_summary.json"
        if summary_path.exists():
            summaries[name] = load_summary(summary_path)

    if not histories:
        raise RuntimeError("No histories found. Please run training first.")

    plot_curves(histories, figure_dir)

    if summaries:
        write_markdown_table(summaries, figure_dir / "results_table.md")
        print(f"Saved summary table to {figure_dir / 'results_table.md'}")
    print(f"Saved figures to {figure_dir}")


if __name__ == "__main__":
    main()

