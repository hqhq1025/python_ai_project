# 常用指令清单（本地/服务器通用）

## 1. 环境准备
- 创建虚拟环境（可选）：`python -m venv .venv && source .venv/bin/activate`
- 升级 pip：`python -m pip install --upgrade pip`
- 安装依赖（含 dev）：`python -m pip install -e '.[dev]'`
- 如需 wandb：`python -m pip install -e '.[logging]'`（或单独 `pip install wandb`）
- 如需特定 CUDA 版 PyTorch，按官网命令替换安装行（例如 `--index-url https://download.pytorch.org/whl/cu118`）

## 2. 快速检查
- 验证包可用：`python -c "import cifar_compare; print(cifar_compare.__version__)"`

## 3. 运行训练（示例）
- 小型 CNN smoke test：`python -m cifar_compare.train --model small_cnn --epochs 1 --batch-size 64 --device auto`
- 正式跑 4 模型示例（每个建议至少跑 30 轮，统一 wandb）：
  - `python -m cifar_compare.train --model mlp --epochs 30 --batch-size 64 --device auto --wandb`
  - `python -m cifar_compare.train --model small_cnn --epochs 30 --batch-size 64 --device auto --wandb`
  - `python -m cifar_compare.train --model resnet18 --epochs 30 --batch-size 64 --device auto --pretrained --wandb`
  - `python -m cifar_compare.train --model vit_b16 --epochs 30 --batch-size 64 --device auto --pretrained --freeze-backbone --wandb`
- 指定数据目录（可选）：加 `--data-dir /path/to/datasets`
- 设备说明：`--device auto|cpu|cuda|mps`，在 Mac M3 上建议 `--device mps`
- 如果希望 30+ 轮自动提前结束，可加 `--early-stop-patience 5`

## 4. 结果汇总与绘图
- 生成曲线与表格：`python -m cifar_compare.plot_results --log-dir outputs/logs --figure-dir outputs/figures`
- 输出：
  - 曲线图：`outputs/figures/train_loss.png`、`outputs/figures/val_acc.png`
  - 汇总表：`outputs/figures/results_table.md`

## 5. 常见调优
- 降低显存/内存：`--batch-size` 调小
- 数据加载线程：`--num-workers` 调整（CPU/MPS 环境适当降低）
- 训练轮数：`--epochs` 调整（快速测试 1–3，正式 10+）
- 断点续跑：使用 `--resume outputs/checkpoints/<model>_last.pth`（每轮都会保存 last），best 模型在 `<model>_best.pth`
- 批量续跑命令示例（先跑 15/20 轮后续跑到 30）：
  ```bash
  python -m cifar_compare.train --model mlp --epochs 30 --batch-size 64 --device auto --resume outputs/checkpoints/mlp_last.pth --wandb
  ```
