# 项目执行计划（plan.md）

## 0. 项目概览

- 题目：不同神经网络结构在 CIFAR-10 图像分类任务中的性能对比研究：从 MLP 到 CNN、ResNet 与 ViT
- 数据集：CIFAR-10（32×32×3，10 类，50K 训练 / 10K 测试）
- 框架与工具：PyTorch、torchvision、matplotlib、numpy
- 代码结构目标：使用 `src/` 布局，核心包名暂定为 `src/cifar_compare/`

本计划围绕「先跑通完整代码与实验，再逐步丰富分析与文档」这一原则展开。

## 1. 本轮计划目标与范围

**目标（这一版完成即视为 plan 完成）：**

- 在 CIFAR-10 上实现并训练以下四类模型，并得到可复现的实验结果：
  - MLP（不利用空间结构的反例）
  - SmallCNN（基础卷积网络）
  - DeepCNN 或 ResNet18（深层 CNN）
  - 预训练 ViT（Frozen backbone + 只训练分类头）
- 统一的训练与评估脚本，能够：
  - 保存每个 epoch 的 train_loss / val_acc
  - 保存基于 val_acc 的最优模型参数
  - 输出测试集准确率与参数量
- 至少生成：
  - train_loss 与 val_acc 的曲线图（四条线）
  - 汇总比较表（各模型参数量 / Val Acc / Test Acc / 训练时长估计）

**本轮不强制完成（可做初版框架）：**

- 误分类样本的详细分析文本
- 完整报告成稿与 PPT（可以先预留结构）

## 2. 阶段划分（里程碑）

为便于执行，本轮计划拆为五个阶段：

- 阶段 A：项目骨架与环境准备
- 阶段 B：数据管线与预处理
- 阶段 C：模型实现（四类结构）
- 阶段 D：训练脚本与实验管理
- 阶段 E：结果整理与初步分析

后续执行时优先保证 A→D 跑通，E 可在实验结果出来后逐步补充。

## 3. 阶段 A：项目骨架与环境准备

**目标：** 建立基础工程结构和环境，使得后续开发都在统一框架下进行。

- [MUST] 创建 `src/cifar_compare/` 目录及基础模块结构：
  - `src/cifar_compare/__init__.py`
  - `src/cifar_compare/models/`
  - `src/cifar_compare/data/`
  - `src/cifar_compare/train.py`
  - `src/cifar_compare/utils.py`
- [MUST] 在根目录添加或更新必要的配置文件（如 `pyproject.toml` 或 `setup.cfg`，若尚不存在，可以先简化，只要支持 `python -m pip install -e .[dev]` 即可）。
- [SHOULD] 在 `tests/` 下预留测试文件结构，对应 `src/` 布局（例如 `tests/test_models.py`、`tests/test_data.py`）。
- [COULD] 编写一个最小可运行示例（例如在 `train.py` 中打印环境与设备信息），验证包导入无误。

## 4. 阶段 B：数据管线与预处理

**目标：** 实现 CIFAR-10 的加载、划分与预处理，为四类模型统一提供数据接口。

- [MUST] 在 `src/cifar_compare/data/datasets.py` 中实现：
  - 加载 CIFAR-10 数据集（train/test）
  - 从训练集中划分出 45,000/5,000 的 train/val
- [MUST] 定义 CNN/ResNet 使用的数据增强：
  - `RandomCrop(32, padding=4)`
  - `RandomHorizontalFlip()`
  - `Normalize(mean, std)`（使用 CIFAR-10 标准 mean/std）
- [MUST] 定义 ViT 使用的预处理：
  - `Resize(224), CenterCrop(224)`
  - 使用与预训练权重匹配的 Normalize 配置
- [SHOULD] 提供统一的数据加载接口函数（例如 `get_dataloaders(config)`），支持 batch_size、num_workers 等参数配置。
- [COULD] 支持通过命令行或配置文件选择是否启用数据增强、是否打乱数据等。

## 5. 阶段 C：模型实现（MLP / SmallCNN / DeepCNN-ResNet / ViT）

**目标：** 在统一接口下实现四类模型，便于在训练脚本中按名称切换。

- [MUST] 在 `src/cifar_compare/models/mlp.py` 中实现 `MLPClassifier`：
  - 输入：展平后的 `32×32×3` 图像
  - 结构：`Flatten → Linear → ReLU → Linear → ReLU → Linear`，参数量约 1–2M
- [MUST] 在 `src/cifar_compare/models/cnn_small.py` 中实现 `SmallCNN`：
  - 至少包含两级 `Conv → ReLU → MaxPool`
  - 最终通过 `Flatten + 全连接` 输出 10 类
- [MUST] 在 `src/cifar_compare/models/resnet.py` 中封装 DeepCNN / ResNet18：
  - 推荐直接使用 `torchvision.models.resnet18`，修改首层/最后一层以适配 CIFAR-10
  - 支持加载 ImageNet 预训练权重或从头训练（通过参数控制）
- [MUST] 在 `src/cifar_compare/models/vit.py` 中封装 ViT：
  - 采用预训练 ViT-Tiny（如 `torchvision` 或 `timm`，具体实现按可用依赖选择）
  - 默认冻结 backbone，仅训练分类头（如 `heads.head` 或对应层）
- [SHOULD] 提供统一的模型构建接口：
  - 在 `src/cifar_compare/models/__init__.py` 中实现 `build_model(name: str, **kwargs)`，`name` 可为 `"mlp"`, `"small_cnn"`, `"resnet18"`, `"vit_tiny"` 等。
- [COULD] 在各模型中增加获取参数量的工具函数或统一在 `utils.py` 中实现。

## 6. 阶段 D：训练脚本与实验管理

**目标：** 实现统一的训练/验证/测试流程，记录指标并保存最佳模型。

- [MUST] 在 `src/cifar_compare/train.py` 中实现主训练入口：
  - 解析命令行参数或配置（模型名称、batch_size、lr、epoch 数、是否使用 GPU 等）
  - 构建数据加载器与模型
  - 选择优化器：`Adam(lr=1e-3)`
  - 损失函数：`CrossEntropyLoss`
  - 训练 epoch 数：默认 10（不少于 5）
- [MUST] 实现统一训练循环：
  - 每个 epoch 记录：`train_loss`（平均）和 `val_acc`
  - 在验证集上评估，并根据 `val_acc` 维护当前最优模型
  - 保存最优模型参数到指定路径（例如 `outputs/checkpoints/{model_name}_best.pth`）
- [MUST] 在训练过程中打印关键信息：
  - 当前 epoch / 总 epoch
  - train_loss、val_acc
  - 估计所需时间（可以大致用 epoch 耗时估计）
- [SHOULD] 在 `utils.py` 中实现：
  - 计算模型参数量的函数（`count_parameters(model)`）
  - 保存/加载 checkpoint 的通用函数
- [SHOULD] 支持对四类模型分别运行训练脚本，并将结果（如日志文件）保存到 `outputs/` 指定目录，便于后续统一读取绘图。
- [COULD] 引入简单的配置文件（如 `yaml/json`）来管理不同实验设置。

## 7. 阶段 E：结果整理与初步分析

**目标：** 将训练得到的日志与模型结果整理成图表和初步结论。

- [MUST] 在 `src/cifar_compare/utils.py` 或单独文件（如 `src/cifar_compare/plot_results.py`）中实现：
  - 从日志中读取四个模型的 `train_loss` 与 `val_acc`
  - 生成两张曲线图：
    - train_loss vs epoch（四条线）
    - val_acc vs epoch（四条线）
  - 将图保存到 `outputs/figures/`
- [MUST] 整理一张结果汇总表（可用 `csv` 或 `markdown`），包含：
  - 模型名称
  - 参数量
  - 最佳 Val Acc
  - Test Acc
  - 单 epoch 训练耗时（可近似估计）
- [SHOULD] 在根目录创建 `report_outline.md`，初步写出报告结构：
  - 动机与问题设置
  - 数据与方法
  - 实验结果
  - 错误样本分析（预留位置）
  - 结论与心得
- [COULD] 实现一个简单脚本从结果表中自动生成 markdown 表格片段，便于直接复制到报告中。

## 8. 误分类样本分析与扩展（后续可选阶段）

这一部分作为扩展/加分内容，可以在主实验跑通后按需追加。

- [COULD] 编写脚本：
  - 从测试集中收集若干误分类样本（例如每个模型 5–10 张）
  - 可视化这些样本，标出真实标签与预测标签
- [COULD] 对比不同模型在相同样本上的预测差异（例如猫 vs 狗、鹿 vs 马），为报告撰写提供素材。
- [COULD] 实验扩展方向：
  - 更换到 Tiny-ImageNet 或增加图像分辨率
  - 引入早停（Early Stopping）或学习率调度器（如 StepLR、CosineAnnealingLR）
  - 对比 Swin Transformer / ConvNeXt 等结构
  - 对 ViT 不同微调策略（冻结/部分解冻/全量微调）进行比较

## 9. 执行顺序建议

为了在有限时间内得到稳定的对比结果，建议按照以下优先顺序执行：

1. 完成阶段 A 与 B（工程骨架 + 数据管线）
2. 完成阶段 C 中的 MLP 与 SmallCNN 实现与训练，先拿到最基本对比
3. 完成 ResNet18 的集成与训练
4. 集成与验证 ViT 模型（如遇依赖问题，可先跳过，等主结果固定后再解决）
5. 完成阶段 D 的统一训练脚本与实验管理，确保四个模型都可一键训练与评估
6. 执行阶段 E，生成曲线和结果表，整理初步分析

## 10. 交付物对齐

结合 `draft.md` 中的要求，本轮计划完成后，预期能较顺畅地得到以下三类最终交付：

1. **代码文件夹**
   - `src/cifar_compare/` 下的模型实现、训练脚本与辅助工具
   - 可选的 `tests/` 单元测试
2. **成果展示**
   - 存放在 `outputs/` 下的训练曲线图、结果汇总表、（可选）错误样本可视化
3. **文档雏形**
   - `report_outline.md` 作为报告草稿框架
   - 后续可基于该框架扩展出完整报告与 5 分钟演示 PPT

---

后续执行过程中，如有新想法或需求变化，可以在不打乱大结构的前提下，对本 `plan.md` 中各阶段的细节任务进行增删与标记（例如标注已完成、延期等），但建议保持阶段划分和优先级标记不变，以方便跟踪进度。

