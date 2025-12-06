# 报告大纲（初稿）

## 1. 研究动机与问题设置
- 图像分类的基础性
- 不同结构的归纳偏置：MLP vs CNN vs ResNet vs ViT
- 本项目的对比问题与假设

## 2. 数据与预处理
- CIFAR-10 划分（train/val/test）
- CNN/ResNet 增强：RandomCrop, Flip, Normalize
- ViT 预处理：Resize/CenterCrop/Normalize（ImageNet 统计）

## 3. 模型与训练设置
- MLP / SmallCNN / ResNet18 / ViT（预训练 + 冻结策略）
- 优化器、学习率、批大小、epoch 设置
- 训练/验证/测试流程与最优模型保存

## 4. 实验结果
- 训练曲线（train_loss / val_acc）
- 汇总表（参数量 / 最佳 Val Acc / Test Acc / 训练时长估计）
- 误分类样本示例（占位，待后续补充）

## 5. 分析与讨论
- 归纳偏置：MLP 明显劣于 CNN
- 深度与正则化：ResNet 相比浅层 CNN 的收益
- 预训练 ViT 的表现及小数据集下的性价比

## 6. 结论与未来工作
- 关键结论梳理
- 未来方向：更大数据集、LR 调度/早停、Swin/ConvNeXt、微调策略对比

## 7. 心得体会
- 训练中遇到的问题与解决
- 数据增强与 BN 的体会
- 预训练模型的可复现性
