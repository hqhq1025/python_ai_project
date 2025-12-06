# 一、项目基本信息
题目：《不同神经网络结构在 CIFAR-10 图像分类任务中的性能对比研究：从 MLP 到 CNN、ResNet 与 ViT》
数据集：CIFAR-10（32x32x3，10类，50K训练 + 10K测试）
工具：PyTorch、torchvision、matplotlib、numpy

# 二、研究动机与问题设置
- 图像分类是计算机视觉的基础任务
- 不同网络结构具有不同的归纳偏置（inductive bias）
- 本项目主要研究问题：
  1. 不利用空间结构的 MLP 与 CNN 在图像任务上差距有多大？
  2. 深度与正则化（BN/Dropout）对 CNN 性能提升效果如何？
  3. 预训练 Vision Transformer 在小数据集上能否达到或超过 CNN？
  4. 结构设计、深度、预训练三者分别对泛化性能的影响是什么？

# 三、数据与预处理方案
- 数据集划分：
  - 训练集：45,000
  - 验证集：5,000（从训练集中切分）
  - 测试集：10,000（官方提供）
- 数据增强（CNN/ResNet）：
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
  - Normalize(mean, std)
- ViT 数据预处理：
  - Resize(224), CenterCrop(224)
  - Normalize标准不同（按预训练权重要求）

# 四、模型设计（四类模型）
1) MLP（反例，忽略图像结构）
   - Flatten → Linear → ReLU → Linear → ReLU → Linear
   - 参数约 ~1-2M

2) SmallCNN（基础卷积模型）
   - Conv → ReLU → MaxPool ×2
   - Flatten + 全连接
   - 参数约 ~1-3M

3) DeepCNN 或 ResNet18（深层次 CNN）
   - 多个卷积块 + BN + Dropout
   - 或直接采用 torchvision.models.resnet18
   - 参数约 ~11M

4) Vision Transformer（预训练 ViT-Tiny）
   - 只训练分类头（heads.head）
   - 参数约 ~21M（但可冷冻绝大部分）

# 五、训练策略与实验设置
- Optimizer：Adam(lr=1e-3)
- Batch size：64
- Loss：CrossEntropyLoss
- Epoch：10（至少 5）
- 设备：GPU 优先（CPU 也可跑）

实验设计：
1. 单模型训练，多次记录 val accuracy
2. 统一记录每 epoch 的 train_loss / val_acc
3. 保存最优模型参数（基于 val_acc）

# 六、评测指标与结果展示
- 指标：
  - Test accuracy
  - 参数量（model.parameters() 统计）
  - 单 epoch 训练耗时（大致估计即可）
- 曲线图：
  - train_loss vs epoch（四条线）
  - val_acc vs epoch（四条线）
- 表格示例：
  模型 | 参数量 | 最佳 Val Acc | Test Acc | 训练时长/epoch
  --------------------------------------------------------
  MLP | xM | xx.x% | xx.x% | xx s
  SmallCNN | xM | xx.x% | xx.x% | xx s
  DeepCNN | xM | xx.x% | xx.x% | xx s
  ViT-Tiny | xM | xx.x% | xx.x% | xx s

# 七、错误样本分析（可选加分项）
- 抽取 5–10 张误分类样本
- 说明不同模型倾向错误的类别（如猫 vs 狗、鹿 vs 马）
- 分析原因：局部纹理 vs 轮廓 vs 语义信息

# 八、结论与分析（必须写得“像研究”）
- MLP 明显劣于 CNN，证明卷积的归纳偏置有效
- DeepCNN/ResNet 明显优于浅层 CNN，深度与正则化提升泛化能力
- 预训练 ViT 接近甚至超过深 CNN，说明预训练在无归纳偏置架构上的关键作用
- 小数据下：CNN 仍高性价比；大模型/预训练才能激活 ViT 的潜力

# 九、心得体会（报告需要有“人话”）
- 训练中遇到的问题（过拟合、收敛慢、调参）
- 数据增强 & BatchNorm的重要性
- 预训练模型对可复现性的帮助
- 理解不同结构的优缺点与应用场景

# 十、扩展方向（建议作为“未来工作”写到报告末尾）
- 替换数据集到 Tiny-ImageNet
- 引入早停策略或学习率调度器
- 对比 Swin Transformer / ConvNeXt 等现代结构
- 进一步对比微调策略（冻结 vs 部分解冻）

# 十一、最终交付（你必须准备的 3 个文件）
1. 代码文件夹（models/ train.py utils.py）
2. 成果展示（训练曲线图、表格、错误样本）
3. 文档（总结报告 + 5分钟演示PPT）