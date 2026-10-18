# d2l-learn

本仓库用于配合《动手学深度学习》（Dive into Deep Learning，d2l）逐步学习和实现深度学习模型，关注教材概念、模型实现与实际实验结果。

## 学习内容

- **Fashion-MNIST**：softmax 回归与多层感知机。
- **多项式拟合**：模型阶数、欠拟合与过拟合，以及 L2 正则化和特征 Dropout 对照。
- **CIFAR-10**：CNN、NiN、ResNet-18 和 DenseNet-121 等卷积网络实验。

## 仓库结构

```text
d2l-learn/
├── data/             # 数据准备脚本与本地数据
├── linear-softmax/   # Softmax 回归
├── mlp/              # 多层感知机
├── fit/              # 多项式拟合
├── cifar10/          # CIFAR-10 图像分类
├── conv/             # 卷积网络学习内容（预留）
└── 实验记录.md        # 实验条件、结果与分析
```

模型目录按学习单元组织实现代码、配置和运行输出。数据集与训练输出由本地准备，不随仓库源码提交。

## 实验记录

各学习单元的实验设置、对照结果、观察与结论边界记录在[实验记录](./实验记录.md)中。指标以对应运行的 `metrics.csv` 为准，历史参数以 `run.json` 为准。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)
