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
├── result/           # 四个学习单元的实验截图
└── 实验记录.md        # 实验条件、结果与分析
```

模型目录按学习单元组织实现代码、配置和运行输出。数据集与训练输出由本地准备，不随仓库源码提交。

## 环境配置与运行

本机使用 Windows + WSL2（Ubuntu 24.04），在 Linux 中通过 Conda 管理 `cv` 环境。脚本使用标准库 `tomllib`，需要 Python 3.11 或更新版本。以下命令均在 WSL/Linux 终端执行。

### 已验证的环境

2026-10-18 从实际训练环境读取的版本如下；这是实验环境记录，不代表其他版本已完成兼容性测试。

| 组件 | 版本 |
|---|---|
| Python | 3.11.17 |
| PyTorch / torchvision | 2.13.0+cu126 / 0.28.0+cu126 |
| PyTorch CUDA 运行时 | 12.6 |
| NumPy / scikit-learn | 2.4.6 / 1.9.1 |
| Matplotlib | 3.11.2 |
| TensorBoard / WandB | 2.21.0 / 0.31.0 |

### 创建环境与安装依赖

先安装 Conda，并在 GPU 训练机器上准备可用的 NVIDIA 驱动。已有 `cv` 环境时只需激活，不必重复创建或升级。

```bash
git clone https://github.com/LaiZheHao/research-isagoge-d2l-learn.git
cd research-isagoge-d2l-learn
conda create -n cv python=3.11 pip -y
conda activate cv

# CUDA 12.6 wheel 的安装示例
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy scikit-learn matplotlib tensorboard wandb
```

上述命令用于新环境安装，没有锁定全部依赖版本；不会保证获得与表中完全相同的环境。其他硬件或 CUDA 平台按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择匹配命令。只使用 CPU 时，将 PyTorch 安装命令中的索引改为 `https://download.pytorch.org/whl/cpu`，训练时指定 `--device cpu`。

仓库当前训练脚本不直接导入 d2l，因此运行这些脚本无需安装 d2l。跟随教材运行 `d2l.train_ch3` 等旧接口时，本机另装了 `d2l==0.17.6`；它不属于当前 CIFAR-10 训练的必需依赖。

### 验证环境

```bash
python - <<'PY'
import torch
import torchvision
import numpy
import sklearn
import matplotlib
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(4, 8, device=device, requires_grad=True)
x.square().mean().backward()
print("Forward/backward passed on", device)
PY
```

GPU 训练前应确认 `CUDA available: True`，且前向/反向检查在 `cuda` 上通过；CPU 检查通过不代表 CUDA 已可用。

### 准备数据与运行 CIFAR-10

```bash
# 首次准备数据；已有完整数据时会检查并复用
python data/code/download_cifar10.py

# 单轮流程检查，不依赖日志服务登录（会实际训练并保存输出）
python cifar10/code/train.py --epochs 1 --logger none

# 默认完整训练，指标写入本地 TensorBoard
python cifar10/code/train.py --logger tensorboard
tensorboard --logdir cifar10/output --port 6006
```

浏览器打开 `http://localhost:6006` 查看 TensorBoard。模型选择、增广、早停、学习率衰减及 WandB 使用方法见 [CIFAR-10 运行说明](cifar10/README.md)。WandB 在线记录需先配置服务地址并完成账户登录；不使用在线服务时选择 `tensorboard` 或 `none` 即可。

训练输出保存在 `cifar10/output/<运行时间>/`，包含配置、逐轮指标、最佳/末轮权重和混淆矩阵；该目录及数据集不纳入 Git。用于交付的截图单独保存在 [result/](result/)，由实验记录引用。

## 实验记录

各学习单元的实验设置、对照结果、观察与结论边界记录在[实验记录](./实验记录.md)中。指标以对应运行的 `metrics.csv` 为准，历史参数以 `run.json` 为准。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)
