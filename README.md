# d2l-learn

依据《动手学深度学习》（Dive into Deep Learning，d2l）逐步学习和复现关键概念与代码。重点理解数学含义、Python 语法、张量形状及实际运行结果，每次推进一个学习单元。

## 当前进度

- 已完成 Fashion-MNIST 下载脚本及本地数据读取。
- 已实现 Fashion-MNIST 十分类 softmax 回归，对应中文教材 3.7 节的简洁实现。
- 支持 CSV 指标保存、TensorBoard 和 WandB 实验记录、模型参数与预测样例保存。
- 多层感知机和卷积网络目录已预留，具体实现随学习逐步增加。

本仓库独立于 CIFAR-10 项目；不自动复制其 baseline 或延续其阶段迁移计划。

## 本机环境

本机使用 WSL Ubuntu-24.04 和已有的 Conda 环境 `cv`（原名 `cv-week1`）：

```bash
conda activate cv
cd /home/hao/research/00-getting-start/d2l-learn
```

训练代码依赖 PyTorch、torchvision；配置解析使用 Python 3.11 起提供的标准库 `tomllib`。启用哪种日志工具，就需要安装对应的 `tensorboard` 或 `wandb`；`--logger none` 不加载这两个工具。

教材辅助库使用环境中安装的 d2l 包，不在仓库内维护 `d2l/` 目录。当前 softmax 脚本直接使用 PyTorch，并展开训练循环以便阅读，无需调用 `d2l.train_ch3`。本 README 不是完整的环境锁定文件，不要求重新安装或降级已有依赖。

## 目录组织

```text
d2l-learn/
├── README.md
├── .gitignore
├── data/
│   ├── code/download_fashion_mnist.py
│   └── FashionMNIST/                 # 本地数据，不提交
├── linear-softmax/
│   ├── code/train.py
│   ├── config/softmax.toml
│   └── output/                       # 运行结果，不提交
├── multilayer-perceptrons/
│   ├── code/
│   ├── config/
│   └── output/
└── conv/
    ├── basics/
    ├── lenet/
    ├── alexnet/
    ├── vgg/
    ├── nin/
    ├── googlenet/
    └── resnet/
```

`conv/` 下每个目录同样包含 `code/`、`config/`、`output/`。`data/code/` 放数据准备代码，数据集在 `data/` 下共享；各模型目录分别保存实现、配置和结果。只维护根 README。Git 不跟踪空目录，因此新克隆的仓库可能没有尚未添加文件的预留目录。

## 准备数据

首次使用时，在仓库根目录执行：

```bash
python data/code/download_fashion_mnist.py
```

下载脚本使用 `root=data/`，torchvision 将文件存放到 `data/FashionMNIST/`。训练脚本使用相同的数据根目录并设置 `download=False`，不会自动下载。

## 训练 softmax 回归

模型为 `Flatten → Linear(784, 10)`，输入形状为 `[B, 1, 28, 28]`，输出 logits 为 `[B, 10]`。使用交叉熵损失和 SGD；损失计算前不额外应用 softmax。

默认参数在 `linear-softmax/config/softmax.toml` 中：批量大小 256、学习率 0.1、10 轮、随机种子 42，设备 `auto` 优先使用 CUDA。

```bash
# 默认使用 WandB 在线记录；首次使用需先完成登录
wandb login
python linear-softmax/code/train.py

# 仅记录 TensorBoard
python linear-softmax/code/train.py --logger tensorboard

# 同时记录两个工具
python linear-softmax/code/train.py --logger both

# 不启用两个工具，仍保存 CSV、模型、配置和预测结果
python linear-softmax/code/train.py --logger none

# WandB 离线记录
python linear-softmax/code/train.py --logger wandb --wandb-mode offline

# 单轮检查，使用 CPU，不启用日志工具
python linear-softmax/code/train.py --epochs 1 --device cpu --logger none
```

支持 `--config` 指定其他 TOML 文件、`--epochs` 和 `--device` 覆盖对应配置。`--config` 的相对路径按终端当前目录解释；TOML 内的数据与输出相对路径按仓库根目录解释。

`--wandb-project` 默认是 `d2l-learn`；`--wandb-mode` 可选 `online` 或 `offline`，仅在启用 WandB 时生效。

## 查看实验

TensorBoard 需要本次或以往运行启用了 `tensorboard` 或 `both`。在 WSL 终端执行并保持服务运行：

```bash
tensorboard --logdir linear-softmax/output --port 6006
```

打开 [http://localhost:6006](http://localhost:6006)，在 Scalars 中选择运行记录，横轴 Step 对应 epoch。

WandB 在线运行时，终端会输出本次实验的网页链接；点击即可查看。离线运行不会立即更新网页，可在联网后使用终端给出的 `wandb sync` 命令同步。

两个工具均记录训练/测试准确率、训练/测试损失、单轮和累计训练耗时。训练准确率是训练过程中各批量的累计统计，测试准确率是在每轮更新结束后计算的；耗时包含训练时的数据加载和计算，不包含测试评估、日志同步与结果保存。

## 输出与版本管理

每次运行创建独立目录，例如 `linear-softmax/output/20261017-1430/`。同一分钟重复运行时追加 `-1`、`-2` 等序号，避免覆盖。

| 文件或目录 | 内容 |
|---|---|
| `metrics.csv` | 每轮训练/测试损失、准确率、累计训练耗时 `time` 和本轮耗时 `delta_time`，单位为秒 |
| `run.json` | 本次有效配置、设备、PyTorch 版本、数据数量、类别及日志选项 |
| `model.pt` | 最后一轮模型的 `state_dict`；不包含优化器状态，不是完整续训检查点 |
| `predictions.json` | 测试集首批中最多 10 个样本的标签、预测类别和概率 |
| `tensorboard/` | 仅启用 TensorBoard 时生成的事件日志 |
| `wandb/` | 仅启用 WandB 时生成的本地记录 |

当前版本不再生成 `accuracy.png`，也不提供 `--plot-only`。模型权重和预测 JSON 不会由当前代码自动上传到 WandB。

`.gitignore` 已忽略 `.vscode/`、Python 缓存、指定的数据集路径和各级 `output/`。提交时检查变更，仅提交源码、配置和必要文档，不使用强制暂存上传实验日志或数据。日志工具不会替代本地模型与结果备份。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [3.7 softmax 回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html)
- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)
