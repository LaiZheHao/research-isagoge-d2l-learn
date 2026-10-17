# d2l-learn

依据《动手学深度学习》（Dive into Deep Learning，d2l）逐步学习和复现关键概念与代码。重点理解数学含义、Python 语法、张量形状及实际运行结果，每次推进一个学习单元。

## 当前进度

- 已完成 Fashion-MNIST 下载脚本及本地数据读取。
- 已实现 Fashion-MNIST 十分类 softmax 回归，对应中文教材 3.7 节的简洁实现。
- 支持 CSV 指标保存、TensorBoard 和 WandB 实验记录、模型参数与预测样例保存。
- 已实现多层感知机训练脚本，支持激活函数、参数初始化选择和可配置训练参数。
- 已实现 d2l 4.4 多项式拟合，数据生成与训练分离，支持模型阶数选择及 loss 记录。
- 卷积网络目录已预留，具体实现随学习逐步增加。

CIFAR-10 学习代码已整理到 `cifar10/`，当前支持简单 CNN、ResNet-18、NiN 和 DenseNet-121。详细使用说明见 [cifar10/README.md](cifar10/README.md)。旧 week1 GitHub 仓库已由用户确认删除；E 盘 `.codex` 中的 baseline 保留为历史参考。

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
├── 实验记录.md
├── .gitignore
├── data/
│   ├── code/
│   │   ├── download_fashion_mnist.py
│   │   └── generate_polynomial.py
│   ├── FashionMNIST/                 # 本地数据，不提交
│   └── polynomial/dataset.pt         # 合成数据，不提交
├── linear-softmax/
│   ├── code/train.py
│   ├── config/softmax.toml
│   └── output/                       # 运行结果，不提交
├── mlp/
│   ├── code/train.py
│   ├── config/mlp.toml
│   └── output/                       # 运行结果，不提交
├── fit/
│   ├── code/train.py
│   ├── config/polynomial.toml
│   └── output/                       # 运行结果，不提交
└── conv/
    ├── basics/
    ├── lenet/
    ├── alexnet/
    ├── vgg/
    ├── nin/
    ├── googlenet/
    └── resnet/
```

`conv/` 下每个目录同样包含 `code/`、`config/`、`output/`。`data/code/` 放数据准备代码，数据集在 `data/` 下共享；各模型目录分别保存实现、配置和结果。默认只维护根 README；用户明确要求时可在子目录另建 README。Git 不跟踪空目录，因此新克隆的仓库可能没有尚未添加文件的预留目录。

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

# 覆盖学习率和批量大小
python linear-softmax/code/train.py --learning-rate 0.03 --batch-size 64 --logger none
```

支持 `--config` 指定其他 TOML 文件；`--epochs`、`--learning-rate`、`--batch-size` 和 `--device` 可覆盖对应配置。命令行参数只影响本次运行，不会修改 TOML 文件。`--config` 的相对路径按终端当前目录解释；TOML 内的数据与输出相对路径按仓库根目录解释。

`--wandb-project` 默认是 `d2l-learn`；`--wandb-mode` 可选 `online` 或 `offline`，仅在启用 WandB 时生效。

## 训练多层感知机

MLP 配置位于 `mlp/config/mlp.toml`。默认隐藏层为 `[256]`，激活函数为 ReLU，初始化为标准差 0.01 的正态分布；训练参数默认 batch size 256、learning rate 0.3、20 epochs。可在命令行或 TOML 配置中调整训练参数、隐藏层宽度、激活函数和初始化方法。脚本默认从 `mlp/config/mlp.toml` 读取配置，数据集仍使用仓库根目录下的 `data/`。

```bash
# 默认配置，WandB 在线记录
python mlp/code/train.py

# 指定激活函数
python mlp/code/train.py --activation relu
python mlp/code/train.py --activation sigmoid
python mlp/code/train.py --activation tanh
python mlp/code/train.py --activation leaky_relu

# 设置隐藏层结构和 Xavier uniform 初始化
python mlp/code/train.py --hidden-sizes 256 256 256 256 256 --initialization xavier

# 覆盖训练参数
python mlp/code/train.py --batch-size 128 --learning-rate 0.03 --epochs 50
```

MLP 每次运行会在 `mlp/output/` 下保存指标、运行配置、模型参数和预测样例。实验参数与运行结果摘要见根目录的[实验记录](./实验记录.md)。

## 训练多项式回归

先在仓库根目录生成数据，再训练：

```bash
python data/code/generate_polynomial.py
python fit/code/train.py --i 3

# 指定阶数和主要超参数
python fit/code/train.py --i 1 --batch-size 10 --lr 0.01 --epochs 100 --logger none
python fit/code/train.py --i 19 --epochs 1500 --logger both

# 仅使用 TensorBoard
python fit/code/train.py --i 3 --logger tensorboard
tensorboard --logdir fit/output --port 6006
```

配置位于 `fit/config/polynomial.toml`，默认 i=3、batch size=10、lr=0.01、100 epochs。`--i` 是最高次数，允许非负整数，0 表示常数模型，特征数为 i+1。`--lr` 也可写作 `--learning-rate`。固定使用 CPU 和 seed=42。

生成脚本将训练集和测试集各 100 个样本保存到 `data/polynomial/dataset.pt`，每次运行会覆盖该文件。训练脚本只读取数据，缺失时提示先生成。真实规律为带标准差 0.1 噪声的三次多项式；模型使用阶乘缩放特征和无额外偏置的线性层，以 SGD 最小化 MSE。

`--logger` 支持 wandb（默认）、tensorboard、both、none。fit 未提供分类脚本的 `--config`、`--device`、`--wandb-mode` 或 `--wandb-project` 参数；WandB 项目固定为 `d2l-learn`。需要离线记录时，可在 WSL 使用：

```bash
WANDB_MODE=offline python fit/code/train.py --i 3
```

输出目录为 `fit/output/时间-i阶数/`，重名时追加序号。始终保存 `run.json`、`metrics.csv`、最终 `model.pt`，按需生成日志目录；不保存分类预测 JSON。CSV 和日志记录每轮更新后评估的训练/测试 MSE，没有准确率或耗时指标。

## 查看实验

TensorBoard 需要本次或以往运行启用了 `tensorboard` 或 `both`。在 WSL 终端执行并保持服务运行：

```bash
tensorboard --logdir linear-softmax/output --port 6006
```

打开 [http://localhost:6006](http://localhost:6006)，在 Scalars 中选择运行记录，横轴 Step 对应 epoch。

WandB 在线运行时，终端会输出本次实验的网页链接；点击即可查看。离线运行不会立即更新网页，可在联网后使用终端给出的 `wandb sync` 命令同步。

对于 softmax 和 MLP，两个工具均记录训练/测试准确率、训练/测试损失、单轮和累计训练耗时。训练准确率是训练过程中各批量的累计统计，测试准确率是在每轮更新结束后计算的；耗时包含训练时的数据加载和计算，不包含测试评估、日志同步与结果保存。

## 输出与版本管理

每次运行创建独立目录，例如 `linear-softmax/output/20261017-1430/`。同一分钟重复运行时追加 `-1`、`-2` 等序号，避免覆盖。

以下表格描述 softmax 和 MLP 的输出；fit 的输出见上节。

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

## CIFAR-10

`cifar10/code/train.py` 与 `cifar10/config/cifar10.toml` 是历史 baseline 的简化适配，结果保存在 `cifar10/output/`。历史 baseline 不变，不复制其数据或实验成绩。

```bash
# 本地没有数据时先准备数据
python data/code/download_cifar10.py
# 以下命令会启动训练
python cifar10/code/train.py --model cnn
python cifar10/code/train.py --model resnet18 --logger tensorboard
python cifar10/code/train.py --model nin --logger none
python cifar10/code/train.py --model densenet121 --logger none
```

支持 `--config`、`--epochs`、`--batch-size`、`--learning-rate`、`--device`，以及与 MLP 相同的 logger、WandB mode/project 参数。使用现有 cv 环境，分层划分依赖 scikit-learn。

训练仅读取本地 `data/cifar-10-batches-py/`。固定种子分层划分 45000/5000，训练使用随机裁剪、翻转和标准化，验证/测试仅标准化。保留 SGD、StepLR 和按验证损失早停; 每轮记录训练/验证指标，结束后用选中的模型评估一次测试集。

始终保存 `run.json`、`split.npz`、`metrics.csv`、`model.pt` (最后一轮)、`best.pt` (验证损失按 min_delta 改善时保存)、`test_metrics.json`、`predictions.json` (最佳模型的样例)。权重均为 state_dict，不提供断点续训; 最终测试同时保存混淆矩阵原始计数 CSV、计数热力图和按真实类别归一化的百分比热力图，不生成历史 baseline 的其他 PNG 图表。使用 cuDNN 确定性设置，但未启用全算子严格确定性检查，不声称与历史 baseline 逐位复现。

本次完成代码适配，借用 baseline 数据做只读加载、划分与前向检查，尚未训练新模型。仓库 data 下的 CIFAR-10 文件尚未准备完整，正式训练前先运行下载脚本。

## 参考资料

- [《动手学深度学习》中文版](https://zh.d2l.ai/)
- [PyTorch 官方教程](https://docs.pytorch.org/tutorials/)
