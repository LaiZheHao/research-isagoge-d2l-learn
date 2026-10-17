# CIFAR-10 分类实验

本目录包含 CIFAR-10 分类训练脚本、TOML 配置和实验输出。训练脚本按固定种子从 CIFAR-10 官方训练集分层划分训练集和验证集; 独立测试集只在训练结束后评估。

## 目录结构

```text
cifar10/
├── code/
│   └── train.py
├── config/
│   └── cifar10.toml
└── output/
    └── <每次运行独立的实验目录>/
```

数据集存放在仓库根目录的 `data/cifar-10-batches-py/`。模型、指标和日志保存在 `cifar10/output/` 下，每次运行创建独立目录，不覆盖已有实验。

## 环境与准备数据

在仓库根目录打开终端，激活已有的 `cv` Conda 环境：

```bash
conda activate cv
```

首次运行前下载 CIFAR-10：

```bash
python data/code/download_cifar10.py
```

训练脚本只读取已存在的数据，不会自动下载。如果数据不存在或损坏，脚本会提示先运行下载脚本。

## 开始训练

从仓库根目录运行默认配置:

```bash
python cifar10/code/train.py
```

默认模型是项目中的简单 CNN，默认日志工具是 WandB。可选模型包括该 CNN、ResNet-18、NiN 和 DenseNet-121; 选择其他模型或日志方式：

```bash
# 使用 CIFAR-10 输入尺寸适配过的 ResNet-18
python cifar10/code/train.py --model resnet18

# 使用 Network in Network
python cifar10/code/train.py --model nin

# 使用 CIFAR-10 输入尺寸适配过的 DenseNet-121
python cifar10/code/train.py --model densenet121

# 不启用 WandB、TensorBoard，仅保留本地 CSV
python cifar10/code/train.py --logger none

# 记录到 TensorBoard
python cifar10/code/train.py --logger tensorboard

# WandB 本地离线记录
python cifar10/code/train.py --logger wandb --wandb-mode offline
```

默认 logger 是 WandB; 在线记录需要安装 WandB 并完成账户认证。若只想使用本地 CSV，可指定 `--logger none`，此时不会导入 WandB 或 TensorBoard。选择 `tensorboard` 时需要环境中已安装 TensorBoard。

脚本也可显示所有命令行选项：

```bash
python cifar10/code/train.py --help
```

## 配置和单次运行覆盖

默认配置文件是 `cifar10/config/cifar10.toml`。训练脚本不会回写配置文件; 命令行参数只覆盖当前运行。

主要训练配置：

| TOML 配置项 | 含义 |
|---|---|
| `data_root` | 数据目录，相对于仓库根目录 |
| `output_root` | 实验输出目录，相对于仓库根目录 |
| `model` | 模型名称: `cnn`、`resnet18`、`nin` 或 `densenet121` |
| `batch_size` | 每个训练批次的样本数 |
| `learning_rate` | SGD 学习率 |
| `epochs` | 最大训练轮数; 可能因早停而提前结束 |
| `momentum` | SGD 动量 |
| `weight_decay` | SGD 权重衰减 |
| `step_size`、`gamma` | 每隔 `step_size` 轮将学习率乘以 `gamma` |
| `patience` | 验证损失连续未改善多少轮后提前停止 |
| `min_delta` | 验证损失下降量必须严格超过此值才视为改善 |
| `seed` | 数据划分、模型初始化和数据顺序使用的随机种子 |
| `device` | `auto` 自动选择 CUDA/CPU，也可指定 `cpu` 或 `cuda` |
| `num_workers` | DataLoader 子进程数; `0` 表示主进程加载数据 |
| `num_threads` | PyTorch CPU 运算使用的线程数 |

常用命令行参数：

| 参数 | 用途 |
|---|---|
| `--config PATH` | 指定 TOML 配置文件 |
| `--model cnn` / `--model resnet18` / `--model nin` / `--model densenet121` | 选择模型 |
| `--batch-size N` | 覆盖批量大小 |
| `--learning-rate VALUE` | 覆盖学习率 |
| `--epochs N` | 覆盖最大训练轮数 |
| `--patience N` | 覆盖验证损失早停等待轮数 |
| `--min-delta VALUE` | 覆盖验证损失最小改善阈值 |
| `--step-size N` | 覆盖学习率衰减间隔 |
| `--gamma VALUE` | 覆盖学习率衰减系数，范围为 `(0, 1]` |
| `--device auto` / `--device cpu` / `--device cuda` | 选择运行设备 |
| `--logger wandb/tensorboard/both/none` | 选择指标记录工具 |
| `--wandb-mode online/offline` | 选择 WandB 在线或离线模式 |
| `--wandb-project NAME` | 指定 WandB 项目名称 |

例如，以较小批量在 CPU 上训练 2 轮且不启用外部记录器：

```bash
python cifar10/code/train.py --batch-size 64 --epochs 2 --device cpu --logger none
```

也可以在单次运行中覆盖早停和学习率调度参数，而不修改 TOML：

```bash
python cifar10/code/train.py --patience 8 --min-delta 0.001 --step-size 8 --gamma 0.7 --logger none
```

`min_delta` 是相对于上次保存权重时的验证损失，绝对下降量必须严格超过的阈值，数值越小越容易把微小改善计作进步; `gamma` 是衰减时学习率的乘数，越接近 1 衰减越温和。例如 `gamma=0.5` 将学习率减半，`gamma=0.8` 则每次保留原学习率的 80%。这些取值是可调的实验超参数，建议结合 `metrics.csv` 中验证损失和学习率变化做对照实验。

配置中的数据和输出路径以仓库根目录为基准，不依赖启动脚本时的当前工作目录。

## 图像预处理与增广

配置文件中的 `[augmentation]` 部分控制训练集的随机增广：

```toml
[augmentation]
enabled = true
random_crop = true
crop_padding = 4
horizontal_flip = true
horizontal_flip_probability = 0.5
```

| 配置项 | 含义 |
|---|---|
| `enabled` | 是否启用随机图像增广。关闭时仍会执行张量转换和标准化 |
| `random_crop` | 是否随机裁剪训练图片 |
| `crop_padding` | 裁剪前在图片四周补边的像素数 |
| `horizontal_flip` | 是否随机水平翻转训练图片 |
| `horizontal_flip_probability` | 每张训练图片进行水平翻转的概率，范围为 0 到 1 |

验证集和测试集不做随机增广。所有数据都会先转换为张量，再使用代码中的 CIFAR-10 RGB 通道均值和标准差进行标准化。

命令行可以临时覆盖增广设置：

```bash
# 单次运行中禁用所有随机增广
python cifar10/code/train.py --no-augmentation --logger none

# 保留水平翻转，单独关闭随机裁剪
python cifar10/code/train.py --no-random-crop --logger none

# 临时调整裁剪补边和水平翻转概率
python cifar10/code/train.py --crop-padding 6 --horizontal-flip-probability 0.25 --logger none
```

`--augmentation`/`--no-augmentation` 控制增广总开关; `--random-crop`/`--no-random-crop` 和 `--horizontal-flip`/`--no-horizontal-flip` 可分别控制单项变换。命令行覆盖不会保存回 TOML。

## 数据划分、训练和评估

- 使用 CIFAR-10 的 50000 张训练图片，通过类别分层抽样划分为 45000 张训练图片和 5000 张验证图片。
- 数据划分、模型初始化和训练数据顺序使用配置中的随机种子（默认 `42`）。
- 训练集用于更新模型; 验证集用于计算每轮验证指标和选择最佳权重。
- 测试集的 10000 张图片不参与训练和模型选择，只在训练结束后用按 `min_delta` 改善标准保存的 `best.pt` 评估一次。该权重不一定对应所有轮次中绝对最低的验证损失。
- 优化器使用 SGD; `StepLR` 按配置调整学习率; 若验证损失连续若干轮未改善，会提前停止训练。
- 当前可选模型为 `cnn`、`resnet18`、`nin` 和 `densenet121`。所有模型均针对 CIFAR-10 输出 10 类分数，不加载预训练权重。`cnn` 是本项目的简单三层卷积网络，并非经典 LeNet; ResNet-18 和 DenseNet-121 的入口层已调整为适合 32×32 图片。

固定随机种子有助于重现实验，但不能保证不同硬件、软件版本和所有运算环境下的结果完全一致。

## 输出文件

每次运行在 `cifar10/output/` 下创建一个新的实验目录，目录名以启动时间为基础。同一分钟内再次运行会追加序号。

| 文件或目录 | 内容 |
|---|---|
| `run.json` | 本次有效配置、数据规模、类别、设备和运行元数据 |
| `split.npz` | 训练集和验证集对应的样本索引 |
| `metrics.csv` | 每轮训练/验证损失、准确率、学习率和耗时 |
| `best.pt` | 验证损失按 `min_delta` 标准改善时保存的最佳模型权重 |
| `model.pt` | 最后一轮训练结束时的模型权重 |
| `test_metrics.json` | 最佳模型在测试集上的损失、准确率等指标 |
| `confusion_matrix.csv` | 完整测试集的原始计数，行是真实类别、列是预测类别，含类别名称 |
| `confusion_matrix.png` | 原始计数热力图，每格标注样本数 |
| `confusion_matrix_normalized.png` | 按真实类别归一化的百分比热力图; 无样本的类别显示 N/A |
| `predictions.json` | 测试集首批最多 10 张图片的真实标签、预测类别和类别概率 |
| `tensorboard/` | 启用 TensorBoard 时生成的事件文件 |

`best.pt` 和 `model.pt` 保存的是模型 `state_dict`，不包含优化器状态，因此不是完整的断点续训文件。训练和测试指标始终写入本地文件; WandB 或 TensorBoard 是可选的额外记录方式。

混淆矩阵与最终测试准确率在同一次评估中计算，均使用 `best.pt`。对角线计数之和除以总计数等于测试准确率。生成 PNG 需要环境中安装 Matplotlib; 图像和 CSV 保存在本次 output 目录，不自动上传至 WandB 或 TensorBoard。
