"""CIFAR-10 分类: 将历史 baseline 整理为独立学习脚本。

主要参数:
模型:       --model cnn/resnet18/nin/densenet121、
常见超参数: --batch-size、--learning-rate、--epochs、
增广:       --augmentation/--no-augmentation、--random-crop/--no-random-crop、
            --horizontal-flip/--no-horizontal-flip。
训练控制:   --patience 早停等待轮数、--min-delta 最小验证损失改善值、
            --step-size 学习率衰减间隔、--gamma 学习率衰减系数。
检查点:     best.pt 保存验证损失改善后的最佳权重; model.pt 保存最后一轮权重。
记录:       --logger wandb/tensorboard/both/none。

配置见 cifar10/config/cifar10.toml;
本地数据只读; 训练集分层划分为 45000/5000，测试集仅在训练结束后评估。
按验证损失选择 best.pt; model.pt 为最后一轮参数，均不包含续训状态。
"""

import argparse
import csv
import json
import math
import random

import numpy as np
from sklearn.model_selection import train_test_split
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from time import perf_counter
import tomllib

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cifar10/config/cifar10.toml"


MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def validate_config(config):
    for key in ("epochs", "batch_size", "num_threads", "step_size", "patience"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers 必须为非负整数")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2**32:
        raise ValueError("seed 必须为 [0, 2**32) 内的整数")
    for key in ("learning_rate", "gamma", "momentum", "weight_decay", "min_delta"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} 必须为有限非负数")
    if config["learning_rate"] == 0 or not 0 < config["gamma"] <= 1:
        raise ValueError("learning_rate 必须为正数，gamma 必须在 (0, 1] 内")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device 只能为 auto、cpu 或 cuda")
    if config["model"] not in ("cnn", "resnet18", "nin", "densenet121"):
        raise ValueError("model 只能为 cnn、resnet18、nin 或 densenet121")
    augmentation = config["augmentation"]
    if not isinstance(augmentation, dict):
        raise ValueError("augmentation 必须为配置表")
    for key in ("enabled", "random_crop", "horizontal_flip"):
        if type(augmentation[key]) is not bool:
            raise ValueError(f"augmentation.{key} 必须为布尔值")
    if type(augmentation["crop_padding"]) is not int or augmentation["crop_padding"] < 0:
        raise ValueError("augmentation.crop_padding 必须为非负整数")
    probability = augmentation["horizontal_flip_probability"]
    if (type(probability) not in (int, float) or not math.isfinite(probability)
            or not 0 <= probability <= 1):
        raise ValueError("augmentation.horizontal_flip_probability 必须在 [0, 1] 内")


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_data(config):
    """使用分层划分和增强; 验证集、测试集不随机增强。"""
    data_root = REPO_ROOT / config["data_root"]
    plain = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    augmentation = config["augmentation"]
    train_transforms = []
    if augmentation["enabled"]:
        if augmentation["random_crop"]:
            train_transforms.append(
                transforms.RandomCrop(32, padding=augmentation["crop_padding"])
            )
        if augmentation["horizontal_flip"]:
            train_transforms.append(
                transforms.RandomHorizontalFlip(
                    p=augmentation["horizontal_flip_probability"]
                )
            )
    train_transforms.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    train_transform = transforms.Compose(train_transforms)

    try:
        raw = datasets.CIFAR10(data_root, train=True, transform=plain, download=False)
    except RuntimeError as error:
        raise RuntimeError(
            f"CIFAR-10 数据缺失或损坏: {data_root}; 请先运行 "
            "python data/code/download_cifar10.py"
        ) from error

    train_ids, val_ids = train_test_split(
        np.arange(len(raw)), test_size=5000, stratify=raw.targets, random_state=config["seed"],
    )
    train_data = Subset(datasets.CIFAR10(
        data_root, train=True, transform=train_transform, download=False), train_ids)
    val_data = Subset(raw, val_ids)
    test_data = datasets.CIFAR10(data_root, train=False, transform=plain, download=False)
    common = dict(batch_size=config["batch_size"], num_workers=config["num_workers"],
                  worker_init_fn=seed_worker)

    train_loader = DataLoader(train_data, shuffle=True,
                              generator=torch.Generator().manual_seed(config["seed"]), **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)

    return train_loader, val_loader, test_loader, raw.classes, train_ids, val_ids


def build_model(config):
    """返回指定模型，输出为 10 类。"""
    if config["model"] == "cnn":
        # [B,3,32,32] -> [B,32,16,16] -> [B,64,8,8] -> [B,128,1,1] -> [B,10]。
        return nn.Sequential(                                           # [B,3,32,32]
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> [B,32,16,16] 卷积层+激活+池化
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# -> [B,64,8,8] 卷积层+激活+池化
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),                # -> [B,128,8,8] 卷积层+激活
            nn.AdaptiveAvgPool2d(1),                                    # -> [B,128,1,1] 全局平均池化
            nn.Flatten(), nn.Linear(128, 10),                           # -> [B,10] 全连接层
        )
    if config["model"] == "resnet18":
        # 小图像使用 3x3、步长 1 的入口卷积，移除入口最大池化。
        model = models.resnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    if config["model"] == "nin":
        # NiN 使用卷积层组成的 MLP 块，并以全局平均池化输出类别分数。
        return nn.Sequential(
            nn.Conv2d(3, 96, 5, padding=2), nn.ReLU(),
            nn.Conv2d(96, 96, 1), nn.ReLU(),
            nn.Conv2d(96, 96, 1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(0.5),
            nn.Conv2d(96, 192, 5, padding=2), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1), nn.Dropout(0.5),
            nn.Conv2d(192, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 192, 1), nn.ReLU(),
            nn.Conv2d(192, 10, 1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
    if config["model"] == "densenet121":
        # 小图像使用 3x3、步长 1 的入口卷积，并移除入口最大池化。
        model = models.densenet121(weights=None, num_classes=10)
        model.features.conv0 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
        return model
    raise ValueError(f"不支持的模型: {config['model']}")


def run_epoch(model, loader, loss_fn, device, optimizer=None, confusion_matrix=None):
    """返回按样本数加权的平均交叉熵和准确率。"""
    training = optimizer is not None
    if training and confusion_matrix is not None:
        raise ValueError("混淆矩阵仅用于固定模型的评估")
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现 NaN/Inf，请检查输入、学习率与初始化方法。")
            if training:
                optimizer.zero_grad()  # 清除上一个批量留下的梯度。
                loss.backward()        # 计算当前批量的梯度。
                nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()       # 用 SGD 更新权重和偏置。
            count = labels.numel()
            total_loss += loss.item() * count
            predicted = logits.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()
            if confusion_matrix is not None:
                # 行为真实类别，列为预测类别; 与准确率使用同一次前向结果。
                n_classes = confusion_matrix.shape[0]
                indices = (labels * n_classes + predicted).detach().cpu()
                confusion_matrix += torch.bincount(
                    indices, minlength=n_classes**2
                ).reshape(n_classes, n_classes)
            total_samples += count
    if total_samples == 0:
        raise ValueError("数据加载器为空，无法计算指标")
    return total_loss / total_samples, total_correct / total_samples



def save_confusion_matrix(matrix, classes, run_dir):
    """保存原始计数 CSV、计数热力图和按真实类别归一化的百分比热力图。"""
    # 使用非交互画布，支持 WSL 和无图形界面的终端。
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    counts = matrix.detach().cpu().numpy()
    with (run_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label / predicted_label", *classes])
        for name, row in zip(classes, counts):
            writer.writerow([name, *row.tolist()])

    totals = counts.sum(axis=1, keepdims=True)
    # 若某真实类别没有样本，其百分比无定义，图中显示 N/A。
    percentages = np.divide(
        counts * 100.0, totals, out=np.full(counts.shape, np.nan), where=totals != 0
    )
    for values, filename, title, unit, maximum in (
        (counts, "confusion_matrix.png", "Confusion matrix (counts)", "Samples", max(1, counts.max())),
        (percentages, "confusion_matrix_normalized.png",
         "Confusion matrix (row-normalized)", "Percent of true class (%)", 100),
    ):
        fig = Figure(figsize=(10, 9), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum, interpolation="nearest")
        fig.colorbar(image, ax=ax, label=unit)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
               xticklabels=classes, yticklabels=classes,
               xlabel="Predicted label", ylabel="True label", title=title)
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")
        for i in range(len(classes)):
            for j in range(len(classes)):
                value = values[i, j]
                text = ("N/A" if not np.isfinite(value) else
                        f"{value:.1f}%" if unit != "Samples" else str(int(value)))
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if value > maximum * 0.6 else "black")
        fig.savefig(run_dir / filename, dpi=180)
        fig.clear()


def main():

    # 解析命令行参数; TOML 文件由后面的 tomllib 读取。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的批量大小")
    parser.add_argument("--learning-rate", type=float, help="覆盖配置中的学习率")
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument("--patience", type=int, help="覆盖配置中的早停等待轮数")
    parser.add_argument("--min-delta", type=float, help="覆盖配置中的最小验证损失改善值")
    parser.add_argument("--step-size", type=int, help="覆盖配置中的学习率衰减间隔")
    parser.add_argument("--gamma", type=float, help="覆盖配置中的学习率衰减系数")
    parser.add_argument("--model", choices=["cnn", "resnet18", "nin", "densenet121"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction,
                        help="启用或禁用所有随机图像增广")
    parser.add_argument("--random-crop", action=argparse.BooleanOptionalAction,
                        help="启用或禁用随机裁剪")
    parser.add_argument("--crop-padding", type=int, help="随机裁剪的补边像素数")
    parser.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction,
                        help="启用或禁用随机水平翻转")
    parser.add_argument("--horizontal-flip-probability", type=float,
                        help="随机水平翻转的概率")
    parser.add_argument("--logger", choices=["wandb", "tensorboard", "both", "none"],
                        default="wandb", help="选择实验记录工具，默认 wandb")
    parser.add_argument("--wandb-mode", choices=["online", "offline"],
                        default="online", help="启用 WandB 时选择在线同步或本地记录")
    parser.add_argument("--wandb-project", default="d2l-learn", help="WandB 项目名称")
    args = parser.parse_args()

    # 只加载本次选用的日志工具。
    use_tb = args.logger in ("tensorboard", "both")
    use_wandb = args.logger in ("wandb", "both")
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
    if use_wandb:
        import wandb

    # 读取配置文件
    with args.config.open("rb") as file:
        config = tomllib.load(file)
    for key in ("batch_size", "learning_rate", "epochs", "device", "model",
                "patience", "min_delta", "step_size", "gamma"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for argument, setting in (
        ("augmentation", "enabled"),
        ("random_crop", "random_crop"),
        ("crop_padding", "crop_padding"),
        ("horizontal_flip", "horizontal_flip"),
        ("horizontal_flip_probability", "horizontal_flip_probability"),
    ):
        value = getattr(args, argument)
        if value is not None:
            config["augmentation"][setting] = value

    # 检查配置参数的有效性
    validate_config(config)

    # 设置随机种子、线程数和设备
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(config["num_threads"])
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用 --device cpu。")
    device = torch.device(device_name)
    train_loader, val_loader, test_loader, classes, train_ids, val_ids = load_data(config)
    model = build_model(config).to(device)
    loss_fn = nn.CrossEntropyLoss()  # 默认对当前批量求均值。
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"],
                                momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    print(f"device={device}; train={len(train_loader.dataset)}; test={len(test_loader.dataset)}", flush=True)

    # 创建输出目录并保存配置和元数据
    output_root = REPO_ROOT / config["output_root"]
    run_name = datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = output_root / run_name
    # 同一分钟重复运行时加简短序号，不覆盖已有实验。
    suffix = 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_dir = output_root / f"{run_name}-{suffix}"
            suffix += 1
    metadata = {"logger": args.logger,
                "wandb_mode": args.wandb_mode if use_wandb else None,
                "wandb_project": args.wandb_project if use_wandb else None,
                "config": config, "device": str(device), "torch_version": str(torch.__version__),
                "train_samples": len(train_loader.dataset), "test_samples": len(test_loader.dataset),
                "classes": classes, "val_samples": len(val_loader.dataset),
                "mean": MEAN, "std": STD, "source": "adapted from week1 CIFAR-10 baseline"}
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    np.savez(run_dir / "split.npz", train=train_ids, val=val_ids)

    # 每次实验使用独立日志目录; 正常结束或异常退出时自动关闭记录器。
    with ExitStack() as stack:
        tb_writer = None
        wb_run = None
        if use_tb:
            tb_writer = stack.enter_context(
                SummaryWriter(log_dir=str(run_dir / "tensorboard"))
            )
            tb_writer.add_text("config", json.dumps(config, ensure_ascii=False, indent=2))
            print(f"TensorBoard 日志：{run_dir / 'tensorboard'}", flush=True)
        if use_wandb:
            wb_run = stack.enter_context(wandb.init(
                project=args.wandb_project,
                name=f"cifar10-{config['model']}-{run_dir.name}",
                config={**config, "actual_device": str(device)},
                dir=str(run_dir),
                mode=args.wandb_mode,
            ))
            wb_run.define_metric("epoch")
            wb_run.define_metric("accuracy/*", step_metric="epoch")
            wb_run.define_metric("loss/*", step_metric="epoch")
            wb_run.define_metric("time/*", step_metric="epoch")
            wb_run.define_metric("learning_rate", step_metric="epoch")
            if args.wandb_mode == "online":
                print(f"WandB 网页：{wb_run.url}", flush=True)
            else:
                print("WandB 离线记录：当前不会同步到网页。", flush=True)

        # 训练和评估
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate",
                "time", "delta_time",
            ])

            # 进行模型训练和评估
            total_training_time = 0.0
            best_loss, bad_epochs, best_epoch = float("inf"), 0, 0
            for epoch in range(1, config["epochs"] + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_start = perf_counter()
                train_loss, train_acc = run_epoch(
                    model, train_loader, loss_fn, device, optimizer
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                delta_time = perf_counter() - epoch_start
                total_training_time += delta_time
                val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
                learning_rate = optimizer.param_groups[0]["lr"]
                if val_loss < best_loss - config["min_delta"]:
                    best_loss, bad_epochs, best_epoch = val_loss, 0, epoch
                    torch.save(model.state_dict(), run_dir / "best.pt")
                else:
                    bad_epochs += 1
                scheduler.step()
                writer.writerow([
                    epoch, train_loss, train_acc, val_loss, val_acc, learning_rate, total_training_time, delta_time,
                ])
                file.flush()
                # train 为训练过程中的统计; val 为本轮训练结束后的验证集评估。
                # 将指标写入本次启用的记录工具; CSV 始终保留。
                metrics = {
                    "accuracy/train": train_acc, "accuracy/val": val_acc,
                    "loss/train": train_loss, "loss/val": val_loss,
                    "learning_rate": learning_rate,
                    "time/epoch_seconds": delta_time,
                    "time/total_seconds": total_training_time,
                }
                if tb_writer is not None:
                    for tag, value in metrics.items():
                        tb_writer.add_scalar(tag, value, epoch)
                    tb_writer.flush()
                if wb_run is not None:
                    wb_run.log({"epoch": epoch, **metrics}, step=epoch)

                if bad_epochs >= config["patience"]:
                    break

        torch.save(model.state_dict(), run_dir / "model.pt")
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True))
        confusion_matrix = torch.zeros((len(classes), len(classes)), dtype=torch.int64)
        test_loss, test_acc = run_epoch(
            model, test_loader, loss_fn, device, confusion_matrix=confusion_matrix
        )
        save_confusion_matrix(confusion_matrix, classes, run_dir)
        result = {"test_loss": test_loss, "test_accuracy": test_acc,
                  "best_epoch": best_epoch, "epochs_completed": epoch,
                  "best_val_loss": best_loss, "prediction_checkpoint": "best.pt"}
        (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if tb_writer is not None:
            tb_writer.add_scalar("test/loss", test_loss, epoch)
            tb_writer.add_scalar("test/accuracy", test_acc, epoch)
        if wb_run is not None:
            wb_run.summary.update(result)
        # 预测时才把 logits 转成概率; 每行十个类别的概率之和为 1。
        model.eval()
        with torch.no_grad():
            images, labels = next(iter(test_loader))
            probabilities = model(images[:10].to(device)).softmax(dim=1).cpu()
        predictions = [{"true_label": int(label), "predicted_label": int(probs.argmax()),
                        "probabilities": probs.tolist()}
                       for label, probs in zip(labels[:10], probabilities)]
        (run_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(f"结果已保存：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
