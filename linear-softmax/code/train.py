"""Fashion-MNIST softmax 回归: 对应 d2l 中文版 3.7 节。

可修改参数:
超参数 (默认值: --epochs 10、--batch-size 256、--learning-rate 0.1):
--epochs、--batch-size 为正整数; --learning-rate 为有限正数。
系统配置:
--config 为 TOML 文件路径;
--device 可选 auto、cpu、cuda;
记录配置:
--logger 可选 wandb、tensorboard、both、none。
--wandb-mode 可选 online、offline; --wandb-project 为项目名称。
不自动下载数据。配置中的相对路径以仓库根目录为基准。
"""

import argparse
import csv
import json
import math
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from time import perf_counter
import tomllib

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "linear-softmax/config/softmax.toml"


def build_model():
    # 输入为[B, 1, 28, 28]，B为当前批量大小。
    model = nn.Sequential(nn.Flatten(),                     #  -> [B, 784]
                          nn.Linear(784, 10))               #  -> [B, 10]
    nn.init.normal_(model[1].weight, mean=0.0, std=0.01)    # weight1 ~ N(0, 0.01^2)
    nn.init.zeros_(model[1].bias)                           # bias1 = 0
    return model


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """训练或评估一轮，返回按样本数加权的平均损失和准确率。"""
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("损失出现 NaN/Inf，请检查输入与学习率。")
            if training:
                optimizer.zero_grad()  # 清除上一个批量留下的梯度。
                loss.backward()        # 计算当前批量的梯度。
                optimizer.step()       # 用 SGD 更新权重和偏置。
            count = labels.numel()
            total_loss += loss.item() * count
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += count
    return total_loss / total_samples, total_correct / total_samples


def main():

    # 解析命令行参数; TOML 文件由后面的 tomllib 读取。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--epochs", type=int, help="覆盖配置中的训练轮数")
    parser.add_argument("--learning-rate", type=float, help="覆盖配置中的学习率")
    parser.add_argument("--batch-size", type=int, help="覆盖配置中的批量大小")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
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
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device

    # 检查配置参数的有效性
    for key in ("epochs", "batch_size", "num_threads"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须为正整数")
    if type(config["num_workers"]) is not int or config["num_workers"] < 0:
        raise ValueError("num_workers 必须为非负整数")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate 必须为有限正数")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device 只能为 auto、cpu 或 cuda")

    # 设置随机种子、线程数和设备
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["num_threads"])
    device_name = config["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请使用 --device cpu。")
    device = torch.device(device_name)
    data_root = (REPO_ROOT / config["data_root"]).resolve()

    # 读取本地 Fashion-MNIST 数据集; download=False，不执行下载。
    train_data = datasets.FashionMNIST(
        data_root, train=True, transform=transforms.ToTensor(), download=False
    )
    test_data = datasets.FashionMNIST(
        data_root, train=False, transform=transforms.ToTensor(), download=False
    )
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], generator=generator,
    )
    test_loader = DataLoader(
        test_data, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"],
    )
    model = build_model().to(device)
    loss_fn = nn.CrossEntropyLoss()  # 默认对当前批量求均值。
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    print(f"device={device}; train={len(train_data)}; test={len(test_data)}", flush=True)

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
                "train_samples": len(train_data), "test_samples": len(test_data),
                "classes": train_data.classes}
    (run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

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
                name=f"softmax-{run_dir.name}",
                config={**config, "actual_device": str(device)},
                dir=str(run_dir),
                mode=args.wandb_mode,
            ))
            wb_run.define_metric("epoch")
            wb_run.define_metric("accuracy/*", step_metric="epoch")
            wb_run.define_metric("loss/*", step_metric="epoch")
            wb_run.define_metric("time/*", step_metric="epoch")
            if args.wandb_mode == "online":
                print(f"WandB 网页：{wb_run.url}", flush=True)
            else:
                print("WandB 离线记录：当前不会同步到网页。", flush=True)

        # 训练和评估
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy",
                "time", "delta_time",
            ])

            # 进行模型训练和评估
            total_training_time = 0.0
            for epoch in range(1, config["epochs"] + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                epoch_start = perf_counter()
                train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                delta_time = perf_counter() - epoch_start
                total_training_time += delta_time
                test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)
                writer.writerow([
                    epoch, train_loss, train_acc, test_loss, test_acc, total_training_time, delta_time,
                ])
                file.flush()
                # train 为训练过程中的统计; test 为本轮训练结束后的评估。
                # 将指标写入本次启用的记录工具; CSV 始终保留。
                metrics = {
                    "accuracy/train": train_acc, "accuracy/test": test_acc,
                    "loss/train": train_loss, "loss/test": test_loss,
                    "time/epoch_seconds": delta_time,
                    "time/total_seconds": total_training_time,
                }
                if tb_writer is not None:
                    for tag, value in metrics.items():
                        tb_writer.add_scalar(tag, value, epoch)
                    tb_writer.flush()
                if wb_run is not None:
                    wb_run.log({"epoch": epoch, **metrics}, step=epoch)

        torch.save(model.state_dict(), run_dir / "model.pt")
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
