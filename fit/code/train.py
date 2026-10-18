"""多项式回归: 依据 d2l 4.4.4 节实现。

可修改参数 (默认值: --i 3、--batch-size 10、--lr 0.01、--epochs 100):
--i 为多项式阶数，取非负整数;
--batch-size、--epochs 为正整数; --lr 为有限正数。
--regularization / --no-regularization 开关 L2; --regularization-strength 指定 lambda。
--dropout / --no-dropout 开关特征暂退; --dropout-probability 指定 p。
--train-samples 选择原训练集前 N 个样本，不改写数据文件。
训练和测试 loss 均为每轮更新结束后、关闭 Dropout 时的纯 MSE。
"""

import argparse
import csv
import json
import math
import hashlib
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import tomllib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/polynomial/dataset.pt"
CONFIG_PATH = REPO_ROOT / "fit/config/polynomial.toml"


def load_data():
    """读取由 data/code/generate_polynomial.py 生成的训练集和测试集。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据集 {DATA_PATH}，请先运行 "
            "'python data/code/generate_polynomial.py'"
        )
    return torch.load(DATA_PATH, weights_only=True)


def polynomial_features(x, degree):
    """将 x 展开为 [1, x, x^2/2!, ..., x^i/i!]，共 i+1 个特征。"""
    features = torch.ones((len(x), degree + 1), dtype=torch.float64)
    # 递推计算，避免直接先计算很大的 x^i 和 i!。
    for power in range(1, degree + 1):
        features[:, power] = features[:, power - 1] * x[:, 0].double() / power
    features = features.float()
    if not torch.isfinite(features).all():
        raise ValueError("多项式特征超出有限数范围")
    return features


def dropout_features(features, probability):
    """只暂退非常数特征，保留截距; 使用 inverted dropout 保持特征期望。"""
    if probability == 0:
        return features
    return torch.cat((features[:, :1], nn.functional.dropout(
        features[:, 1:], p=probability, training=True)), dim=1)


def l2_penalty(model):
    """lambda 乘本函数后加入目标; 常数特征对应的截距不参与惩罚。"""
    return model[0].weight[:, 1:].square().sum() / 2


def train_epoch(model, loader, optimizer, loss_fn, strength=0.0, dropout_probability=0.0):
    """完成一轮小批量 SGD 更新。"""
    model.train()
    for features, labels in loader:
        prediction = model(dropout_features(features, dropout_probability))
        loss = loss_fn(prediction, labels)  # 预测和标签均为 [B, 1]。
        if strength:
            loss = loss + strength * l2_penalty(model)
        if not torch.isfinite(loss):
            raise RuntimeError("MSE 出现 NaN/Inf，请检查学习率与阶数")
        optimizer.zero_grad()  # 清除上一个批量留下的梯度。
        loss.backward()        # 计算当前批量的梯度。
        optimizer.step()       # 用 SGD 更新模型参数。


def main():

    # 读取默认参数并解析命令行参数
    with CONFIG_PATH.open("rb") as file:
        config = tomllib.load(file)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", type=int, default=config["i"], help="多项式阶数，0 表示常数模型")
    parser.add_argument("--batch-size", type=int, default=config["batch_size"])
    parser.add_argument("--lr", "--learning-rate", type=float, default=config["learning_rate"])
    parser.add_argument("--epochs", type=int, default=config["epochs"])
    parser.add_argument("--logger", choices=["wandb", "tensorboard", "both", "none"], default="wandb")
    parser.add_argument("--regularization", action=argparse.BooleanOptionalAction,
                        default=config.get("regularization", False), help="是否启用 L2 惩罚")
    parser.add_argument("--regularization-strength", type=float,
                        default=config.get("regularization_strength", 0.001), help="L2 的 lambda")
    parser.add_argument("--dropout", action=argparse.BooleanOptionalAction,
                        default=config.get("dropout", False), help="是否暂退非常数多项式特征")
    parser.add_argument("--dropout-probability", type=float,
                        default=config.get("dropout_probability", 0.1), help="特征暂退概率 p")
    parser.add_argument("--train-samples", type=int, default=config.get("train_samples", 100),
                        help="使用原训练集前 N 个样本，测试集保持不变")
    args = parser.parse_args()
    if args.i < 0 or args.batch_size <= 0 or args.epochs <= 0:
        parser.error("i 必须为非负整数; batch-size、epochs 必须为正整数")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr 必须为有限正数")

    if not math.isfinite(args.regularization_strength) or args.regularization_strength < 0:
        parser.error("regularization-strength 必须为有限非负数")
    if not math.isfinite(args.dropout_probability) or not 0 <= args.dropout_probability < 1:
        parser.error("dropout-probability 必须在 [0, 1) 内")
    if args.train_samples <= 0:
        parser.error("train-samples 必须为正整数")
    strength = args.regularization_strength if args.regularization else 0.0
    probability = args.dropout_probability if args.dropout else 0.0

    # 固定种子和 CPU 线程数; 小规模数据直接使用 CPU。
    torch.manual_seed(42)
    torch.set_num_threads(1)
    data = load_data()
    if args.train_samples > len(data["train_x"]):
        parser.error("train-samples 不能超过已有训练集样本数")
    train_x = polynomial_features(data["train_x"][:args.train_samples], args.i)
    test_x = polynomial_features(data["test_x"], args.i)
    train_y, test_y = data["train_y"][:args.train_samples], data["test_y"]
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(42))
    # 第一列为常数 1，已经承担偏置作用; 参数量为 i+1。
    model = nn.Sequential(nn.Linear(args.i + 1, 1, bias=False))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # 创建输出目录; 同一分钟重复运行时追加序号，避免覆盖。
    name = datetime.now().strftime("%Y%m%d-%H%M") + f"-i{args.i}"
    output_root = REPO_ROOT / "fit/output"
    run_dir, suffix = output_root / name, 1
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            run_dir = output_root / f"{name}-{suffix}"
            suffix += 1
    metadata = {**vars(args), "seed": 42, "device": "cpu", "dataset": str(DATA_PATH),
                "n_train": len(train_y), "n_test": len(test_y), "noise_std": 0.1,
                "basis": "x^j / j!", "true_weights": [5, 1.2, -3.4, 5.6],
                "effective_l2": strength, "effective_dropout": probability,
                "dropout_location": "nonconstant polynomial input features, training only",
                "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (run_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 只加载选用的记录器; 正常或异常退出时自动关闭。
    with ExitStack() as stack:
        tb_writer, wb_run = None, None
        if args.logger in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter
            tb_writer = stack.enter_context(SummaryWriter(log_dir=str(run_dir / "tensorboard")))
            tb_writer.add_text("config", json.dumps(metadata, indent=2))
        if args.logger in ("wandb", "both"):
            import wandb
            wb_run = stack.enter_context(wandb.init(
                project="d2l-learn", name=f"polynomial-{run_dir.name}",
                config=metadata, dir=str(run_dir),
            ))
            wb_run.define_metric("epoch")
            wb_run.define_metric("loss/*", step_metric="epoch")

        # 每轮更新后用同一个模型评估两组 MSE，并记录 loss 曲线。
        with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "test_loss"])
            for epoch in range(1, args.epochs + 1):
                train_epoch(model, loader, optimizer, loss_fn, strength, probability)
                model.eval()
                with torch.no_grad():
                    train_loss = loss_fn(model(train_x), train_y).item()
                    test_loss = loss_fn(model(test_x), test_y).item()
                if not all(math.isfinite(v) for v in (train_loss, test_loss)):
                    raise RuntimeError("评估 MSE 出现 NaN/Inf，请检查学习率与阶数")
                writer.writerow([epoch, train_loss, test_loss])
                file.flush()
                metrics = {"loss/train": train_loss, "loss/test": test_loss}
                if tb_writer is not None:
                    for tag, value in metrics.items():
                        tb_writer.add_scalar(tag, value, epoch)
                if wb_run is not None:
                    wb_run.log({"epoch": epoch, **metrics}, step=epoch)
        torch.save(model.state_dict(), run_dir / "model.pt")
        if wb_run is not None:
            wb_run.summary.update({"final_train_mse": train_loss, "final_test_mse": test_loss,
                                   "epochs_completed": epoch})
    print(f"i={args.i}; train MSE={train_loss:.6f}; test MSE={test_loss:.6f}", flush=True)
    print(f"结果已保存: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
