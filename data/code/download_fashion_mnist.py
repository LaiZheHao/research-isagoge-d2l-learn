"""下载并检查 d2l 基础分类示例使用的 Fashion-MNIST。

在 cv 环境、仓库根目录运行 python data/code/download_fashion_mnist.py。
只在直接运行本文件时下载; 导入本模块不会启动下载。
"""

from pathlib import Path
from torchvision import datasets, transforms


def main():
    # 数据集根目录: data/。torchvision 会在它下面按数据集类名自动创建子目录 FashionMNIST。
    data = Path(__file__).resolve().parents[1]
    print("准备下载 Fashion-MNIST", flush=True)

    # ToTensor 在读取样本时将图片转成 [通道, 高, 宽] 的张量，
    # 并将像素由 0~255 转为 0~1; 不会改写磁盘上的原始数据。
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(
        root=data, train=True, transform=transform, download=True
    )
    test_data = datasets.FashionMNIST(
        root=data, train=False, transform=transform, download=True
    )

    # 输出各数据集的数量、样本形状和第一个样本的标签。
    for name, dataset in (
        ("训练集", train_data),
        ("测试集", test_data),
    ):
        image, label = dataset[0]
        print(f"{name}：{len(dataset)} 张，样本形状 {tuple(image.shape)}")
        print(f"  第一个样本标签：{label}（{dataset.classes[label]}）")

    print("\n下载及基本读取检查完成。")
    print(f"数据根目录（root）：{data}")
    print(f"原始文件目录：{train_data.raw_folder}")
    print("数据以 IDX 二进制文件保存，不是逐张 PNG/JPG 图片。")
    print("后续读取请使用同一个 root，并设置 download=False。")


if __name__ == "__main__":
    main()
