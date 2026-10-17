"""下载并检查 d2l 基础分类示例使用的 CIFAR-10。

在 cv 环境、仓库根目录运行 python data/code/download_cifar10.py。
只在直接运行本文件时下载; 导入本模块不会启动下载。
"""

from pathlib import Path
from torchvision import datasets, transforms


def main():
    # 数据集根目录: data/。torchvision 会在它下面创建 cifar-10-batches-py 子目录。
    data = Path(__file__).resolve().parents[1]
    print("准备下载 CIFAR-10", flush=True)

    # ToTensor 在读取样本时将图片转成 [通道, 高, 宽] 的张量，
    # 并将像素由 0~255 转为 0~1; 不会改写磁盘上的原始数据。
    transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(
        root=data, train=True, transform=transform, download=True
    )
    test_data = datasets.CIFAR10(
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
    print(f"原始文件目录：{data / train_data.base_folder}")
    print("数据保存在 CIFAR-10 Python batch 文件中。")
    print("后续读取请使用同一个 root，并设置 download=False。")


if __name__ == "__main__":
    main()
