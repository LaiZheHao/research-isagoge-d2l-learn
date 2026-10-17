"""生成多项式回归使用的固定训练集和测试集。

在仓库根目录运行 python data/code/generate_polynomial.py。
"""

from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data/polynomial/dataset.pt"


def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 1))
    rng.shuffle(x)
    # 真实规律固定为三次多项式，与待训练模型的阶数无关。
    y = 5 + 1.2 * x - 3.4 * x**2 / 2 + 5.6 * x**3 / 6
    y += rng.normal(scale=0.1, size=y.shape)
    x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_x": x[:100], "train_y": y[:100],
                "test_x": x[100:], "test_y": y[100:]}, DATA_PATH)
    print(f"数据集已保存: {DATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
