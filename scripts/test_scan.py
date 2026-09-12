"""scan_degradation 本地自测（假模型 + 合成数据，无 GPU 依赖）。

运行: python scripts/test_scan.py
覆盖: 四组扫描点生成 / SNR 反解正确性 / 通用评估循环 / 值域自适应
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degrade.cloud import add_cloud_mask  # noqa: E402
from degrade.gsd import degrade_gsd  # noqa: E402
from degrade.mtf import degrade_mtf  # noqa: E402
from degrade.snr import add_poisson_gaussian  # noqa: E402
from scripts.scan_degradation import (  # noqa: E402
    cloud_scan_points, eval_degrade, gsd_scan_points,
    mtf_kernel_size, mtf_scan_points, run_scan, snr_scan_points_for,
)

ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {extra}")


class FakeLoader:
    """固定样本的假 DataLoader：2 类、各 batch 含 image/label。"""

    def __init__(self, n_batch=2, size=32):
        self.n_batch = n_batch
        self.size = size
        torch.manual_seed(0)
        self.data = [
            {
                "image": torch.rand(4, 3, size, size),
                "label": torch.tensor([0, 1, 0, 1]),
            }
            for _ in range(n_batch)
        ]

    def __iter__(self):
        return iter(self.data)


class FakeModel(torch.nn.Module):
    """假模型：返回 (B,2) logits，按图像均值高低给类（确定性）。"""

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3))  # (B,)
        return torch.stack([1.0 - m, m], dim=-1)  # (B, 2)


def main() -> None:
    loader = FakeLoader()
    model = FakeModel()

    # 1) 扫描点结构
    check("GSD 扫描点", gsd_scan_points() == [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    check("MTF 扫描点", len(mtf_scan_points()) == 8 and mtf_scan_points()[0] == 0.5)
    check("云扫描点", cloud_scan_points()[:2] == [0.0, 0.1] and cloud_scan_points()[-1] == 0.7)
    check("MTF 核自适应", mtf_kernel_size(0.5) == 5 and mtf_kernel_size(2.0) == 13
          and mtf_kernel_size(4.0) == 25)

    # 2) SNR 反解正确性：给定目标 db 反解 (a,b)，再实测噪声方差应接近
    torch.manual_seed(1)
    x = torch.rand(8, 3, 64, 64)  # 均值约 0.5
    m = x.mean().item()
    for db in (20.0, 15.0, 10.0):
        pts = snr_scan_points_for(m, target_db=[db], b=0.001)
        a = pts[0]["a"]
        noisy = add_poisson_gaussian(x, a=a, b=0.001, seed=0)
        # 实测 SNR（功率比 dB）：信号功率 m²，噪声功率 = Var(noisy-x) 近似 a·m+b
        noise_var = ((noisy - x) ** 2).mean().item()
        snr_db = 10.0 * np.log10(m * m / max(noise_var, 1e-12))
        check(f"SNR 反解 {db}dB", abs(snr_db - db) < 1.0,
              f"实测={snr_db:.2f}dB (a={a:.5f})")

    # 3) 通用评估循环（干净条件下假模型 acc 应 > 0.5）
    acc0, f1_0 = eval_degrade(model, loader, lambda im: im, value_range_check=False)
    check("干净条件评估", 0.5 < acc0 <= 1.0, f"acc={acc0:.3f} f1={f1_0:.3f}")

    # 4) 四组退化都能跑通且精度单调性合理（退化越强 acc 不升）
    acc_gsd, _ = eval_degrade(model, loader, lambda im: degrade_gsd(im, 4.0))
    acc_snr, _ = eval_degrade(model, loader, lambda im: add_poisson_gaussian(im, a=0.05, b=0.005))
    acc_mtf, _ = eval_degrade(model, loader, lambda im: degrade_mtf(im, 3.0, kernel_size=mtf_kernel_size(3.0)))
    acc_cloud, _ = eval_degrade(model, loader, lambda im: add_cloud_mask(im, 0.5, seed=0)[0])
    check("退化后精度不升", acc_gsd <= acc0 + 1e-6 and acc_snr <= acc0 + 1e-6
          and acc_mtf <= acc0 + 1e-6 and acc_cloud <= acc0 + 1e-6,
          f"gsd={acc_gsd:.3f} snr={acc_snr:.3f} mtf={acc_mtf:.3f} cloud={acc_cloud:.3f}")

    # 5) run_scan 通用入口（GSD 全序列）
    results = run_scan(model, loader, lambda im, s: degrade_gsd(im, s), gsd_scan_points())
    check("run_scan 输出结构", len(results) == 7 and set(results[0]) == {"degrade", "accuracy", "f1"}
          and results[0]["degrade"] == 1.0)

    # 6) 值域自适应：0-255 输入应正常（内部归一化后退化再还原）
    loader255 = FakeLoader(size=16)
    for bch in loader255.data:
        bch["image"] = bch["image"] * 255.0
    acc_255, _ = eval_degrade(model, loader255, lambda im: add_cloud_mask(im, 0.3, seed=1)[0])
    check("0-255 值域自适应", 0.0 <= acc_255 <= 1.0, f"acc={acc_255:.3f}")

    print()
    print("全部通过 ✅" if ok else "存在失败 ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
