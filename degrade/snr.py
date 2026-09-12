"""SNR（信噪比）退化：泊松-高斯噪声模型。

物理含义（成像链路的噪声模型）：
- 散粒噪声（泊松）：光子计数统计涨落，方差 ∝ 信号强度 a·I
- 读出噪声（高斯）：ADC/读出电路，方差恒定 b
- 总方差：σ² = a·I + b  —— 这是"懂传感器物理"的噪声模型，
  区别于纯计算机背景常用的加性高斯白噪声。
"""
from __future__ import annotations

import torch


def add_poisson_gaussian(
    x: torch.Tensor,
    a: float = 0.01,
    b: float = 0.001,
    seed: int | None = None,
) -> torch.Tensor:
    """添加泊松-高斯噪声。

    Args:
        x: (B, C, H, W) 归一化 [0,1] 影像。
        a: 散粒噪声系数（乘性，∝ 信号）。
        b: 读出噪声方差（加性常数）。
        seed: 随机种子（可复现）。

    Returns:
        加噪后的影像（clamp 到 [0,1]）。
    """
    gen = torch.Generator(device=x.device)
    if seed is not None:
        gen.manual_seed(seed)

    # 散粒噪声：标准差 = sqrt(a * I)，乘性
    shot_std = torch.sqrt(torch.clamp(a * x, min=0.0))
    shot = shot_std * torch.randn_like(x, generator=gen)

    # 读出噪声：标准差 = sqrt(b)，加性常数
    readout_std = torch.sqrt(torch.tensor(b, device=x.device, dtype=x.dtype))
    readout = readout_std * torch.randn_like(x, generator=gen)

    noisy = x + shot + readout
    return noisy.clamp(0.0, 1.0)
