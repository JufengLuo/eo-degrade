"""MTF（调制传递函数）退化：光学/运动模糊。

物理含义：
- MTF_motion = sinc(π·f·d)，像移量 d 来自平台微振动/推扫运动
- MTF_optics 近似高斯型滚降
- 近似实现：高斯模糊核（σ 越大 → MTF 滚降越早 → 高频损失越严重）

注意：本实现为 MTF 的近似模拟（高斯核），精确 MTF 需要真实光学链路参数，
但用于"退化→精度"定量扫描已足够。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    grid = coords[:, None] ** 2 + coords[None, :] ** 2
    k = torch.exp(-grid / (2.0 * sigma**2))
    return k / k.sum()


def degrade_mtf(x: torch.Tensor, sigma: float = 1.0, kernel_size: int = 5) -> torch.Tensor:
    """模拟 MTF 退化（高斯模糊近似）。

    Args:
        x: (B, C, H, W) 归一化 [0,1] 影像。
        sigma: 模糊强度（MTF 滚降程度），越大越糊。
        kernel_size: 卷积核尺寸（需为奇数）。

    Returns:
        退化后的影像。
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    k = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    k = k.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, k, padding=kernel_size // 2, groups=x.shape[1])
