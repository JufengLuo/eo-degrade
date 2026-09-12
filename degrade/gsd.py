"""GSD（地面采样距离）退化：模拟更粗空间分辨率。

物理含义：GSD 增大 → 地面细节被积分平均 → 高频信息丢失。
实现：双线性降采样到目标分辨率，再上采样回原尺寸（信息已不可恢复）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def degrade_gsd(x: torch.Tensor, scale: float) -> torch.Tensor:
    """模拟 GSD 退化（scale 倍）。

    Args:
        x: (B, C, H, W) 归一化 [0,1] 影像。
        scale: 分辨率退化倍数，>=1.0（1.0 表示不退化；2.0 表示 GSD 变粗 2 倍）。

    Returns:
        退化后的影像，与输入同尺寸。
    """
    assert scale >= 1.0, "scale must be >= 1.0"
    if scale == 1.0:
        return x

    b, c, h, w = x.shape
    h2 = max(1, int(round(h / scale)))
    w2 = max(1, int(round(w / scale)))

    down = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up
