"""eo-degrade: 像质退化模拟核心库。

模拟四类真实遥感像质退化，用于"退化 → 解译精度"定量实验：
- gsd:   地面采样距离（分辨率）退化
- snr:   泊松-高斯噪声（散粒 + 读出）
- mtf:   调制传递函数退化（光学/运动模糊）
- cloud: 云遮挡（真实缺失）

所有函数输入输出为 (B, C, H, W) 归一化 [0,1] 张量，可批处理、可复现（seed）。
"""

from .gsd import degrade_gsd
from .snr import add_poisson_gaussian
from .mtf import degrade_mtf
from .cloud import add_cloud_mask

__all__ = ["degrade_gsd", "add_poisson_gaussian", "degrade_mtf", "add_cloud_mask"]
