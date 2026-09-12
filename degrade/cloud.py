"""云遮挡模拟：云团模型 + 比尔-朗伯透过 + 云影（物理合理的"缺失数据"）。

物理含义（对照旧版"随机像素"近似的升级）：
1. 云场结构：真实云是离散的云团（积云块 / 层云片），云团中心近似泊松分布，
   径向剖面用超高斯 exp(-(d/σ)^1.6) 逼近（边缘比高斯更平缓，更像真实云），
   云团内部叠加小幅高频纹理（云内不均质）。
   形态可控：云团数量、大小、强度可调；浓云比例用分位数阈值精确锁定。
2. 透过率：比尔-朗伯定律 T = exp(-τ)，不透明度 opacity = 1 - exp(-τ)。
   - 浓云（积云）：τ 大 → opacity → 1 → 地表信号完全被云顶反射替代（数据缺失）
   - 薄云（层云）：τ 小 → opacity 小 → 地表信号部分透过（亮度调制，非完全缺失）
3. 云影：太阳光被云体遮挡后在地面形成的暗区——云掩膜沿太阳方位平移 + 膨胀，
   并扣除云体自身区域（阴影不落在云上），阴影区亮度乘以衰减因子。
4. 云顶反射：可见光波段高反射，云中心（光学厚度峰值）更亮，带空间结构。

用途：
- 缺失率扫描（10% → 70%）：mask=1 表示数据缺失（浓云 + 云影）
- 掩膜-条件重建 / 鲁棒训练（P1）：opacity 场可作半透明云软标签

旧接口 `add_cloud_mask` 签名保持兼容；完整信息用 `add_cloud_field`。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# 云类型 → 云团几何参数（sigma 范围、强度范围、光学厚度放大 k）
_CLOUD_PARAMS = {
    "cumulus": dict(sigma=(0.05, 0.13), amp=(0.8, 2.2), k=6.0),   # 浓密块状积云
    "stratus": dict(sigma=(0.08, 0.18), amp=(0.4, 1.0), k=1.5),   # 铺开半透明层云
    "mixed":   dict(sigma=(0.05, 0.16), amp=(0.5, 1.8), k=3.0),   # 混合
}


def _cloud_field(
    b: int,
    h: int,
    w: int,
    cloud_type: str,
    n_clouds: int | None = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """云团场 (B,1,H,W)：泊松分布云团中心 + 超高斯剖面 + 内部纹理。

    云团数量默认按影像面积取 4-8 个（约 1 团 / (64px)² 尺度）。
    """
    p = _CLOUD_PARAMS[cloud_type]
    s_min, s_max = p["sigma"]
    a_min, a_max = p["amp"]

    if n_clouds is None:
        n_clouds = max(3, min(10, int(round((h * w) / (64 * 64)))))
    n_clouds = max(1, int(n_clouds))

    field = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
    yy = torch.arange(h, device=device, dtype=dtype).view(-1, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, -1)

    for bi in range(b):
        cx = torch.rand(n_clouds, device=device, dtype=dtype, generator=gen) * w
        cy = torch.rand(n_clouds, device=device, dtype=dtype, generator=gen) * h
        sigma = (s_min + (s_max - s_min) * torch.rand(n_clouds, device=device, dtype=dtype, generator=gen)) * min(h, w)
        amp = a_min + (a_max - a_min) * torch.rand(n_clouds, device=device, dtype=dtype, generator=gen)
        for i in range(n_clouds):
            d2 = (xx - cx[i]) ** 2 + (yy - cy[i]) ** 2
            # 超高斯剖面（指数 0.8 → 形状参数 1.6，边缘平缓如云）
            field[bi, 0] = field[bi, 0] + amp[i] * torch.exp(-(d2 / (2.0 * sigma[i] ** 2)) ** 0.8)

    # 内部纹理：小幅高频调制（云内不均质）
    if n_clouds > 0:
        texture = _fbm_field(b, h, w, octaves=3, freq0=8.0, device=device, dtype=dtype, gen=gen)
        field = field * (1.0 + 0.15 * (texture - 0.5))
    return field


def _fbm_field(
    b: int,
    h: int,
    w: int,
    octaves: int = 3,
    freq0: float = 8.0,
    persistence: float = 0.5,
    device: torch.device = None,
    dtype: torch.dtype = None,
    gen: torch.Generator | None = None,
) -> torch.Tensor:
    """分形布朗运动噪声场 (B,1,H,W)，值域近似 [0,1]（内部纹理用）。"""
    field = torch.zeros(b, 1, h, w, device=device, dtype=dtype)
    amp = 1.0
    total_amp = 0.0
    freq = freq0
    for _ in range(octaves):
        gh = max(2, int(h / freq))
        gw = max(2, int(w / freq))
        n = torch.rand(b, 1, gh, gw, device=device, dtype=dtype, generator=gen)
        n = F.interpolate(n, size=(h, w), mode="bilinear", align_corners=False)
        field = field + amp * n
        total_amp += amp
        amp *= persistence
        freq *= 2.0
    return field / total_amp


def _quantile_threshold(field: torch.Tensor, keep_frac: float) -> torch.Tensor:
    """取标量阈值，使 field >= thr 的像素占比 ≈ keep_frac（0-1）。

    用于精确控制"浓云覆盖率"，保证扫描实验中遮挡率可复现、可对齐。
    """
    if keep_frac <= 0.0:
        return torch.full((), float("inf"), device=field.device, dtype=field.dtype)
    if keep_frac >= 1.0:
        return torch.full((), -float("inf"), device=field.device, dtype=field.dtype)
    flat = field.flatten()
    # 从大到小排序，取前 keep_frac 比例的最大值对应的下界作为阈值
    k = max(0, min(flat.numel() - 1, int(round(keep_frac * flat.numel())) - 1))
    return torch.sort(flat, descending=True).values[k]


def _dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """mask 膨胀（max pooling），用于把云体扩散成云影覆盖区。"""
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def add_cloud_field(
    x: torch.Tensor,
    cloud_fraction: float = 0.3,
    seed: int | None = None,
    cloud_type: str = "cumulus",
    shadow: bool = True,
    shadow_offset: tuple[int, int] | None = None,
    shadow_atten: float = 0.4,
    n_clouds: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成带完整云场的退化影像。

    Args:
        x: (B, C, H, W) 归一化 [0,1] 影像。
        cloud_fraction: 浓云覆盖比例（0-1，分位数精确控制）。
        seed: 随机种子（可复现）。
        cloud_type: "cumulus"（积云，浓密块状）| "stratus"（层云/薄云，半透明）
                    | "mixed"（混合）。
        shadow: 是否生成云影。
        shadow_offset: 云影相对云的偏移 (dx, dy) 像素；None 时按影像尺寸自动取
                       (~8% 宽度, ~6% 高度)，模拟低太阳角。
        shadow_atten: 云影区地表亮度衰减因子（0-1，越小越暗）。
        n_clouds: 云团数量；None 时按影像尺度自动取。

    Returns:
        (退化影像, 缺失掩膜, 不透明度场, 云影掩膜)
        - 缺失掩膜 (B,1,H,W)：1 = 数据缺失（浓云 opacity 高 + 云影），0 = 可用
        - 不透明度场 (B,1,H,W)：opacity ∈ [0,1]，薄云区域取中间值（软标签）
        - 云影掩膜 (B,1,H,W)：仅阴影区域为 1
    """
    assert 0.0 <= cloud_fraction <= 1.0, "cloud_fraction must be in [0,1]"
    assert cloud_type in _CLOUD_PARAMS, f"cloud_type must be one of {list(_CLOUD_PARAMS)}"
    assert 0.0 <= shadow_atten <= 1.0, "shadow_atten must be in [0,1]"

    b, c, h, w = x.shape
    device, dtype = x.device, x.dtype
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    k = _CLOUD_PARAMS[cloud_type]["k"]

    # 1) 云团场 → 光学厚度场 τ
    field = _cloud_field(b, h, w, cloud_type, n_clouds=n_clouds, device=device, dtype=dtype, gen=gen)
    # 陡峭化（积云更"块"，层云更"铺"）
    gamma = 1.4 if cloud_type == "cumulus" else (1.2 if cloud_type == "mixed" else 0.9)
    field = field ** gamma
    tau = k * field

    # 2) 比尔-朗伯 → 不透明度
    opacity = 1.0 - torch.exp(-tau)

    # 3) 浓云掩膜：分位数阈值，覆盖率精确 = cloud_fraction
    mask_thick = (opacity >= _quantile_threshold(opacity, cloud_fraction)).float()

    # 4) 云影：云掩膜平移 + 膨胀，扣除云体自身区域（阴影不落在云上）
    shadow_mask = torch.zeros_like(mask_thick)
    if shadow and cloud_fraction > 0.0:
        if shadow_offset is None:
            dx, dy = max(4, int(0.08 * w)), max(3, int(0.06 * h))
        else:
            dx, dy = shadow_offset
        shifted = torch.zeros_like(mask_thick)
        sx0, sx1 = max(0, dx), min(w, w + dx)
        sy0, sy1 = max(0, dy), min(h, h + dy)
        shifted[..., max(0, dy):sy1, max(0, dx):sx1] = mask_thick[
            ...,
            max(0, -dy):max(0, -dy) + (sy1 - max(0, dy)),
            max(0, -dx):max(0, -dx) + (sx1 - max(0, dx)),
        ]
        shadow_mask = _dilate_mask(shifted, radius=max(2, min(h, w) // 128))
        # 阴影不覆盖云体自身
        shadow_mask = torch.clamp(shadow_mask - mask_thick, 0.0, 1.0)

    # 5) 合成退化影像
    #    云顶反射（空间结构：云中心更亮）
    cloud_ref = 0.72 + 0.28 * (field / (field.max() + 1e-8))
    cloud_ref = cloud_ref.expand(b, c, h, w)
    #    比尔-朗伯混合：观测 = (1-opacity)·地表 + opacity·云顶反射
    degraded = (1.0 - opacity) * x + opacity * cloud_ref
    #    云影区：地表亮度按 shadow_atten 衰减
    if shadow and cloud_fraction > 0.0:
        atten = torch.where(shadow_mask > 0.5, shadow_atten, 1.0)
        visible = 1.0 - opacity
        degraded = x * visible * atten + opacity * cloud_ref
    degraded = degraded.clamp(0.0, 1.0)

    # 6) 缺失掩膜 = 浓云 | 云影
    mask = torch.clamp(mask_thick + shadow_mask, 0.0, 1.0)
    return degraded, mask, opacity, shadow_mask


def add_cloud_mask(
    x: torch.Tensor,
    cloud_fraction: float = 0.3,
    seed: int | None = None,
    cloud_type: str = "cumulus",
    shadow: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """随机云遮挡（兼容旧接口）。

    Args:
        x: (B, C, H, W) 归一化 [0,1] 影像。
        cloud_fraction: 遮挡比例（0-1）。
        seed: 随机种子（可复现）。
        cloud_type: "cumulus" | "stratus" | "mixed"。
        shadow: 是否生成云影。

    Returns:
        (退化影像, 掩膜)，掩膜为 (B, 1, H, W)，1 = 被遮挡（缺失）。
    """
    degraded, mask, _, _ = add_cloud_field(
        x,
        cloud_fraction=cloud_fraction,
        seed=seed,
        cloud_type=cloud_type,
        shadow=shadow,
    )
    return degraded, mask
