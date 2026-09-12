"""退化扫描引擎：GSD / SNR / MTF / 云遮挡 四组"退化参数 → 精度"扫描的共用代码。

设计目标：
1. 四组扫描共用同一套评估循环（模型推理 + 精度/F1），Kaggle notebook 03 直接 import。
2. 扫描点物理可解释、可复现：
   - GSD：分辨率退化倍数（1.0×–8.0×）
   - SNR：目标信噪比 dB 序列，由 (a, b) 反解保证横轴均匀
   - MTF：高斯核 σ（自动适配核尺寸）
   - Cloud：浓云覆盖率（分位数精确控制）
3. 本地可测：`eval_degrade` 只依赖 (degrade_fn, model, loader)，可用假模型 + 合成数据验证。

本地测试：python scripts/test_scan.py
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch


# ---------------------------------------------------------------------------
# 扫描点生成
# ---------------------------------------------------------------------------

def gsd_scan_points() -> list[float]:
    """GSD 扫描：分辨率退化倍数（1.0 = 不退化）。"""
    return [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]


def snr_scan_points(
    target_db: Iterable[float] = (30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0),
    b: float = 0.001,
) -> list[dict]:
    """SNR 扫描点：由目标 SNR(dB) 反解泊松系数 a（固定读出噪声 b）。

    物理推导（对归一化 [0,1] 影像，图像级近似）：
        - 信号功率 ≈ m²（m 为图像均值）
        - 噪声方差 = a·m + b（泊松-高斯：散粒 a·I + 读出 b）
        - 线性 SNR = m² / (a·m + b)  →  a = (m²/SNR_lin − b) / m
    注意：a 依赖数据均值 m，因此扫描点须在拿到数据后调用
    `snr_scan_points_for(x_mean, ...)` 生成；此处仅为未定数据时的占位。

    Args:
        target_db: 目标 SNR 序列（dB，功率比定义 10·log10）。
        b: 读出噪声方差（固定硬件特性）。

    Returns:
        [{'db': ..., 'a': ..., 'b': ...}]，a 未定数据时为 None（见下）。
    """
    return [
        {"db": db, "a": None, "b": b}  # a 待数据就绪后由 snr_scan_points_for 填充
        for db in target_db
    ]


def snr_scan_points_for(
    x_mean: float,
    target_db: Iterable[float] = (30.0, 25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0),
    b: float = 0.001,
) -> list[dict]:
    """给定数据均值，生成可执行的 SNR 扫描点（a 已反解）。

    Args:
        x_mean: 数据集图像均值（0-1 归一化后）。
        target_db: 目标 SNR 序列（dB，功率比）。
        b: 读出噪声方差。

    Returns:
        [{'db':..., 'a':..., 'b':...}]；若某目标 SNR 在给定 b 下不可达
        （a<0），则 a 置 0 并保留，评估时以实测 SNR 为准。
    """
    points = []
    for db in target_db:
        snr_lin = 10.0 ** (db / 10.0)
        # a = (m²/SNR_lin − b) / m
        a = (x_mean * x_mean / snr_lin - b) / max(x_mean, 1e-8)
        points.append({"db": db, "a": max(a, 0.0), "b": b})
    return points


def mtf_scan_points() -> list[float]:
    """MTF 扫描：高斯核 σ（越大越糊）；核尺寸随 σ 自适应。"""
    return [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def mtf_kernel_size(sigma: float) -> int:
    """高斯核尺寸自适应：覆盖 ±3σ（截断误差 <0.1%），最小 5。"""
    size = 2 * math.ceil(3.0 * sigma) + 1
    return max(5, size)


def cloud_scan_points() -> list[float]:
    """云遮挡扫描：浓云覆盖率（0.0 = 干净）。"""
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


# ---------------------------------------------------------------------------
# 通用评估循环
# ---------------------------------------------------------------------------

def eval_degrade(
    model: torch.nn.Module,
    loader: Iterable,
    degrade_fn: Callable[[torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    value_range_check: bool = True,
) -> tuple[float, float]:
    """在退化函数作用下的测试集精度/F1（宏平均）。

    Args:
        model: 分类模型（forward 返回 logits (B, C)）。
        loader: 产生 dict 的 DataLoader，含 'image' 与 'label'。
        degrade_fn: 输入 (B,C,H,W) [0,1] 影像 → 输出退化影像（同形状）。
        device: 推理设备。
        value_range_check: 若影像值域 >1.5（如 0-255），先归一化再退化、退化后还原。

    Returns:
        (accuracy, macro_f1)
    """
    from sklearn.metrics import accuracy_score, f1_score

    all_pred, all_label = [], []
    for batch in loader:
        img = batch["image"].to(device)
        y = batch["label"].to(device)
        scale = 1.0
        if value_range_check and img.max() > 1.5:
            img = img / 255.0
            scale = 255.0
        deg = degrade_fn(img)
        if scale != 1.0:
            deg = deg * scale
        out = model(deg)
        if isinstance(out, dict):  # TerraTorch 兼容：取 logits 或首值
            out = out.get("logits", list(out.values())[0])
        all_pred.append(out.argmax(dim=1).cpu())
        all_label.append(y.cpu())
    pred = torch.cat(all_pred).numpy()
    lab = torch.cat(all_label).numpy()
    return accuracy_score(lab, pred), f1_score(lab, pred, average="macro")


def run_scan(
    model: torch.nn.Module,
    loader: Iterable,
    degrade_fn: Callable[[torch.Tensor], torch.Tensor],
    labels: Iterable[str],
    **eval_kwargs,
) -> list[dict]:
    """通用扫描：对 labels 中的每个退化点评估，返回结果字典列表。

    Args:
        model: 分类模型。
        loader: 测试 DataLoader（dict 格式）。
        degrade_fn: 接收 (退化参数, 影像) → 退化影像。
        labels: 退化参数序列（字符串/数值，用于 CSV）。
        eval_kwargs: 透传给 eval_degrade。

    Returns:
        [{'degrade': label, 'accuracy':..., 'f1':...}, ...]
    """
    results = []
    for lab in labels:
        # 默认参数绑定 lab，避免闭包延迟绑定陷阱
        acc, f1 = eval_degrade(model, loader, lambda im, lab=lab: degrade_fn(im, lab), **eval_kwargs)
        results.append({"degrade": lab, "accuracy": round(float(acc), 4), "f1": round(float(f1), 4)})
    return results
