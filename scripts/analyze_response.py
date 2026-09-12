"""响应曲线拐点分析：H4（云遮挡失效拐点）的量化判定工具。

物理/工程问题：载荷论证者想知道"遮挡率到什么程度，AI 解译精度开始加速崩坏"。
两类判定（互补）：
1. breakpoint（加速下降拐点）：相邻遮挡率步进间，精度下降速率首次超过阈值
   → "从这一点开始，每多 10% 云的代价显著增大"
2. failure_point（失效点）：精度首次跌破 干净精度 × keep_ratio
   → "到这个遮挡率，解译结果已不可信"

用法（本地可测，不依赖训练环境）：
    python scripts/analyze_response.py --csv results/cloud_scan.csv
或作为库：
    from analyze_response import find_breakpoint, find_failure_point
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def find_breakpoint(
    fractions: list[float],
    accs: list[float],
    min_drop: float = 0.02,
) -> tuple[float | None, float | None]:
    """加速下降拐点：单步精度下降首次 >= min_drop 的遮挡率。

    Returns:
        (break_fraction, step_drop)；无拐点返回 (None, None)。
    """
    for i in range(1, len(accs)):
        drop = accs[i - 1] - accs[i]
        if drop >= min_drop:
            return fractions[i], drop
    return None, None


def find_failure_point(
    fractions: list[float],
    accs: list[float],
    keep_ratio: float = 0.7,
) -> tuple[float | None, float | None]:
    """失效点：精度首次跌破 干净精度×keep_ratio 的遮挡率。

    Returns:
        (fail_fraction, fail_acc)；全程未跌破返回 (None, None)。
    """
    if not accs:
        return None, None
    baseline = accs[0]
    for frac, acc in zip(fractions, accs):
        if acc < baseline * keep_ratio:
            return frac, acc
    return None, None


def load_csv(path: str | Path) -> tuple[list[float], list[float]]:
    """读取扫描结果 CSV：列名需含 cloud_fraction 与 accuracy（大小写不敏感）。"""
    fractions, accs = [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV 为空: {path}")
    key_f = next(k for k in rows[0] if k.strip().lower() == "cloud_fraction")
    key_a = next(k for k in rows[0] if k.strip().lower() == "accuracy")
    for r in rows:
        fractions.append(float(r[key_f]))
        accs.append(float(r[key_a]))
    return fractions, accs


def main() -> None:
    ap = argparse.ArgumentParser(description="响应曲线拐点分析（H4）")
    ap.add_argument("--csv", required=True, help="扫描结果 CSV 路径")
    ap.add_argument("--min-drop", type=float, default=0.02, help="加速下降阈值")
    ap.add_argument("--keep-ratio", type=float, default=0.7, help="失效判定比例")
    args = ap.parse_args()

    fractions, accs = load_csv(args.csv)
    bf, drop = find_breakpoint(fractions, accs, min_drop=args.min_drop)
    ff, fa = find_failure_point(fractions, accs, keep_ratio=args.keep_ratio)

    print(f"扫描点: {len(fractions)} 个，遮挡率 {fractions[0]:.0%} → {fractions[-1]:.0%}")
    print(f"干净精度: {accs[0]:.3f}")
    if bf is not None:
        print(f"[H4] 加速下降拐点: 遮挡率 {bf:.0%}（该步下降 {drop:.3f}）")
    else:
        print(f"[H4] 未检测到 >= {args.min_drop:.0%} 的加速下降步（曲线平滑下降）")
    if ff is not None:
        print(f"[H4] 失效点: 遮挡率 {ff:.0%}（精度 {fa:.3f} < 干净×{args.keep_ratio:.0%}）")
    else:
        print(f"[H4] 全程未跌破 干净×{args.keep_ratio:.0%}（鲁棒）")


if __name__ == "__main__":
    main()
