"""degrade.cloud 本地自测（合成数据，无 GPU 依赖）。

运行: python scripts/test_cloud.py
覆盖: 形状 / 遮挡率精度 / 可复现性 / 云影 / 薄云 / 边界 / 向后兼容
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from degrade.cloud import add_cloud_field, add_cloud_mask  # noqa: E402

ok = True


def check(name: str, cond: bool, extra: str = "") -> None:
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {name} {extra}")


def main() -> None:
    torch.manual_seed(0)
    x = torch.rand(2, 3, 64, 64)  # (B,C,H,W)

    # 1) 形状与值域
    deg, mask, opacity, sh = add_cloud_field(x, cloud_fraction=0.3, seed=1)
    check("输出形状", deg.shape == x.shape and mask.shape == (2, 1, 64, 64)
          and opacity.shape == mask.shape and sh.shape == mask.shape)
    check("值域 [0,1]", deg.min() >= 0.0 and deg.max() <= 1.0
          and opacity.min() >= 0.0 and opacity.max() <= 1.0)

    # 2) 遮挡率精度（无阴影时 mask 覆盖率 ≈ cloud_fraction）
    for frac in (0.1, 0.3, 0.5, 0.7):
        _, m, _, _ = add_cloud_field(x, cloud_fraction=frac, seed=7, shadow=False)
        actual = m.mean().item()
        check(f"遮挡率 {frac:.0%} 精度", abs(actual - frac) < 0.02, f"实际={actual:.3f}")

    # 3) 可复现性
    d1, m1, _, _ = add_cloud_field(x, cloud_fraction=0.3, seed=42)
    d2, m2, _, _ = add_cloud_field(x, cloud_fraction=0.3, seed=42)
    check("可复现性（影像）", torch.equal(d1, d2))
    check("可复现性（掩膜）", torch.equal(m1, m2))

    # 4) 云影：开启时缺失率 > 仅浓云；阴影掩膜非空
    _, m_no_sh, _, sh_no = add_cloud_field(x, cloud_fraction=0.3, seed=3, shadow=False)
    _, m_sh, _, sh_yes = add_cloud_field(x, cloud_fraction=0.3, seed=3, shadow=True)
    check("云影增加缺失区", m_sh.mean() > m_no_sh.mean() + 1e-3,
          f"无影={m_no_sh.mean():.3f} 有影={m_sh.mean():.3f}")
    check("阴影掩膜非空", sh_yes.sum() > 0 and sh_no.sum() == 0)

    # 5) 薄云（stratus）：浓云区域内的不透明度应明显低于积云（半透明）
    _, m_cum, op_cum, _ = add_cloud_field(x, cloud_fraction=0.5, seed=5, cloud_type="cumulus", shadow=False)
    _, m_str, op_str, _ = add_cloud_field(x, cloud_fraction=0.5, seed=5, cloud_type="stratus", shadow=False)
    core_cum = op_cum[m_cum > 0.5].mean().item()
    core_str = op_str[m_str > 0.5].mean().item()
    check("薄云浓云区 opacity 更低", core_str < core_cum - 0.1,
          f"cumulus={core_cum:.3f} stratus={core_str:.3f}")

    # 6) 边界：0% / 100%
    _, m0, _, _ = add_cloud_field(x, cloud_fraction=0.0, seed=1, shadow=False)
    _, m1, _, _ = add_cloud_field(x, cloud_fraction=1.0, seed=1, shadow=False)
    check("0% 无遮挡", m0.sum() == 0)
    check("100% 全遮挡", m1.mean() > 0.99, f"实际={m1.mean():.3f}")

    # 7) 向后兼容：add_cloud_mask 旧签名
    d, m = add_cloud_mask(x, cloud_fraction=0.3, seed=9)
    check("旧接口兼容", d.shape == x.shape and m.shape == (2, 1, 64, 64)
          and torch.all((m == 0) | (m == 1)))

    # 8) GPU（若有）
    if torch.cuda.is_available():
        xg = x.cuda()
        dg, mg, _, _ = add_cloud_field(xg, cloud_fraction=0.3, seed=1)
        check("GPU 兼容", dg.device.type == "cuda")
    else:
        print("[SKIP] 无 GPU，跳过设备兼容测试")

    print()
    print("全部通过 ✅" if ok else "存在失败 ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
