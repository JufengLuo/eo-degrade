# eo-degrade

**像质退化如何定量影响 AI 解译精度？——面向载荷指标论证的物理引导遥感实验**

> 身份定位：懂传感器物理的遥感 AI 工程师。
> 研究问题：GSD / SNR / MTF / 云遮挡 四类像质退化如何定量影响深度模型的解译精度？
> 响应曲线中是否存在可用于**载荷指标优化**的拐点（"甜区"）？

## 为什么做这个

遥感 AI 论文大多在"干净数据"上刷精度，但真实的卫星数据受制于：
- **GSD**（地面采样距离）：分辨率不够，小目标直接消失
- **SNR**（信噪比）：泊松（散粒）噪声 + 高斯（读出）噪声，σ² = a·I + b
- **MTF**（调制传递函数）：光学/运动模糊，MTF_motion = sinc(π·f·d)，像移量 d 来自平台微振动
- **云遮挡**：光学影像的天敌，真实世界的"缺失数据"

**载荷论证者问**："这个 GSD 指标够不够？SNR 提多少才值得花钱？"
**本仓库回答**：用定量实验给出 AI 解译精度对像质参数的**响应曲线**，把"指标 → 应用价值"的链条讲清楚。

## 仓库结构

```
eo-degrade/
├── configs/             # TerraTorch YAML 配置（EuroSAT 分类 / LEVIR-CD 变化检测）
├── degrade/             # 退化模拟核心库（gsd / snr / mtf / cloud）
├── notebooks/           # Kaggle 起步 notebook
├── scripts/             # 退化扫描脚本（四组实验）
├── results/             # 实验记录 CSV（可复现）
└── requirements.txt
```

## 快速开始（阶段 0，约 2 小时）

1. 注册 [Kaggle](https://www.kaggle.com)，打开 `notebooks/01_terratorch_eurosat_baseline.ipynb` 并复制到 Kaggle
2. 选择 GPU 加速器（免费额度），依次运行单元格
3. 得到 EuroSAT 分类 mIoU 基准（里程碑 M1）

## 四组退化实验（阶段 B）

| 实验 | 退化函数 | 扫描范围 | 产出 |
|---|---|---|---|
| GSD | `degrade.gsd.degrade_gsd` | 1.0×–8.0× | 精度-GSD 响应曲线 |
| SNR | `degrade.snr.add_poisson_gaussian` | 泊松 a / 高斯 b 扫描 | 精度-SNR 响应曲线 |
| MTF | `degrade.mtf.degrade_mtf` | σ 0.5–4.0 | 精度-MTF 响应曲线 |
| Cloud | `degrade.cloud.add_cloud_mask` | 遮挡率 10%–70% | 精度-遮挡率响应曲线 |

### 云退化物理模型（v2，云团模型）

`degrade/cloud.py` 实现了物理合理的云遮挡模拟，取代早期"随机像素"近似：

1. **云场**：云团中心泊松分布，超高斯径向剖面 `exp(-(d/σ)^1.6)`（边缘平缓如真实云），内部叠加小幅高频纹理
2. **透过率**：比尔-朗伯定律 `opacity = 1 - exp(-τ)`——浓云（积云）完全遮挡，薄云（层云）半透明调制
3. **云影**：云掩膜沿太阳方位平移 + 膨胀、扣除云体自身区域，阴影区亮度按 `shadow_atten` 衰减
4. **精确可控**：浓云覆盖率用分位数阈值锁定（`cloud_fraction` 与实测缺失率误差 <2%），`seed` 完全可复现

```python
from degrade.cloud import add_cloud_field, add_cloud_mask

# 完整信息：退化影像、缺失掩膜、不透明度场（软标签）、云影掩膜
deg, mask, opacity, shadow = add_cloud_field(x, cloud_fraction=0.3, seed=0,
                                             cloud_type='cumulus', shadow=True)

# 兼容旧接口：扫描实验用
deg, mask = add_cloud_mask(x, cloud_fraction=0.3, seed=0)
```

退化效果示例：`results/cloud_degradation_example.png`（原图 / 积云 / 层云 / 云影对比）。

## 可证伪假设

- **H1**：精度随 GSD 劣化单调下降，存在"甜区"（GSD 提升超过某点后收益递减）
- **H2**：降噪对 AI 解译的收益存在饱和（SNR 超阈值后无增益）
- **H3**：MTF 退化对高频任务（变化检测边界）伤害大于低频任务（地物分类）
- **H4**：云遮挡缺失率与精度为非线性关系，存在可定位的失效拐点

**阴性结果同样有价值**："AI 精度对 GSD 的响应是平滑的，不存在可用于指标优化的拐点"——这句话对载荷论证者来说，比发现拐点更值钱。

## 云遮挡扫描（notebook 02）

- `notebooks/02_cloud_degradation_scan.ipynb`：在 M1 模型上做 0%–70% 遮挡率扫描，输出 CSV + 响应曲线图
- `scripts/analyze_response.py`：H4 拐点量化判定（加速下降拐点 + 失效点），`python scripts/analyze_response.py --csv results/cloud_scan.csv`
- `scripts/test_cloud.py`：退化库自测（形状/遮挡率精度/可复现性/云影/薄云/边界/旧接口兼容），`python scripts/test_cloud.py`

## 其余三组扫描（notebook 03，H1/H2/H3）

- `notebooks/03_gsd_snr_mtf_scan.ipynb`：GSD（1.0–8.0×）/ SNR（30→6 dB）/ MTF（σ 0.5–4.0）三组扫描，输出三份 CSV + 曲线图
- `scripts/scan_degradation.py`：通用扫描引擎——四组扫描点生成 + 通用评估循环（值域自适应、TerraTorch 输出兼容）+ SNR 反解
- **SNR 扫描物理口径**：目标 SNR(dB) 功率比 10·log10(m²/(a·m+b))，泊松系数 a 由数据均值反解、读出噪声 b 固定，保证横轴均匀可复现
- **MTF 核自适应**：高斯核尺寸随 σ 自动取 ±3σ（截断误差 <0.1%）
- `scripts/test_scan.py`：扫描引擎自测（假模型+合成数据），`python scripts/test_scan.py`

## 技术栈

- [TerraTorch](https://github.com/terrastackai/terratorch)：ESA 开源的地理空间基础模型微调工具（YAML 即训练配置）
- [TorchGeo](https://github.com/microsoft/torchgeo)：地理空间数据加载（EuroSAT 等基准数据集）
- 基础模型：Prithvi-EO 系列（NASA/IBM）/ DOFA

## 路线

- [x] 阶段 0：Kaggle notebook 跑通（M1 环境可复现）——EuroSAT 89.4%（2026-09）
- [ ] 阶段 A：EuroSAT mIoU + LEVIR-CD F1 双基准（mIoU 待补）
- [ ] 阶段 B：四组退化扫描 → 响应曲线（退化库与扫描脚本就绪，待 Kaggle 运行）
  - [x] 云退化库 v2（云团模型 + 比尔-朗伯 + 云影，本地自测 12 项全绿）
  - [x] 扫描引擎 scripts/scan_degradation.py + 02/03 notebook
  - [ ] 02 云扫描、03 GSD/SNR/MTF 扫描在 Kaggle 运行并回填结果
- [ ] 阶段 C：P1 云遮挡鲁棒改进 + P2 校准
- [ ] 阶段 D：结果可视化 + README 故事化 + 求职作品

## 许可与致谢

MIT License。感谢 TerraTorch / TorchGeo / Prithvi-EO 开源生态。
