# P-Flow: 正交三层架构 — 视频复现框架

通过逐层叠加正交信息（语义 → 结构 → 外观），让 T2V 模型重新生成与参考视频高度一致的版本。**零训练、模型无关**。

---

## 三层架构

| 层 | 技术 | 信息维度 | 核心操作 |
|---|------|---------|---------|
| L2 | SVD Noise Prior | 结构:"运动轨迹" | Inversion → 两阶段SVD → sigmoid自适应α混合 → 渐进多尺度 |
| L3 | Feature Injection | 外观:"纹理细节" | DiT 中间层 cross-attn 特征注入 + 三层自适应门控 |

> L1 (Prompt 优化) 为可选预处理，当前最优配置 **不依赖 L1**。

---

## Quick Start

### 1. 安装

```bash
cd P-Flow && pip install -r requirements.txt
```

### 2. 配置（只需一次）

编辑 `configs/default.toml`，修改模型和数据路径。默认已是最优参数，通常无需改动。

### 3. 运行

```bash
# 最优配置 (SVD + FI 三层门控)
python run.py --data_dir data/videos --caption_dir /path/to/captions

# 纯 baseline (无 SVD/FI)
python run.py --data_dir data/videos --caption_dir /path/to/captions --no-svd --no-feature-inject

# 指定样本
python run.py --data_dir data/videos --caption_dir /path/to/captions --sample_ids 7 99 50
```

所有参数默认值来自 `configs/default.toml`，命令行可覆盖任意值。

### 4. 评测

```bash
# CLIP + XCLIP (快速)
python -m evaluation.run_clip_xclip_eval \
    --orig-dir data/videos --gen-dir outputs/my_exp \
    --caption-dir /path/to/captions --output-dir outputs/my_exp/eval

# 全 7 项指标
python -m evaluation.run_all_metrics \
    --orig-dir data/videos --gen-dir outputs/my_exp \
    --caption-dir /path/to/captions --output-dir outputs/my_exp/eval
```

---

## 最优配置 (2026-07-07)

```toml
# configs/default.toml 核心内容
[noise_prior]
svd   = true       # 一键开启: inversion + SVD + blend
alpha = 0.004      # SVD 混合权重 (富运动数据集用 0.001)

[std_gate]
enabled = true
eta0    = 0.38     # sigmoid 自适应α 中点
kappa   = 20.0     # sigmoid 斜率

[progressive_svd]
enabled = true     # 自适应: k_m 非均匀时启用渐进多尺度

[fi]
enabled     = true
layers      = "mid"       # 注入中间层 (10~19)
lambda      = 0.10        # 基础FI强度 (短视频用 0.05)
schedule    = "middle_peak"
cache_mode  = "attention"

[fi.quality_gate]
enabled = true
k       = 20.0            # QS 斜率 (唯一可调)

[fi.adaptive_gate]
enabled = true
temp    = 5.0             # 余弦门控温度
high    = 0.30            # gate 上限 (None=不限)
```

### 结果 (1.3B, 200-case)

| 指标 | Baseline | P-Flow |
|------|:---:|:---:|
| XCLIP o-g | 0.689 | **0.758** (+10.0%) |
| CLIP o-g | 0.854 | **0.881** (+3.1%) |

---

## SVD 策略

```
η_inv → [全帧 SVD] → η_temporal_full
      → [渐进 SVD: 4窗×8帧, stride=4] → η_temporal → k_m 均匀则回退全帧
      → sigmoid自适应α(eta0=0.38, κ=20) → blend: √α·η_temporal + √(1-α)·η_random
```

## FI 策略

```
λ_max=0.10 → sin_schedule(middle_peak) → cosine_gate(τ=5.0) → QS(k=20.0)
三层门控: 中峰调度 × 余弦自适应 × 质量尺度
```

---

## 评估指标

| 指标 | 维度 | 方向 |
|------|------|:---:|
| CLIP o-g | 语义相似度 | ↑ |
| XCLIP o-g | 时空语义相似度 | ↑ |
| DINO o-g | 实例外观 | ↑ |
| LPIPS | 感知距离 | ↓ |
| SSIM | 结构相似度 | ↑ |
| OF-EPE | 运动一致性 | ↓ |
| FVD | 分布距离 | ↓ |

---

## 目录结构

```
P-Flow/
├── configs/default.toml          # 默认配置
├── run.py                         # CLI 入口
├── src/
│   ├── pipeline.py                # 统一管线
│   ├── flow_matching.py           # Flow Inverter
│   ├── svd_filter.py              # SVD 滤波 (渐进+自适应)
│   ├── vlm_client.py              # VLM 客户端
│   ├── video_utils.py             # 视频 I/O
│   ├── shot_detect.py             # 镜头检测
│   └── distributed.py             # GPU 推理
├── evaluation/
│   ├── run_clip_xclip_eval.py     # CLIP/XCLIP 评测
│   ├── run_all_metrics.py         # 7 项统一评测
│   └── clip_utils.py              # 共享工具
├── scripts/
│   └── rewrite_minimal.py         # L1 prompt 改写
├── docs/
│   └── 三层架构方法框架_修复版.md  # 方法论文档
└── requirements.txt
```

---

## 硬件

| 项目 | 规格 |
|------|------|
| GPU | A800 80GB / 4090 24GB |
| 模型 | Wan2.1-T2V-1.3B / 14B |
| 分辨率 | 480×832, 81 frames |
| 单样本耗时 | baseline ~30s / P-Flow ~250s |
