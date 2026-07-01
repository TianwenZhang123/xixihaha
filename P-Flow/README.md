# P-Flow: 正交三层架构 — 视频复现框架

通过逐层叠加正交信息（语义 → 结构 → 外观），让 T2V 模型重新生成与参考视频高度一致的版本。**零训练、模型无关**。

---

## 三层架构

| 层 | 技术 | 信息维度 | 核心操作 |
|---|------|---------|---------|
| L2 | SVD Noise Prior | 结构："运动轨迹" | Inversion → 渐进多尺度 SVD → 双向门控混合 |
| L3 | Feature Injection | 外观："纹理细节" | DiT 中间层 cross-attn 特征注入 + 四层自适应门控 |

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
# 最优配置 (ABC + 自适应渐进 SVD + FI 四层门控)
python run.py --data_dir data/videos --caption_dir /path/to/captions

# 纯 baseline (无 SVD/FI)
python run.py --data_dir data/videos --caption_dir /path/to/captions --no-inversion --no-svd

# 指定样本
python run.py --data_dir data/videos --caption_dir /path/to/captions --sample_ids 7 99 50

# 换 14B 模型
python run.py --data_dir data/videos --caption_dir /path/to/captions \
    --model_path ~/autodl-tmp/model/Wan2.1-T2V-14B-Diffusers
```

所有参数默认值来自 `configs/default.toml`，命令行可覆盖任意值。

### 4. 评测

```bash
# CLIP + XCLIP (快速)
python -m evaluation.run_clip_xclip_eval \
    --orig-dir data/videos --gen-dir outputs/my_exp \
    --caption-dir /path/to/captions --output-dir outputs/my_exp/eval

# 全 6 项指标
python -m evaluation.run_all_metrics \
    --orig-dir data/videos --gen-dir outputs/my_exp \
    --caption-dir /path/to/captions --output-dir outputs/my_exp/eval_all \
    --skip-fvd
```

---

## 最优配置 (2026-07-01)

```toml
# configs/default.toml 核心内容
[noise_prior]
inversion = true
svd       = true
blend     = true
alpha     = 0.004

[progressive_svd]
enabled = true   # 自适应: k_m 非均匀时启用渐进多尺度

[std_gate]
enabled     = true
low         = 0.32
floor_alpha = 0.006
high        = 0.45
cap_alpha   = 0.002

[fi]
enabled     = true
layers      = "mid"
lambda      = 0.10
schedule    = "middle_peak"
cache_mode  = "attention"

[fi.quality_gate]
skip_threshold = 0.08
skip_svd       = true

[fi.adaptive_gate]
enabled = true
temp    = 5.0
high    = 0.30

[fi.norm_budget]
max_norm  = 10000
decay_min = 0.3
```

### 结果 (1.3B, 200-case)

| 指标 | Baseline | ABC + 自适应渐进SVD |
|------|:---:|:---:|
| XCLIP orig-gen | 0.692 | **0.741** |
| CLIP orig-gen | 0.857 | 0.880 |

---

## SVD 策略

```
η_inv → [全帧 SVD] → η_temporal_full (门控判断)
      → [渐进 SVD: 4窗×8帧, stride=4] → η_temporal (blend)
         → k_m 均匀 → 回退全帧 SVD
         → k_m ≤ 2  → 跳过该窗口
      → α=0.004 + 双向门控 (Floor 0.32 / CAP 0.45)
```

## FI 策略

```
λ_max=0.10 → Quality Gate → Schedule(middle_peak) → Adaptive Gate(cap 0.30) → 累计预算(10000)
门控用全帧 η_temporal_full 的 mean_cos
```

---

## 评估指标

| 指标 | 维度 | 方向 |
|------|------|:---:|
| CLIP o-g | 语义相似度 | ↑ |
| XCLIP o-g | 时空语义相似度 | ↑ |
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
│   ├── prompt_decompose.py        # L1 Prompt 优化 (可选)
│   ├── video_utils.py             # 视频 I/O
│   └── distributed.py             # GPU 推理
├── evaluation/
│   ├── run_clip_xclip_eval.py     # CLIP/XCLIP 评测
│   ├── run_all_metrics.py         # 6 项统一评测
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
| 单样本耗时 | baseline ~30s / ABC ~250s |
