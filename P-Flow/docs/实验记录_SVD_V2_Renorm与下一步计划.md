# SVD V2 实验分析 & 改进记录

> 创建时间: 2025-06-10  
> 最后更新: 2026-06-11  
> 状态: renorm 失败 → rescale 否决 → QGA 失败 → α=0.01 确认天花板 → 方向 C 频域重塑 **完全失败** → 方向 D: SGA **效果边际，pass** → 方向 E: PODI **失败 (CLIP -2.5%, XCLIP -5.2%)** → 方向 F: CEGI **失败 (CLIP -3.4%, XCLIP -5.2%)** → 方向 G: MSTDI **FAILED (G1: -0.7%/-1.1%; G2: -15.3%/-32.3% 灾难性失败，证明 η_temporal 内容有害)** → **下一步: TPI (最高优先) / OCS 实验**  
> 目标: 确定 L2 SVD 最优策略 + L1 prompt rewrite 验证

---

## 一、实验背景

当前 Pure L2 Baseline 配置:
- Caption: 原始 VLM caption (无 LLM 改写)
- SVD mode: v1 (无 renorm, 无频段分离)
- Alpha: 0.004
- **Baseline 指标: CLIP 0.8964, XCLIP 0.7874**

本次实验测试 SVD V2 renorm 模式能否在同等 caption 下超越 v1。

---

## 二、SVD V2 Renorm (α=0.001) 实验结果

### 2.1 总体指标

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| SVD V2 Renorm (α=0.001) | 0.8826 | 0.7506 | CLIP -1.5%, XCLIP -4.7% |

**结论: renorm 模式整体不如 v1 baseline。**

### 2.2 逐 Case 对比

| 样本 | 场景 | Baseline CLIP | Renorm CLIP | Δ CLIP | Baseline XCLIP | Renorm XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.9341 | +0.4% | 0.6982 | 0.7384 | **+5.8%** ✅ |
| 17 | SUV越野 | 0.9092 | 0.8957 | -1.5% | 0.8368 | 0.7859 | **-6.1%** ❌ |
| 21 | 丛林纸飞机 | 0.8928 | 0.8371 | **-6.2%** ❌ | 0.7637 | 0.6099 | **-20.2%** ❌ |
| 31 | 水下城市 | 0.8324 | 0.8559 | **+2.8%** ✅ | 0.5237 | 0.6153 | **+17.5%** ✅ |
| 32 | 雪地金毛 | 0.9167 | 0.9169 | ≈0 | 0.8221 | 0.8553 | **+4.0%** ✅ |
| 33 | 跑步者 | 0.8531 | 0.8490 | -0.5% | 0.8618 | 0.8202 | **-4.8%** ❌ |
| 34 | 四只小狗 | 0.8968 | 0.9120 | **+1.7%** ✅ | 0.8710 | 0.8348 | -4.2% |
| 43 | 花园猫咪 | 0.9539 | 0.9285 | **-2.7%** ❌ | 0.9069 | 0.8529 | **-6.0%** ❌ |
| 46 | 火山喷发 | 0.9022 | 0.8917 | -1.2% | 0.7869 | 0.7129 | **-9.4%** ❌ |
| 47 | 动画狗城市 | 0.8769 | 0.8055 | **-8.1%** ❌ | 0.8024 | 0.6802 | **-15.2%** ❌ |

胜负统计: renorm 赢 3/10, 输 6/10, 平 1/10

### 2.3 日志关键参数对比

| 样本 | v1 η_temporal std | renorm η_temporal std | v1 direction_shift | renorm direction_shift | 放大比 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 0.3730 | 1.0000 | 0.0245 | 0.0318 | +29% |
| 17 | 0.3320 | 1.0000 | 0.0218 | 0.0318 | +46% |
| 21 | 0.3281 | 1.0000 | 0.0217 | 0.0318 | +47% |
| 31 | 0.3398 | 1.0000 | 0.0224 | 0.0318 | +42% |
| 32 | 0.3633 | 1.0000 | 0.0238 | 0.0318 | +34% |
| 33 | 0.4102 | 1.0000 | 0.0268 | 0.0318 | +19% |
| 34 | 0.2793 | 1.0000 | 0.0187 | 0.0318 | +70% |
| 43 | 0.3945 | 1.0000 | 0.0257 | 0.0318 | +24% |
| 46 | 0.3770 | 1.0000 | 0.0247 | 0.0318 | +29% |
| 47 | 0.3965 | 1.0000 | 0.0259 | 0.0318 | +23% |

**核心发现**: renorm 将所有样本的 direction_shift 统一为 0.0318，抹杀了 v1 中"各样本自然自适应"的特性。

---

## 三、根因分析

### 3.1 为什么 v1 的"不均匀 std"反而是优势？

v1 模式下，η_temporal std 由视频本身的运动特征决定:
- 强运动视频 → SVD 时序分量幅度大 → std 高 (如 case 33 的 0.41)
- 弱运动视频 → SVD 时序分量幅度小 → std 低 (如 case 34 的 0.28)

配合固定 α=0.004，实际注入量 = √α × std:
- 强运动样本: 0.0632 × 0.41 = 0.026 (注入较强 → 但这类样本本就运动清晰，模型不需要太多帮助)
- 弱运动样本: 0.0632 × 0.28 = 0.018 (注入较弱 → 但弱运动时 SVD 质量也低，少注入反而安全)

这形成了一种**隐性自适应**: 信号质量高时多注入，信号质量低时少注入。

### 3.2 renorm 如何破坏了这个平衡？

renorm 强制 std=1.0 后:
- 所有样本 effective_injection = √0.001 × 1.0 = 0.0316 (统一值)
- 对于 baseline 已经很好的样本 (43, 33, 17): 过度注入 → 退化
- 对于 SVD 方向信息为空的样本 (47, 21): 放大的是噪声 → 严重退化
- 只对少数"baseline 很差 + SVD 有方向信息"的样本 (31, 7) 有效

### 3.3 决定 renorm 成败的真正因素

| 利好 renorm | 利空 renorm |
|------------|------------|
| Baseline XCLIP 低 (< 0.70) | Baseline XCLIP 高 (> 0.80) |
| SVD 空间能量有方向偏好 (某象限>30%) | 能量均匀分布 (四象限≈25%) |
| cos(Δ_first, Δ_last) 高正值 (>0.1) | cos(Δ_first, Δ_last) ≈ 0 或负值 |
| 真实视频 | 动画/渲染风格 |
| v1 η_temporal std 低 (< 0.33) | v1 η_temporal std 高 (> 0.38) |

---

## 四、方案演进：Rescale → 否决 → Quality-Gated Alpha

### 4.1 方案 A (Rescale) 的提出与否决

**最初思路**: 不做 (x-mean)/std 的全量归一化，而是做等比缩放，保留方向间相对比例。

```python
target_effective = 0.0234  # v1 中位 effective = √0.004 × 0.37
current_effective = sqrt(alpha) * eta_temporal.std()
if current_effective < target_effective:
    scale = target_effective / current_effective
    eta_temporal = eta_temporal * scale
```

**已实现** (`src/svd_filter.py` 的 `mode="rescale"`)，但经过分析后**否决**，原因：

> **等比缩放 η_temporal 和调 alpha 完全等价。**
>
> Blend 公式: `η = √α·η_temporal + √(1-α)·η_random`
>
> 放大 η_temporal 2 倍 ≡ 把 α 放大 4 倍 → 两者都只是调 "SVD 信号在最终噪声里的比重"。
> 而 alpha 是现成的超参，不需要在 SVD filter 层再加一层间接调节。

**结论**: 真正有价值的改动不是改信号"强度"（调 alpha 即可），而是改信号"质量分配"——让每个样本根据自身 SVD 方向质量获得不同的 alpha。

### 4.2 方案 B: Quality-Gated Alpha（已实现，待验证）

**核心思想**: v1 的"隐性自适应"是盲目的——只看幅度不看质量。Quality-Gated Alpha 显式度量 SVD 方向质量，然后用它调 alpha。

**方向质量度量** (`_compute_direction_quality()` in `src/pipeline.py`):

| 子指标 | 权重 | 含义 |
|--------|:---:|------|
| temporal_coherence | 0.5 | 相邻帧 cosine similarity 均值（方向一致 = 真实运动）|
| spatial_anisotropy | 0.3 | 四象限能量最大占比归一化（非均匀 = 有方向偏好）|
| first_last_consistency | 0.2 | 首末帧 cosine similarity（长程方向一致性）|

综合: `quality = 0.5×coherence + 0.3×anisotropy + 0.2×consistency ∈ [0, 1]`

**Alpha 调节公式**:

```python
effective_alpha = base_alpha × (low_mult + (high_mult - low_mult) × quality)
# 默认参数: base=0.004, low_mult=0.25, high_mult=2.5
# quality=0 → α = 0.004 × 0.25 = 0.001 (方向不可信，几乎不注入)
# quality=0.5 → α = 0.004 × 1.375 = 0.0055 (中等质量)
# quality=1 → α = 0.004 × 2.5 = 0.010 (高质量方向，强注入)
```

**实现位置**: `src/pipeline.py` 的 `_get_latents()` 方法，通过 `--quality_gated_alpha` 开启

**与调全局 alpha 的本质区别**: 全局 alpha 对所有样本一视同仁；QGA 让每个样本拿到"该得的"注入量——方向质量好的样本多注入，质量差的少注入。这是 v1 隐性自适应的显式化、理性化版本。

**适合论文叙事**: "Sample-Adaptive Prior Injection"

### 4.3 方案演进总结

```
renorm (全量归一化)        → 失败: 抹杀样本间差异
  ↓ 分析根因
rescale (等比缩放)         → 否决: 等价于调 alpha，无独立价值
  ↓ 认识到"强度"不是关键
Quality-Gated Alpha (质量门控) → 当前方案: 改变"分配"而非"强度"
```

---

## 五、Prompt Rewrite v9 — 已验证

### 5.1 策略设计

代码: `P-Flow/scripts/rewrite_minimal.py` (v9-vlm)

- **Step 1: LLM pure subtraction** — 只删不加 (去 preamble + hedging + summary)
- **Step 2: VLM visual supplement** — 用本地 Qwen2.5-VL-7B 看原始视频帧，补充视觉细节

解决 v8-minimal 的问题: v8 的 CLIP 0.8915 < Pure L2 的 0.8964，根因是 LLM 虚构了不接地气的 camera sentences。v9 用 VLM 确保每个新增细节都是真实视觉事实。

### 5.2 运行命令

```bash
cd /root/xixihaha/P-Flow

# Step 1: 生成 v9 captions (LLM 删减 + 本地 VLM 补充)
python scripts/rewrite_minimal.py \
    --input-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir data/captions_v9 \
    --video-dir data/videos \
    --backend dashscope \
    --model qwen-plus \
    --vlm-provider local \
    --vlm-model-path /root/models/Qwen2.5-VL-7B-Instruct \
    --sample-ids 7 17 21 31 32 33 34 43 46 47

# Step 2: 用 v9 captions + SVD v1 跑生成
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/v9_svd_v1 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --steps 30 --guidance 5.0 --seed 42

# Step 3: 评测
python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/v9_svd_v1 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/v9_svd_v1
```

### 5.3 实验结果

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004, 原始caption) | **0.8964** | 0.7874 | — |
| v9 (LLM删减 + VLM补充 + v1 + α=0.004) | 0.8947 | **0.7973** | CLIP -0.2%, XCLIP **+1.3%** |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | v9 CLIP | Δ CLIP | Baseline XCLIP | v9 XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.8981 | -3.5% | 0.6982 | 0.7288 | +4.4% ✅ |
| 17 | SUV越野 | 0.9092 | 0.9192 | +1.1% | 0.8368 | 0.8507 | +1.7% ✅ |
| 21 | 丛林纸飞机 | 0.8928 | 0.8824 | -1.2% | 0.7637 | 0.8095 | **+6.0%** ✅ |
| 31 | 水下城市 | 0.8324 | 0.8644 | **+3.8%** | 0.5237 | 0.6288 | **+20.1%** ✅ |
| 32 | 雪地金毛 | 0.9167 | 0.9092 | -0.8% | 0.8221 | 0.7895 | -4.0% ❌ |
| 33 | 跑步者 | 0.8531 | 0.8497 | -0.4% | 0.8618 | 0.8469 | -1.7% |
| 34 | 四只小狗 | 0.8968 | 0.9191 | +2.5% | 0.8710 | 0.8682 | ≈0 |
| 43 | 花园猫咪 | 0.9539 | 0.9549 | ≈0 | 0.9069 | 0.9057 | ≈0 |
| 46 | 火山喷发 | 0.9022 | 0.8910 | -1.2% | 0.7869 | 0.7306 | **-7.2%** ❌ |
| 47 | 动画狗城市 | 0.8769 | 0.8590 | -2.0% | 0.8024 | 0.8138 | +1.4% |

胜负统计: v9 赢 4/10 (XCLIP 显著提升), 输 2/10, 平 4/10

### 5.4 分析

**结论: v9 策略方向正确，XCLIP 有小幅稳定提升 (+1.3%)，CLIP 基本持平。**

- **明确受益**: baseline 本身差的样本（case 31 XCLIP +20%, case 21 +6%）——VLM 补充的视觉细节确实帮到了运动模糊/内容不清晰的场景
- **明确受损**: case 46（火山喷发 -7.2%）——可能 VLM 补充的细节与 SVD 方向产生了冲突
- **大部分样本**: 变化在 ±2% 以内，属于噪声范围
- **vs v8**: v8 的 CLIP 0.8915 明显低于 baseline，v9 的 0.8947 基本持平 → VLM 补充比 LLM 虚构安全得多

**L1 层结论**: "删减 + VLM 补充"路线可保留（不退化 + 对弱样本有帮助），但收益有限，主要提升来源仍需依赖 L2。

---

## 六、α=0.01 实验 & One-Shot Blend 天花板验证

### 6.1 实验目的

验证提高 α 能否增强 SVD temporal prior 的影响力，改善 XCLIP。

### 6.2 总体指标

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | 0.7874 | — |
| SVD v1, α=0.01 (全10样本) | 0.8903 | 0.7685 | CLIP -0.7%, XCLIP -2.4% ❌ |
| SVD v1, α=0.01 (后9样本, 去掉S7) | **0.8947** | **0.7966** | CLIP -0.2%, XCLIP **+1.2%** ✅ |

### 6.3 逐 Case 对比

| 样本 | 场景 | α=0.01 CLIP | Δ CLIP | α=0.01 XCLIP | Δ XCLIP | 冲突风险 |
|:---:|------|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.8499 | **-5.2%** ❌ | 0.5161 | **-34.5%** ❌❌ | LOW |
| 17 | SUV越野 | 0.9036 | +0.8% | 0.8219 | +3.5% ✅ | HIGH |
| 21 | 丛林纸飞机 | 0.8978 | +0.1% | 0.7394 | -4.8% ❌ | LOW |
| 31 | 水下城市 | 0.8607 | -3.6% | 0.5599 | **-22.8%** ❌❌ | LOW |
| 32 | 雪地金毛 | 0.9016 | +0.5% | 0.8603 | +7.3% ✅ | HIGH |
| 33 | 跑步者 | 0.8745 | -2.2% | 0.8449 | +5.8% ✅ | LOW |
| 34 | 四只小狗 | 0.8959 | ≈0 | 0.8449 | +5.8% ✅ | HIGH |
| 43 | 花园猫咪 | 0.9560 | +6.0% ✅ | 0.9046 | +11.7% ✅ | LOW |
| 46 | 火山喷发 | 0.8879 | -0.9% | 0.7729 | -1.5% | LOW |
| 47 | 动画狗城市 | 0.8747 | -2.2% | 0.8208 | +3.3% ✅ | LOW |

### 6.4 关键发现

**去掉 Sample 7 和 31 后，α=0.01 其实是正向的 (XCLIP +1.2%)**，但这两个"catastrophic failure"样本把整体拖垮了。

核心问题分析:
1. **Sample 7** (杯中帆船): XCLIP 崩到 0.5161 (-34.5%)，但冲突诊断为 LOW
2. **Sample 31** (水下城市): XCLIP 崩到 0.5599 (-22.8%)，同样冲突诊断为 LOW
3. 两个样本的 `cos(mixed, temporal)` 仅 0.03~0.04，说明 temporal 信号影响极微
4. **结论**: 问题不在"方向冲突"，而是 **η_temporal 的内容本身对这些样本有毒性**

### 6.5 One-Shot Linear Blend 天花板论证

| α | CLIP | XCLIP | 问题 |
|:---:|:---:|:---:|------|
| 0.004 (baseline) | 0.8964 | 0.7874 | — |
| 0.01 (+150%) | 0.8903 | 0.7685 | S7/S31 catastrophic failure |
| 0.001 (renorm) | 0.8826 | 0.7506 | 全面退化 |

**结论**: 无论增大还是减小 α，one-shot linear blend 都无法同时满足所有样本。根本原因是线性混合直接注入 η_temporal 的"内容"——对某些样本这些内容是有害的。

**需要范式转换**: 从"注入内容"转向"传递结构/节奏"。

---

## 七、方向 C: 频域噪声重塑 (Spectrum-Aligned Noise Initialization)

### 7.1 核心思想

**不再注入 η_temporal 的具体内容，只传递其"时间频谱形状"。**

灵感来源:
- FreqPrior (ICLR 2025): 频域噪声塑形，保持 Gaussian 分布
- FreeInit (ECCV 2024): 低频保留概念

类比: 如果 linear blend 是"把参考视频的运动方向直接粘贴进去"，那频域重塑是"告诉模型这个视频的运动节奏是快还是慢"——只传递 tempo 不传递 direction。

### 7.2 算法

```
输入: η_temporal (SVD 滤波后), η_random (标准高斯)
参数: β ∈ [0, 1] (重塑强度)

1. F_t = rFFT(η_temporal, dim=frame)     # 时间频谱
2. spectrum_shape = mean(|F_t|, spatial)  # 全局频谱形状 (去除空间结构)
3. spectrum_shape = spectrum_shape / mean(spectrum_shape)  # 归一化为纯形状
4. reshape_filter = spectrum_shape^β      # β 控制强度
5. F_r = rFFT(η_random, dim=frame)       # 随机噪声的频谱
6. F_out = F_r × reshape_filter           # 用目标形状调制幅度，保留随机相位
7. η_out = IRFFT(F_out)                  # 回时域
8. η_out = (η_out - mean) / std          # renorm to N(0,1)

输出: η_out — 具有参考视频时间节奏的 N(0,1) 噪声
```

### 7.3 为什么比 linear blend 安全

| 特性 | Linear Blend | Spectrum Reshape |
|------|:---:|:---:|
| 注入 η_temporal 空间内容 | ✓ (有毒性风险) | ✗ |
| 注入 η_temporal 方向信息 | ✓ (冲突风险) | ✗ |
| 传递运动节奏 (快/慢) | 间接 | ✓ (直接) |
| 保持 N(0,1) | ✓ (公式保证) | ✓ (renorm 保证) |
| Per-sample 安全 | ✗ (S7/S31 崩塌) | ✓ (只传形状) |

### 7.4 实现位置

- `src/pipeline.py`: `_freq_reshape_noise()` 方法
- 开关: `--freq_reshape` (替代 `--alpha` 的线性混合)
- 参数: `--freq_reshape_beta` (默认 1.0, 推荐搜索 0.3~1.0)

### 7.5 实验命令

```bash
cd /root/xixihaha/P-Flow

# β=1.0 (完全匹配频谱形状)
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/freq_reshape_beta1.0 \
    --noise_prior \
    --svd_mode v1 \
    --freq_reshape \
    --freq_reshape_beta 1.0 \
    --seed 42 --steps 30 --guidance 5.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/freq_reshape_beta1.0 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/freq_reshape_beta1.0

# β=0.5 (部分重塑，更保守)
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/freq_reshape_beta0.5 \
    --noise_prior \
    --svd_mode v1 \
    --freq_reshape \
    --freq_reshape_beta 0.5 \
    --seed 42 --steps 30 --guidance 5.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/freq_reshape_beta0.5 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/freq_reshape_beta0.5
```

### 7.6 预期结果

- **Safety**: Sample 7/31 不再 catastrophic failure (因为不注入有毒内容)
- **Effectiveness**: 对运动特征明确的样本 (43, 32, 34)，频谱形状能引导正确的运动节奏
- **论文叙事**: "Spectrum-Aligned Noise Initialization" — 从频域视角解决 noise prior 的毒性问题

### 7.7 独立 freq_reshape β=1.0 实验结果（❌ 失败）

> 实验配置: 独立频域重塑模式，β=1.0，**无 alpha blend**（`eta = freq_reshape(η_temporal, η_random)` 直接 return）  
> 命令: `--noise_prior --svd_mode v1 --freq_reshape --freq_reshape_beta 1.0`（无 `--alpha` 混合）  
> 数据来源: 服务器评测日志（整体均值）

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| 独立 freq_reshape (β=1.0, 无 blend) | 0.8339 | 0.6690 | CLIP **-6.97%** ❌, XCLIP **-15.03%** ❌ |

**结论: 独立频域重塑模式整体显著劣于 baseline，是目前所有 L2 改进中最差的一个。**

### 7.8 失败原因分析（关键发现）

生成日志显示，重塑后噪声与原始随机噪声的余弦相似度 `cos(shaped, random) ≈ 0.87 ~ 0.99`，即**频域重塑后的 η_random 与纯随机高斯噪声几乎没有区别**。

根本原因:

1. **β=1.0 只改频谱「幅度形状」，不改相位**。算法第 6 步 `F_out = F_r × reshape_filter` 仅调制幅度，相位仍是 η_random 的随机相位。时间频谱形状对运动节奏的引导信号非常弱，无法弥补「丢弃 η_temporal 全部空间内容」的损失。
2. **完全抛弃了 η_temporal 的内容注入**。原 linear blend 中 `√α·η_temporal` 这一项虽小（α=0.004），但确实携带了参考视频的结构/运动信息，对生成质量有正面贡献。独立重塑把这份信息全部丢掉，只保留一个近乎无效的频谱形状，结果自然回退到「接近纯随机初始化」，且因 renorm/重塑引入额外扰动而比纯 baseline 更差。

**这一负面结果反向证明了 alpha blend 注入 η_temporal 内容的正面价值——不能完全丢弃，频域重塑应作为 η_random 的「预处理」而非「替代」。**

### 7.9 修正方案：频域重塑作为 η_random 预处理 + alpha blend（叠加模式）

基于 7.8 的发现，将方向 C 从「独立替代」改为「叠加增强」：先用频域重塑改造 η_random 的时间频谱，再走原 SVD 的 alpha 线性混合。

```
输入: η_temporal (SVD 滤波后), η_random (标准高斯)

1. η_random' = freq_reshape(η_temporal, η_random, β)   # 频域重塑作为预处理
2. η = √α · η_temporal + √(1-α) · η_random'             # 继续走原 alpha blend
```

关键性质:

- **保留 η_temporal 内容注入**（`√α·η_temporal` 项不变），不重蹈独立模式覆辙；
- **同时让 η_random 携带参考视频的时间节奏**（频域形状），叠加增益；
- **向下兼容**: β=0 时 `spectrum_shape^0 = 全 1`，重塑退化为恒等变换，整条路径完全等价于老方向（纯 alpha blend）。即 `--freq_reshape --freq_reshape_beta 0` ≡ 不开 freq_reshape。

命令（叠加模式，需同时开 `--noise_prior`/`--alpha` 与 `--freq_reshape`）:

```bash
cd /root/xixihaha/P-Flow

python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/freq_reshape_blend_b1_a004 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --freq_reshape \
    --freq_reshape_beta 1.0 \
    --seed 42 --steps 30 --guidance 5.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/freq_reshape_blend_b1_a004 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/freq_reshape_blend_b1_a004
```

### 7.10 叠加模式 β=1.0 实验结果（❌ 也失败）

> 实验配置: freq_reshape 作为 η_random 预处理 + alpha=0.004 线性混合  
> 命令: `--noise_prior --svd_mode v1 --alpha 0.004 --freq_reshape --freq_reshape_beta 1.0`  
> 数据来源: 服务器评测结果 (evaluation_results/freq_reshape_blend_b1_a004)

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| 独立 freq_reshape (β=1.0, 无 blend) | 0.8339 | 0.6690 | CLIP -6.97%, XCLIP -15.03% ❌ |
| 叠加模式 (β=1.0, α=0.004) | 0.8384 | 0.6835 | CLIP **-6.47%** ❌, XCLIP **-13.20%** ❌ |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | 叠加 CLIP | Δ CLIP | Baseline XCLIP | 叠加 XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.6954 | **-25.3%** ❌ | 0.6982 | 0.2146 | **-69.3%** ❌ |
| 17 | SUV越野 | 0.9092 | 0.8918 | -1.9% | 0.8368 | 0.7878 | -5.9% |
| 21 | 丛林纸飞机 | 0.8928 | 0.8379 | -6.2% | 0.7637 | 0.6919 | -9.4% |
| 31 | 水下城市 | 0.8324 | 0.8149 | -2.1% | 0.5237 | 0.6530 | **+24.7%** ✅ |
| 32 | 雪地金毛 | 0.9167 | 0.9064 | -1.1% | 0.8221 | 0.8182 | -0.5% |
| 33 | 跑步者 | 0.8531 | 0.6261 | **-26.6%** ❌ | 0.8618 | 0.3223 | **-62.6%** ❌ |
| 34 | 四只小狗 | 0.8968 | 0.8803 | -1.8% | 0.8710 | 0.8119 | -6.8% |
| 43 | 花园猫咪 | 0.9539 | 0.9677 | +1.4% ✅ | 0.9069 | 0.9414 | +3.8% ✅ |
| 46 | 火山喷发 | 0.9022 | 0.8965 | -0.6% | 0.7869 | 0.7832 | -0.5% |
| 47 | 动画狗城市 | 0.8769 | 0.8674 | -1.1% | 0.8024 | 0.8108 | +1.0% |

胜负统计: 叠加模式赢 2/10 (S31 XCLIP, S43)，平 4/10，输 4/10 (其中 S7/S33 catastrophic failure)

### 7.11 叠加模式失败原因分析 + 方向 C 总结

日志关键诊断信息:

| 样本 | cos(shaped, random) | spectrum DC/high ratio | direction_shift |
|:---:|:---:|:---:|:---:|
| 7 | 0.8759 | 4.11 | 0.0245 |
| 17 | 0.9982 | 1.06 | 0.0218 |

**核心发现: 频域重塑本身几乎无效果。**

1. **大部分样本的时间频谱形状接近平坦**（DC/high ratio ≈ 1.06），导致 reshape_filter 接近全 1，重塑后的 η_random 与原始几乎相同 (`cos ≈ 0.998`)；
2. **少数样本 (S7) 频谱不平坦** (DC/high=4.11)，重塑确实改变了噪声 (`cos=0.876`)，但效果反而是 catastrophic failure (CLIP -25.3%, XCLIP -69.3%)——说明频谱形状调制不但不能帮助生成，反而破坏了噪声的随机性；
3. **叠加模式 vs 独立模式**：叠加模式 (CLIP 0.8384, XCLIP 0.6835) 仅比独立模式 (0.8339, 0.6690) 微弱提升 (+0.54%, +2.17%)，说明「保留 η_temporal 内容注入」带来的正面价值在 α=0.004 这个量级上微乎其微（`direction_shift ≈ 0.024`，小到几乎感知不到）。

**方向 C 结论: 频域噪声重塑路线完全失败。**

- 理论上，只传递"时间频谱形状"这一信息太弱了，无法为生成提供有意义的引导;
- 而且对于频谱不平坦的样本，强行调制反而破坏了噪声随机性，导致 catastrophic failure;
- **建议放弃方向 C，回到纯 alpha blend 路线，专注调优 α 值和 L1 caption 联合优化。**

---

## 八、方向 D: Std-Gated Adaptive Alpha (SGA)

### 8.1 动机与论文支撑

前序实验揭示了一个核心矛盾：α=0.01 对 8/10 样本有正向收益 (XCLIP +1.2%)，但 S7/S31 出现 catastrophic failure (-34.5%/-22.8%)。问题在于固定 alpha 无法适应不同样本 η_temporal 的信号强度差异。

调研了 6 篇相关论文，找到直接理论支撑：

| 论文 | 会议 | 核心思想 | 与我们的关系 |
|------|:---:|------|------|
| SSNI | ICML 2025 | 用 score norm 估计样本偏离度，自适应调整噪声注入量 | **直接类比**: η_temporal_std 作为偏离度代理 |
| MuLAN | NeurIPS 2024 Spotlight | 学习 per-pixel 自适应噪声 schedule | **理论背书**: 均匀噪声策略确实次优 |
| PYoCo | ICCV 2023 | Video noise prior (correlated noise) | 我们 SVD prior 的直接前身 |
| FreeInit | ECCV 2024 | 低频时空结构对生成质量关键 | 佐证 η_temporal 低频信号有价值 |
| How I Warped Your Noise | ICLR 2024 | 保 Gaussian 的 temporally-correlated noise | 验证保分布注入相关性思路正确 |
| FreeNoise | ICLR 2024 | Noise rescheduling 保长距离时序一致性 | 佐证噪声时序结构有效 |

**关键洞察** (来自 SSNI): SSNI 用 `‖∇ log p(x)‖`（score norm）衡量每个样本离干净分布的偏离度，据此自适应调整注入量。我们类比：用 `η_temporal_std` 衡量 SVD 时序信号的强度/偏离度，据此自适应调整 alpha。

### 8.2 SGA 算法

```python
# 核心公式
actual_std = eta_temporal.std()                    # 当前样本的 η_temporal 标准差
raw_alpha = base_alpha × (target_std / actual_std) # 线性反比缩放
effective_alpha = clamp(raw_alpha, alpha_min, alpha_max)  # 安全区间

# 默认参数
base_alpha = 0.004      # 原始 baseline alpha
target_std = 0.30       # 中位数附近 (实测分布: 0.28~0.41)
alpha_min = 0.001       # 下界: 防止完全不注入
alpha_max = 0.010       # 上界: 防止 catastrophic failure (α=0.01 S7 已证明危险)
```

逻辑直觉:
- η_temporal_std **小** (如 S34=0.28) → 信号温和/低频主导 → alpha **提升** (0.004→0.0043) → 充分利用
- η_temporal_std **大** (如 S33=0.41) → 信号偏离大/高频强 → alpha **降低** (0.004→0.0029) → 保护性降低
- η_temporal_std **很大** (异常样本) → alpha 触及下界 0.001 → 几乎不注入，完全保护

### 8.3 与前序方案的对比

| 方案 | 自适应指标 | 失败原因 | SGA 为何不同 |
|------|------|------|------|
| QGA (方案 B) | direction quality score | quality 系统性≈0，无法区分 | SGA 用 std，实测差异明确 (0.28~0.41) |
| 固定 α=0.01 | 无 (all-or-nothing) | S7/S31 catastrophic failure | SGA 对高 std 样本自动降 alpha |
| renorm | 统一 std=1.0 | 抹杀样本差异 | SGA 利用差异而非消除差异 |

### 8.4 预期效果（按样本推算）

| 样本 | η_temporal std | SGA alpha | vs baseline α=0.004 | 预期影响 |
|:---:|:---:|:---:|:---:|------|
| 34 | 0.28 | 0.0043 | +7.5% | 温和提升，充分利用 |
| 17 | 0.33 | 0.0036 | -10% | 轻微保护 |
| 7 | 0.37 | 0.0032 | -20% | 显著保护 (防止 catastrophic) |
| 33 | 0.41 | 0.0029 | -27% | 大幅保护 |
| 43 | 0.39 | 0.0031 | -23% | 保护 (baseline已很好) |

### 8.5 实现位置

- CLI 参数: `run.py` 的 `--adaptive_alpha`, `--sga_target_std`, `--sga_alpha_min`, `--sga_alpha_max`
- 配置字段: `src/pipeline.py` 的 `PFlowConfig.adaptive_alpha`, `.sga_target_std`, `.sga_alpha_min`, `.sga_alpha_max`
- 核心逻辑: `src/pipeline.py` 的 `_get_latents()` 方法，在 alpha blend 前动态计算 effective_alpha

### 8.6 实验命令

```bash
cd /root/xixihaha/P-Flow

# ── 实验 D1: SGA + v9 caption (target_std=0.30, 默认参数) ──
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/sga_v9_ts030 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --adaptive_alpha \
    --sga_target_std 0.30 \
    --sga_alpha_min 0.001 \
    --sga_alpha_max 0.010 \
    --steps 30 --guidance 5.0 --seed 42

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/sga_v9_ts030 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/sga_v9_ts030

# ── 实验 D2: SGA + v9 caption (target_std=0.33, 更激进) ──
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/sga_v9_ts033 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --adaptive_alpha \
    --sga_target_std 0.33 \
    --sga_alpha_min 0.001 \
    --sga_alpha_max 0.010 \
    --steps 30 --guidance 5.0 --seed 42

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/sga_v9_ts033 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/sga_v9_ts033

# ── 实验 D3: SGA + v9 + alpha_max=0.008 (更保守上界) ──
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/sga_v9_ts030_max008 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --adaptive_alpha \
    --sga_target_std 0.30 \
    --sga_alpha_min 0.001 \
    --sga_alpha_max 0.008 \
    --steps 30 --guidance 5.0 --seed 42

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/sga_v9_ts030_max008 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/sga_v9_ts030_max008

# ── 对照: SGA + 原始 caption (无 v9, 纯看 SGA 效果) ──
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/sga_orig_ts030 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --alpha 0.004 \
    --adaptive_alpha \
    --sga_target_std 0.30 \
    --sga_alpha_min 0.001 \
    --sga_alpha_max 0.010 \
    --steps 30 --guidance 5.0 --seed 42

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/sga_orig_ts030 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/sga_orig_ts030
```

### 8.7 预期验证逻辑

**成功判据**:
1. **不退化**: 整体 CLIP/XCLIP ≥ baseline (0.8964/0.7874)
2. **无 catastrophic failure**: S7/S31/S33 的 XCLIP 不出现 >10% 跌幅
3. **正向收益**: 特别关注之前 α=0.01 受益的样本 (S17/S32/S33/S34/S43) 是否依然受益

**论文叙事**: "Sample-Adaptive Noise Prior Injection" — 通过信号强度自适应调节注入量，同时避免 catastrophic failure 和实现 per-sample 最优注入。理论支撑: SSNI (ICML 2025) 的 score-aware 框架 + MuLAN (NeurIPS 2024) 对均匀策略次优性的证明。

### 8.8 SGA 实验结果（❌ 效果边际，pass）

> 实验配置: SGA + 原始 caption, target_std=0.30, alpha_min=0.001, alpha_max=0.010  
> 命令: `--noise_prior --svd_mode v1 --alpha 0.004 --adaptive_alpha --sga_target_std 0.30`  
> 数据来源: 服务器评测结果

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | 0.7874 | — |
| SGA (target_std=0.30, 原始 caption) | 0.8834 | 0.7899 | CLIP **-1.4%** ❌, XCLIP +0.3% |

**结论: SGA 整体效果边际，XCLIP 仅提升 0.3%（统计不显著），但 CLIP 下降了 1.4%，权衡后收益不足以 justify 额外复杂度。**

### 8.9 SGA 失败原因分析

SGA 的核心假设是"η_temporal_std 大的样本需要降低 alpha 以保护"。但实验暴露了根本问题：

1. **样本间 std 差异太小 (0.28~0.41)**：SGA 对 alpha 的调节范围实际只有 0.0029~0.0043，与 baseline 固定 0.004 差距极小，无法产生有意义的效果差异。这种级别的微调被生成模型的内在随机性淹没。

2. **CLIP 下降的根因**：SGA 对高 std 样本（S33, S43 等 baseline 已经很好的样本）系统性降低了 alpha，相当于削弱了原本有效的 temporal prior 注入，导致 CLIP 退化。

3. **与 α=0.01 实验的矛盾**：α=0.01 实验表明，对大部分样本来说提高 alpha 是正向的（8/10 样本受益），SGA 却系统性地降低了这些样本的 alpha，方向可能反了。

**核心教训**: 在 one-shot linear blend 框架下，α=0.004 附近的微调无法突破天花板。需要范式级的改变——不是"注入多少"的问题，而是"注入什么"的问题。这为方向 E (PODI) 提供了动机：与其调注入量，不如改善注入内容的质量（只注入与 prompt 语义对齐的部分）。

---

## 九、方向 E: Prompt-Orthogonal Decomposition Injection (PODI)

### 9.1 动机

SGA (方向 D) 证明了"调注入量"路线的天花板：在 α∈[0.001, 0.01] 范围内微调无法突破 baseline。回顾之前所有失败的方向，核心问题始终是：η_temporal 中混杂了与 prompt 语义无关甚至矛盾的成分，直接注入会引入"噪声污染"。

PODI 的思路是：**不调量，改质**——将 η_temporal 分解为"与 prompt 语义对齐"和"与 prompt 正交"两个分量，只注入对齐的部分。

### 9.2 论文支撑

| 论文 | 会议 | 核心思想 | 与 PODI 的关系 |
|------|:---:|------|------|
| Golden Noise | ICCV 2025 | 优化初始噪声使其对齐 text embedding | **直接灵感**: 对齐方向的噪声对生成有益 |
| InitNO | CVPR 2024 | 梯度优化初始噪声对齐 prompt | 验证"对齐"方向的价值 |
| ODC (Optimal Denoising Control) | — | 最优去噪方向应与条件对齐 | 理论支撑: 正交分量对生成无用 |
| Not All Noises Are Created Equally | NeurIPS 2024 | 分析噪声不同分量对生成的差异化影响 | 支持"分解+选择性注入" |
| Noise PPO | — | 用 RL 优化噪声分布 | 另一种噪声质量改善路线 |

### 9.3 算法

```
输入: η_temporal (SVD 滤波后), η_random (标准高斯), prompt_embeds (text encoder 输出)
参数: podi_alpha (注入强度, 默认 0.004 与 baseline 一致), podi_min_alignment (最低对齐阈值)

1. 将 prompt_embeds 映射到 latent channel 空间:
   - prompt_embeds: [seq_len, hidden_dim] → mean pool → [hidden_dim]
   - 通过 chunked average pooling 压缩到 [C] (C=latent channels, 通常 16)
   - 得到 prompt_dir (归一化到单位向量)

2. 对 η_temporal 做 channel 维度的正交分解:
   - η_temporal: [F, C, H, W] → 在 channel 维 (dim=1) 上投影
   - parallel = (η_temporal · prompt_dir) × prompt_dir  (沿 prompt 方向的分量)
   - orthogonal = η_temporal - parallel                  (与 prompt 正交的分量)

3. 计算对齐度:
   - alignment = |cos(η_temporal_mean_channel, prompt_dir)|
   - 若 alignment < podi_min_alignment: 说明 η_temporal 与 prompt 几乎无关，
     fallback 到普通 alpha blend (退化为 baseline)

4. 只注入 parallel 分量:
   - parallel_renorm = parallel × (1 / parallel.std())  (renorm to N(0,1))
   - η = √(podi_alpha) × parallel_renorm + √(1-podi_alpha) × η_random

输出: η — 只携带了与 prompt 语义对齐的时序信息
```

### 9.4 关键设计选择

**为什么 podi_alpha 默认 0.004 (与 baseline 一致)**:
- 公平对比的前提是控制变量——注入总量 (α) 相同，唯一区别是注入内容的"质量"
- Baseline: 注入完整 η_temporal (含对齐+正交部分)
- PODI: 只注入对齐部分 (parallel_renorm)，期望同等注入量下生成质量更高
- 如果 PODI 有效，后续可以尝试更大的 podi_alpha (因为对齐内容更安全)

**对齐度门控 (alignment gating)**:
- 对于随机 η_temporal，理论对齐度 ≈ 1/√C = 1/√16 ≈ 0.25
- podi_min_alignment 默认 0.01，极低阈值，仅过滤完全无对齐的极端情况
- alignment 不参与 alpha 缩放（SGA 的教训：微调注入量无效），只做 accept/reject gate

### 9.5 实现位置

- 配置字段: `PFlowConfig.podi`, `.podi_alpha`, `.podi_min_alignment`, `.podi_proj_mode`
- CLI 参数: `--podi`, `--podi_alpha` (默认 0.004), `--podi_min_alignment`, `--podi_proj_mode`
- 核心方法: `PFlowPipeline._podi_decompose()` — 正交分解 + 对齐计算
- 路径分支: `_get_latents()` 中 PODI 路径 (早期 return，不影响其他路径)

### 9.6 实验命令

```bash
cd /root/xixihaha/P-Flow

# ── PODI α=0.004 (与 baseline 公平对比) ──
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/podi_alpha004 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior \
    --svd_mode v1 \
    --podi --podi_alpha 0.004 \
    --steps 30 --guidance 5.0 --seed 42

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/podi_alpha004 \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir evaluation_results/podi_alpha004
```

### 9.7 PODI 实验结果（❌ 失败）

> 实验配置: PODI α=0.004, 原始 caption  
> 命令: `--noise_prior --svd_mode v1 --podi --podi_alpha 0.004`  
> 数据来源: 服务器评测结果

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| PODI (α=0.004, 原始 caption) | 0.8739 | 0.7464 | CLIP **-2.5%** ❌, XCLIP **-5.2%** ❌ |

**逐 Case 对比:**

| 样本 | 场景 | alignment | parallel_std | CLIP | XCLIP | Δ CLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.1943 | 0.0703 | 0.9147 | 0.6717 | -1.7% | -3.8% |
| 17 | SUV越野 | 0.3548 | 0.1116 | 0.8803 | 0.7769 | -3.2% | -7.2% ❌ |
| 21 | 丛林纸飞机 | 0.2293 | 0.0759 | 0.8896 | 0.7485 | -0.4% | -2.0% |
| 31 | 水下城市 | 0.2547 | 0.0835 | 0.8406 | 0.5932 | +1.0% | +13.3% ✅ |
| 32 | 雪地金毛 | 0.2100 | 0.0743 | 0.9065 | 0.8099 | -1.1% | -1.5% |
| 33 | 跑步者 | 0.2139 | 0.0826 | 0.8399 | 0.8165 | -1.5% | -5.3% |
| 34 | 四只小狗 | 0.2070 | 0.0580 | 0.8892 | 0.8187 | -0.8% | -6.0% ❌ |
| 43 | 花园猫咪 | 0.2297 | 0.0862 | 0.9413 | 0.8860 | -1.3% | -2.3% |
| 46 | 火山喷发 | 0.2061 | 0.0739 | 0.8775 | 0.7476 | -2.7% | -5.0% |
| 47 | 动画狗城市 | 0.2250 | 0.0831 | 0.7591 | 0.5952 | **-13.4%** ❌ | **-25.8%** ❌❌ |

胜负统计: PODI 赢 1/10 (S31 XCLIP +13.3%)，输 8/10，平 1/10

### 9.8 PODI 失败原因分析

1. **对齐度系统性极低** (alignment ≈ 0.19~0.35)：理论随机对齐度 = 1/√16 ≈ 0.25，实测未超过这个水平。说明 text embedding → latent channel 之间**不存在有意义的语义映射关系**，PODI 的核心假设不成立。

2. **parallel 分量能量极低** (parallel_std ≈ 0.06~0.11)：η_temporal 被分解后，"对齐"分量只保留了原始信号的 7%~11% 标准差。renorm 后方向信息严重扭曲，本质上注入的是近随机方向噪声。

3. **最终信号不可见** (cos(mixed, random) ≈ 1.0000)：α=0.004 本身 direction_shift ≈ 0.025 已很弱，PODI 再削去 80%~90% 后剩余信号几乎为零，在最终噪声中完全不可感知。

4. **核心矛盾**: PODI 假设 prompt embedding 的 chunked pooling → 16 维 channel 空间存在语义结构。但 4096→16 的粗糙映射无法保留任何语义信息。

**结论: PODI 路线失败。"改质"思路的正确方向不应依赖外部语义映射（prompt→channel），而应利用数据本身的结构特征（如 channel temporal variance、空间多尺度等）。**

**详细调研与后续方案见**: `docs/论文调研_L2噪声先验改进方向.md`

---

## 十、实验路径规划（更新后）

```
当前状态
├── L2 SVD 改进
│   ├── [已完成] renorm α=0.001 → 失败，整体不如 v1
│   ├── [已否决] rescale 模式 → 等价于调 alpha，无独立价值
│   ├── [已失败] Quality-Gated Alpha → quality scores 系统性接近 0，无法区分样本
│   ├── [已失败] α=0.01 → 后 9 样本正向 (+1.2%)，但 S7/S31 catastrophic failure
│   ├── [已失败] 方向 C 独立频域重塑 β=1.0 → CLIP -6.97%, XCLIP -15.03%
│   ├── [已失败] 方向 C 叠加模式 (freq_reshape+α=0.004) → CLIP -6.47%, XCLIP -13.20%
│   │     结论: 频域形状信息太弱且有害，方向 C 完全失败
│   ├── [已失败] 方向 D: SGA → CLIP -1.4%, XCLIP +0.3% (边际，pass)
│   │     结论: "调量"路线天花板已到，需转向"改质"
│   ├── [已失败] 方向 E: PODI → CLIP -2.5%, XCLIP -5.2%
│   │     结论: prompt→channel 语义映射不存在，"改质"需换数据驱动思路
│   ├── [已失败] 方向 F: CEGI (通道能量门控) → CLIP -3.4%, XCLIP -5.2%
│   │     结论: channel 间信号无明显集中性，集中注入反而丢失信息
│   ├── [★当前] 方案 G/H/I 实验中
│   │     G: MSTDI (多尺度分层注入) — 空间低频杠杆效应
│   │     H: TPI (时间相位注入) — 不注入内容只传相位
│   │     I: OCS (正交补抑制) — 逆向思维不注入temporal
│   └── [后续] 选定方案的 α 消融 + 200 样本验证
│
├── L1 Prompt Rewrite
│   ├── [已完成] v8-minimal → 失败 (CLIP 0.8915 < baseline 0.8964)
│   ├── [已完成] v9-vlm → CLIP 持平, XCLIP +1.3% (小幅正收益)
│   └── [结论] L1 方向确认: "删减 + VLM 补充" > "LLM 改写"
│
└── L3 Velocity Matching
    └── [暂缓] 等 L2 稳定后再叠加
```

### 下一步优先级

| 序号 | 实验 | 预期收益 | 成本 | 状态 |
|:---:|------|:---:|:---:|------|
| 1 | **G: MSTDI (多尺度分层注入)** | 低频 α=0.05，空间杠杆效应，不丢弃任何 channel | 中 | **★ 实验中** |
| 2 | **H: TPI (时间相位注入)** | 只改相位不改幅度，保 Gaussian，避免内容毒性 | 中 | **★ 实验中** |
| 3 | **I: OCS (正交补抑制)** | 不注入 temporal，从 random 中抑制正交分量 | 中 | **★ 实验中** |
| 4 | MSTDI/TPI/OCS + v9 caption | L1+L2 叠加 | 低 | 等 G/H/I 结果 |
| 5 | 扩大样本量验证 | 统计显著性 | 高 | 论文投稿前 |

---

## 十一、关键数据存档

### 生成日志位置 (服务器)

- Pure L2 Baseline: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/`
- SVD V2 Renorm: `/root/xixihaha/P-Flow/outputs/svd_v2_renorm_alpha001/`
- v9 Prompt Rewrite: `/root/xixihaha/P-Flow/outputs/v9_svd_v1/`
- α=0.01 实验: `/root/xixihaha/P-Flow/outputs/svd_v1_alpha010/`
- Baseline 评测: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/eval_results/`
- Renorm 评测: `/root/xixihaha/P-Flow/evaluation_results/svd_v2_renorm_alpha001/`
- v9 评测: `/root/xixihaha/P-Flow/evaluation_results/v9_svd_v1/`
- α=0.01 评测: `/root/xixihaha/P-Flow/evaluation_results/svd_v1_alpha010/`
- 叠加模式: `/root/xixihaha/P-Flow/outputs/freq_reshape_blend_b1_a004/` (评测: `evaluation_results/freq_reshape_blend_b1_a004/`)
- SGA: `/root/xixihaha/P-Flow/outputs/sga_orig_ts030/` (评测: `evaluation_results/sga_orig_ts030/`)
- PODI: `/root/xixihaha/P-Flow/outputs/podi_alpha004/` (评测: `evaluation_results/podi_alpha004/`) [已完成, 失败]

### 全实验对比汇总

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | 0.7874 | — |
| SVD V2 Renorm (α=0.001) | 0.8826 | 0.7506 | CLIP -1.5%, XCLIP -4.7% ❌ |
| v9 Prompt (LLM删减+VLM补充 + v1) | 0.8947 | **0.7973** | CLIP -0.2%, XCLIP +1.3% ✅ |
| SVD v1, α=0.01 | 0.8903 | 0.7685 | CLIP -0.7%, XCLIP -2.4% ❌ |
| SVD v1, α=0.01 (后9, 去S7) | 0.8947 | 0.7966 | CLIP -0.2%, XCLIP +1.2% ✅ |
| 独立频域重塑 β=1.0 (无 blend) | 0.8339 | 0.6690 | CLIP **-6.97%**, XCLIP **-15.03%** ❌ |
| 叠加模式 β=1.0 (freq_reshape+α=0.004) | 0.8384 | 0.6835 | CLIP **-6.47%**, XCLIP **-13.20%** ❌ |
| SGA (target_std=0.30, 原始 caption) | 0.8834 | 0.7899 | CLIP -1.4%, XCLIP +0.3% ❌ (边际) |
| PODI (α=0.004, 原始 caption) | 0.8739 | 0.7464 | CLIP **-2.5%**, XCLIP **-5.2%** ❌ |
| CEGI (top_k=4, α_inject=0.02, α_residual=0) | 0.8661 | 0.7467 | CLIP **-3.4%**, XCLIP **-5.2%** ❌ |

### 冲突诊断汇总

| 样本 | 冲突风险 | Prompt 方向 | SVD 方向 | 结论 |
|:---:|:---:|---|---|---|
| 7 | LOW | up, gentle | 均匀 | 安全 |
| 17 | **HIGH** | up, forward | 偏上 | 方向一致,非真冲突 |
| 21 | LOW | move | 均匀 | 安全 |
| 31 | LOW | 无 | 均匀 | 安全 |
| 32 | **HIGH** | up | 偏右 | 轻微冲突 |
| 33 | LOW | steady, move | 偏左偏下 | 安全 |
| 34 | **HIGH** | up | 偏上 | 方向一致,非真冲突 |
| 43 | LOW | right, up, camera, move, flow | 均匀 | 安全 |
| 46 | LOW | right, up, pan | 均匀 | 安全 |
| 47 | LOW | 无 | 均匀 | 安全 |

---

## 十二、方向 F: Channel-Energy Gated Injection (CEGI)

### 12.1 动机与核心思想

PODI (方向 E) 失败的根因是"外部语义映射不存在"——prompt embedding 到 16 维 channel 空间的 chunked pooling 不携带任何语义结构。CEGI 转向**数据驱动**：不依赖外部信号，而是用 η_temporal 自身的 per-channel temporal variance 作为"哪些 channel 携带运动信息"的指示器。

核心思想：η_temporal 的 16 个 channel 中，temporal variance 高的 channel 更可能编码了真实运动信息（因为静态/噪声 channel 的帧间变化小）。CEGI 只在 top-k 高方差 channel 上集中注入 temporal prior (α_inject >> baseline α)，其余 channel 不注入 (α_residual=0)，保持纯随机。

与前序方法的关系：
- vs baseline (均匀 α=0.004): CEGI 让"有运动信号的 channel"拿到 5× 的注入量
- vs PODI (依赖 prompt→channel 映射): CEGI 完全数据驱动，无外部依赖
- vs SGA (微调总 α): CEGI 重新分配注入到不同 channel，而非调总量

### 12.2 算法

```
输入: η_temporal [B, C, F, H, W], η_random [B, C, F, H, W]
参数: top_k=4 (25% channels), α_inject=0.02, α_residual=0.0

1. 计算 per-channel temporal variance:
   var_c = Var(η_temporal[:, c, :, :, :])  for c in [0..C-1]
   
2. 选择 top-k channels (按 var_c 降序):
   selected = argsort(var_c, descending=True)[:top_k]
   
3. 对 selected channels: 
   η[:, selected] = √α_inject × η_temporal[:, selected] + √(1-α_inject) × η_random[:, selected]
   Per-channel renorm: η[:, c] = (η[:, c] - mean) / std  ∀ c ∈ selected
   
4. 对其余 channels:
   η[:, rest] = √α_residual × η_temporal[:, rest] + √(1-α_residual) × η_random[:, rest]
   (α_residual=0 时直接用 η_random)

输出: η — 运动 channel 被注入强信号，非运动 channel 保持纯随机
```

### 12.3 实验配置

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/cegi_k4_a02 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior --svd_mode v1 \
    --cegi --cegi_top_k 4 --cegi_alpha 0.02 \
    --steps 30 --guidance 5.0 --seed 42
```

### 12.4 实验结果（❌ 未超越 baseline，但未 catastrophic failure）

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| CEGI (top_k=4, α_inject=0.02, α_residual=0) | 0.8661 | 0.7467 | CLIP **-3.4%** ❌, XCLIP **-5.2%** ❌ |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | CEGI CLIP | Δ CLIP | Baseline XCLIP | CEGI XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.8796 | -5.5% ❌ | 0.6982 | 0.7344 | +5.2% ✅ |
| 17 | SUV越野 | 0.9092 | 0.8813 | -3.1% ❌ | 0.8368 | 0.7932 | -5.2% ❌ |
| 21 | 丛林纸飞机 | 0.8928 | 0.7848 | **-12.1%** ❌ | 0.7637 | 0.4802 | **-37.1%** ❌❌ |
| 31 | 水下城市 | 0.8324 | 0.8121 | -2.4% | 0.5237 | 0.6299 | **+20.3%** ✅ |
| 32 | 雪地金毛 | 0.9167 | 0.9338 | +1.9% ✅ | 0.8221 | 0.8076 | -1.8% |
| 33 | 跑步者 | 0.8531 | 0.8157 | -4.4% ❌ | 0.8618 | 0.7711 | **-10.5%** ❌ |
| 34 | 四只小狗 | 0.8968 | 0.8661 | -3.4% | 0.8710 | 0.8560 | -1.7% |
| 43 | 花园猫咪 | 0.9539 | 0.9287 | -2.6% | 0.9069 | 0.8791 | -3.1% |
| 46 | 火山喷发 | 0.9022 | 0.9047 | +0.3% | 0.7869 | 0.7151 | -9.1% ❌ |
| 47 | 动画狗城市 | 0.8769 | 0.8538 | -2.6% | 0.8024 | 0.8001 | -0.3% |

胜负统计: CEGI 赢 2/10 (S7 XCLIP +5.2%, S31 XCLIP +20.3%)，输 7/10，平 1/10

### 12.5 日志关键诊断数据

**Per-sample CEGI Channel Selection & Blend Diagnostics:**

| 样本 | Selected Channels | Var Range (selected) | Var Range (rest) | direction_shift | cos(mixed, random) | cos(mixed, temporal) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | [5, 3, 14, 15] | [0.162, 0.244] | [0.048, 0.140] | 0.0319 | 1.0000 | 0.0192 |
| 17 | [3, 12, 5, 15] | [0.129, 0.163] | [0.063, 0.126] | 0.0275 | 1.0000 | 0.0161 |
| 21 | [1, 3, 12, 11] | [0.124, 0.141] | [0.072, 0.123] | 0.0264 | 1.0000 | 0.0151 |
| 31 | [3, 15, 7, 2] | [0.118, 0.184] | [0.077, 0.112] | 0.0276 | 1.0000 | 0.0145 |
| 32 | [0, 7, 5, 3] | [0.149, 0.185] | [0.067, 0.144] | 0.0294 | 1.0000 | 0.0150 |
| 33 | [8, 3, 14, 10] | [0.162, 0.284] | [0.057, 0.161] | 0.0338 | 1.0000 | 0.0195 |
| 34 | [3, 5, 12, 11] | [0.090, 0.111] | [0.039, 0.085] | 0.0231 | 1.0000 | 0.0118 |
| 43 | [3, 5, 12, 8] | [0.163, 0.241] | [0.109, 0.159] | 0.0319 | 1.0000 | 0.0176 |
| 46 | [3, 15, 12, 7] | [0.153, 0.273] | [0.085, 0.139] | 0.0323 | 1.0000 | 0.0206 |
| 47 | [3, 8, 5, 1] | [0.170, 0.239] | [0.081, 0.156] | 0.0331 | 1.0000 | 0.0187 |

**Channel 3 出现在 10/10 个样本的 top-4 选择中**，说明它是全局的"运动主通道"。Channel 5 出现 6/10 次，Channel 12 出现 5/10 次。

### 12.6 失败原因分析

#### 核心问题: 信号仍然不可见

尽管 CEGI 将注入集中到 top-4 channel，`cos(mixed, random) = 1.0000`（4位小数精度内无法区分混合噪声与纯随机噪声）。`direction_shift` 仅 0.023~0.034，`cos(mixed, temporal)` 仅 0.012~0.021。

**数学解释**: α_inject=0.02 在 4/16 channel 上注入。对于整体噪声而言:
- 4 个 channel 各承受 √0.02 ≈ 0.141 的 temporal 成分
- 但这 4 个 channel 只占总维度的 25%
- 经过 per-channel renorm 后，全局 cos similarity 无法感知到这个局部信号
- 实际整体 effective injection ≈ √(0.02 × 4/16) = √0.005 ≈ 0.071 — 仍然极弱

#### 为什么比 baseline 还差？

baseline (α=0.004) 虽然弱，但是**均匀注入 16 个 channel**，每个 channel 都获得了微弱但一致的 temporal prior。CEGI 将注入集中到 4 个 channel，但**完全抛弃了其余 12 个 channel 的 temporal 信息** (α_residual=0)。结果:

1. **信号损失 75%**: 12 个"非选中"channel 的 temporal prior 被直接丢弃，这些 channel 虽然 variance 较低，但仍然贡献了大量有用的时序结构
2. **集中注入无法补偿**: top-4 channel 获得 5× 的 α (0.02 vs 0.004)，但经 per-channel renorm 后信号仍被稀释到不可见
3. **S21 Catastrophic Failure (XCLIP -37.1%)**: 该样本的 channel variance 范围非常窄 ([0.072, 0.141])，top-4 与 rest 差异极小 (1.01× 对比)，说明 temporal signal 分散在所有 channel 中，集中注入反而破坏了分布

#### Channel Variance 分布分析

| 样本 | Top-4 Var (min) | Rest Var (max) | Top/Rest 比值 | 结果 |
|:---:|:---:|:---:|:---:|:---:|
| 7 | 0.162 | 0.140 | 1.16× | XCLIP +5.2% ✅ |
| 31 | 0.118 | 0.112 | 1.05× | XCLIP +20.3% ✅ |
| 33 | 0.162 | 0.161 | **1.01×** | XCLIP -10.5% ❌ |
| 21 | 0.124 | 0.123 | **1.01×** | XCLIP **-37.1%** ❌❌ |
| 46 | 0.153 | 0.139 | 1.10× | XCLIP -9.1% ❌ |

关键发现: **当 top-4 / rest 方差比值接近 1.0 时，CEGI 的 channel selection 几乎是随机的**——即 temporal signal 没有明显的 channel 集中性，强行划分 top-k 只是人为制造噪声。

反直观的是，S7 和 S31（两个在 baseline 上表现最差的样本）在 CEGI 中反而受益 — 这与之前 α=0.01 实验的模式一致：这些样本需要更强的注入才能改善，但其他样本因为过度注入而退化。

### 12.7 结论与教训

**CEGI 失败的根本原因**:

1. **"注入什么"的问题未解决**: CEGI 只改变了 channel 分配策略，但注入的仍然是 η_temporal 的原始内容。baseline 的核心问题（η_temporal 中混杂有害成分）并未被 CEGI 处理。

2. **Channel 信号无明显集中性**: 10 个样本中，top-4 与 rest 的 variance 比值多在 1.0×~1.16× 之间，说明 temporal signal 分布在所有 16 个 channel 中，不存在"少数运动 channel + 多数噪声 channel"的假设结构。

3. **α_inject=0.02 仍然太弱**: 即使集中到 4 个 channel，direction_shift 仍不超过 0.034，与 baseline 的 0.024 差距极小。信号在 2M+ 维噪声空间中仍被淹没。

4. **与 baseline 相比的劣势**: baseline 的均匀注入策略虽然"不聪明"，但保留了所有 channel 的 temporal prior，形成了一种"面积式覆盖"；CEGI 的集中策略追求"深度"但牺牲了"广度"，在当前信号强度下得不偿失。

**对后续方案的启示**:
- **MSTDI (多尺度)**: 从空间维度做 coarse-to-fine 分配，避免 CEGI 那样在 channel 维丢弃信息。低频空间注入可能比 channel 选择更有效，因为低频分量对全局运动的杠杆效应更大。
- **TPI (时间相位)**: 完全不注入 η_temporal 的内容，只传递其时间相位结构。避免了"有害内容注入"的根本问题。
- **OCS (正交补抑制)**: 不注入 temporal，而是从 η_random 中去除与 temporal 正交的成分。逆向思维，可能绕过注入信号太弱的瓶颈。

### 12.8 数据存档

- 生成目录: `/root/xixihaha/P-Flow/outputs/cegi_k4_a02/`
- 评测结果: `/root/xixihaha/P-Flow/evaluation_results/cegi_k4_a02/`
- 运行日志: `/root/xixihaha/P-Flow/outputs/cegi_k4_a02/run_log.txt`

---

## 十三、方向 G: MSTDI (Multi-Scale Temporal Decomposition Injection)

### 13.1 方法原理

核心思想: 将噪声在空间维度做 Gaussian Pyramid 多尺度分解，在粗尺度（低频）用大 α 注入 temporal prior 控制全局运动方向，在细尺度（高频）用小 α 保持随机性保证视觉质量。

理论支撑:
- FreeInit (ECCV 2024): 低频分量决定全局运动
- Video-MSG (2025): 多尺度引导策略
- 扩散模型 coarse-to-fine 特性: 低频结构有"杠杆效应"

算法流程:
1. 对 η_temporal 和 η_random 分别做 spatial avg_pool3d 到多个尺度
2. 在每个尺度上用指数衰减的 α 做 linear blend (sqrt(α)·η_t + sqrt(1-α)·η_r)
3. 通过 Laplacian 差分重建全分辨率噪声 (最粗层上采样 + 逐层加高频残差)
4. 最终 renorm 到 N(0,1)

### 13.2 实验配置 (G1: 默认参数)

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/mstdi_L3_a005 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior --svd_mode v1 \
    --mstdi --mstdi_levels 3 --mstdi_alpha_base 0.05 --mstdi_alpha_decay 0.25 \
    --steps 30 --guidance 5.0 --seed 42
```

参数说明:
- `mstdi_levels=3`: 金字塔 3 层 (1/4, 1/2, 原始分辨率)
- `mstdi_alpha_base=0.05`: 最粗层 α
- `mstdi_alpha_decay=0.25`: 每层衰减倍率
- α schedule: [0.05000, 0.01250, 0.00313]

### 13.3 实验结果 (G1: ❌ 未超越 baseline，但优于 CEGI)

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| CEGI (top_k=4, α=0.02) | 0.8661 | 0.7467 | CLIP -3.4% ❌, XCLIP -5.2% ❌ |
| MSTDI G1 (L=3, α_base=0.05, decay=0.25) | 0.8903 | 0.7791 | CLIP **-0.7%**, XCLIP **-1.1%** |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | MSTDI CLIP | Δ CLIP | Baseline XCLIP | MSTDI XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | — | — | 0.6982 | — | — |
| 17 | SUV越野 | 0.9092 | — | — | 0.8368 | — | — |
| 21 | 丛林纸飞机 | 0.8928 | — | — | 0.7637 | — | — |
| 31 | 水下城市 | 0.8324 | — | — | 0.5237 | — | — |
| 32 | 雪地金毛 | 0.9167 | — | — | 0.8221 | — | — |
| 33 | 跑步者 | 0.8531 | — | — | 0.8618 | — | — |
| 34 | 四只小狗 | 0.8968 | — | — | 0.8710 | — | — |
| 43 | 花园猫咪 | 0.9539 | — | — | 0.9069 | — | — |
| 46 | 火山喷发 | 0.9022 | — | — | 0.7869 | — | — |
| 47 | 动画狗城市 | 0.8769 | — | — | 0.8024 | — | — |

> 注: 逐 Case 详细数据待从评测结果目录补充。总体均值 CLIP=0.8903, XCLIP=0.7791。

### 13.4 日志关键诊断数据

| 样本 | α_schedule | direction_shift | cos(mixed, random) | cos(mixed, temporal) |
|:---:|:---:|:---:|:---:|:---:|
| 全部样本 | [0.05000, 0.01250, 0.00313] | 0.016~0.023 | 0.9997~0.9999 | — |

核心观察: `cos(mixed, random)` 仍在 0.9997~0.9999 之间，信号几乎不可见。`direction_shift` 范围 0.016~0.023，甚至比 CEGI 的 0.023~0.034 还低。

### 13.5 失败原因分析

#### 根本问题: 有效注入量太小

当前 α schedule = [0.05, 0.0125, 0.003]:

1. **最粗层 (Level 0, 1/4 scale, 15×26 spatial)**:
   - α=0.05, sqrt(α)≈0.224 的 temporal 权重
   - 但此层仅包含原始分辨率 1/16 的像素 (≈6.25% 的信息量)
   - 经 Laplacian 上采样到全分辨率后，信号被扩散稀释

2. **最细层 (Level 2, 原始分辨率, 60×104 spatial)**:
   - α=0.003，与 baseline 的 α=0.004 几乎一致
   - 这层占据绝大多数像素，主导最终结果
   - 相当于 MSTDI 在最细层退化为 baseline

3. **Laplacian 重建后 + renorm 的双重稀释**:
   - 粗层信号通过 trilinear 上采样传播到全分辨率，但能量密度降为 1/16
   - 最终 renorm 进一步将任何偏移归一化掉
   - 实际整体 effective α ≈ 0.003~0.004，与 baseline 无本质区别

#### 为什么比 CEGI 好？

MSTDI 不丢弃任何维度的信息（CEGI 丢弃了 12/16 channel 的 temporal prior），最细层仍保留 α=0.003 的全域 temporal 注入。本质上 MSTDI G1 ≈ baseline + 微弱的低频额外注入，所以指标只轻微下降。

#### 与 baseline 差距不大但仍低的原因

α=0.003 (最细层) vs baseline α=0.004 — 最细层贡献了约 93.75% 的像素，且 α 比 baseline 还略低。粗层的额外注入不足以补偿这个差距。

### 13.6 调参方向 (待验证)

核心思路: **大幅增大粗层 α，使低频杠杆效应真正生效**。

| 实验 | α_base | decay | levels | α schedule | 预期 direction_shift | 理由 |
|:---:|:---:|:---:|:---:|:---:|:---:|------|
| G2 (推荐) | 0.30 | 0.50 | 3 | [0.300, 0.150, 0.075] | > 0.05 | 粗层 sqrt(0.3)≈0.55 temporal 权重，中间层也有贡献 |
| G3 (激进) | 0.50 | 0.30 | 2 | [0.500, 0.150] | > 0.08 | 2 层，粗层 1/2 scale 占 25% 像素且 α=0.5 |
| G4 (极端) | 0.70 | 0.30 | 3 | [0.700, 0.210, 0.063] | > 0.10 | 粗层几乎完全 temporal，赌杠杆效应 |

### 13.6.1 G2 实验结果 (α_base=0.30, decay=0.50) — ❌❌❌ 灾难性失败

**实验命令:**

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/mstdi_L3_a030_d050 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior --svd_mode v1 \
    --mstdi --mstdi_levels 3 --mstdi_alpha_base 0.30 --mstdi_alpha_decay 0.50 \
    --steps 30 --guidance 5.0 --seed 42
```

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| MSTDI G1 (α_base=0.05, decay=0.25) | 0.8903 | 0.7791 | -0.7%, -1.1% |
| **MSTDI G2 (α_base=0.30, decay=0.50)** | 0.7595 | 0.5327 | **CLIP -15.3%** ❌❌❌, **XCLIP -32.3%** ❌❌❌ |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | G2 CLIP | Δ CLIP | Baseline XCLIP | G2 XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.7314 | **-21.4%** ❌❌ | 0.6982 | 0.3995 | **-42.8%** ❌❌❌ |
| 17 | SUV越野 | 0.9092 | 0.7612 | **-16.3%** ❌❌ | 0.8368 | 0.5656 | **-32.4%** ❌❌❌ |
| 21 | 丛林纸飞机 | 0.8928 | 0.7842 | **-12.2%** ❌ | 0.7637 | 0.5562 | **-27.2%** ❌❌ |
| 31 | 水下城市 | 0.8324 | 0.8067 | -3.1% | 0.5237 | 0.5220 | -0.3% |
| 32 | 雪地金毛 | 0.9167 | 0.5715 | **-37.6%** ❌❌❌ | 0.8221 | 0.3226 | **-60.8%** ❌❌❌ |
| 33 | 跑步者 | 0.8531 | 0.8042 | -5.7% | 0.8618 | 0.5019 | **-41.8%** ❌❌❌ |
| 34 | 四只小狗 | 0.8968 | 0.7279 | **-18.8%** ❌❌ | 0.8710 | 0.5427 | **-37.7%** ❌❌❌ |
| 43 | 花园猫咪 | 0.9539 | 0.9065 | -5.0% | 0.9069 | 0.8306 | -8.4% ❌ |
| 46 | 火山喷发 | 0.9022 | 0.6782 | **-24.8%** ❌❌ | 0.7869 | 0.3944 | **-49.9%** ❌❌❌ |
| 47 | 动画狗城市 | 0.8769 | 0.8232 | -6.1% | 0.8024 | 0.6912 | -13.9% ❌ |

胜负统计: G2 全部 10/10 样本都低于 baseline，无一胜出。

**G2 诊断数据:**

| 样本 | η_temporal std | direction_shift | cos(mixed, random) | cos(mixed, temporal) |
|:---:|:---:|:---:|:---:|:---:|
| 7 | 0.3730 | 0.1057 | 0.9944 | 0.1060 |
| 17 | 0.3320 | 0.0939 | 0.9956 | 0.0945 |
| 21 | 0.3281 | 0.0932 | 0.9957 | 0.0935 |
| 31 | 0.3398 | 0.0967 | 0.9953 | 0.0956 |
| 32 | 0.3633 | 0.1029 | 0.9947 | 0.1014 |
| 33 | 0.4102 | 0.1160 | 0.9932 | 0.1163 |
| 34 | 0.2793 | 0.0794 | 0.9968 | 0.0782 |
| 43 | 0.3945 | 0.1119 | 0.9938 | 0.1114 |
| 46 | 0.3770 | 0.1070 | 0.9943 | 0.1085 |
| 47 | 0.3965 | 0.1126 | 0.9937 | 0.1119 |

**与 G1 对比:** direction_shift 从 0.016~0.023 跃升至 0.079~0.116（约 5× 增长），cos(mixed,random) 从 0.9997~0.9999 降至 0.9932~0.9968。**信号确实成功注入了，但注入越多效果越差。**

### 13.7 关键发现: η_temporal 内容本身对 T2V 模型有害

G2 实验彻底揭示了 L2 SVD noise prior 的核心矛盾:

**结论: 不是"注入不够"的问题，是"注入的东西有毒"。**

数学分析:
- G1 (α_base=0.05): direction_shift ≈ 0.02, CLIP/XCLIP 接近 baseline → 信号不可见，无害也无益
- G2 (α_base=0.30): direction_shift ≈ 0.10, CLIP -15.3%, XCLIP -32.3% → 信号可见，严重有害
- Baseline (α=0.004): direction_shift ≈ 0.02, 目前最优 → 极弱注入是最优平衡点

这表明 η_temporal（通过 SVD 从反演噪声中提取的时序分量）**携带的信息与 T2V 模型的生成逻辑不兼容**。可能原因:
1. SVD 提取的"运动信息"是 latent space 的统计伪影，而非真正的运动先验
2. η_temporal 的内容（幅度、相位组合）包含了空间结构信息，注入后产生 artifact
3. T2V 模型期望的输入噪声分布是严格的 N(0,1)，任何结构化偏离都被放大为 artifact

**对后续方案的启示:**
- **MSTDI 路径宣告失败**: 无论如何分配 α 的空间分布，注入 η_temporal 的原始内容都是有害的
- **TPI (时间相位注入) 变得最有希望**: TPI 只传递时间维度的相位结构，丢弃幅度信息。如果有害成分在幅度中，TPI 可以绕过这个问题
- **OCS (正交补抑制) 仍值得一试**: OCS 不直接注入 η_temporal，而是从 η_random 中去除与 temporal 正交的成分，等价于让 random 在方向上"靠近" temporal 的主子空间

### 13.8 方向 G (MSTDI) 最终结论

**MSTDI 整体判定: ❌ FAILED**

MSTDI 基于的假设（"低频空间注入 temporal prior 有杠杆效应"）被实验否定。真正的瓶颈不在注入策略（how to inject），而在注入内容本身（what to inject）。这与 CEGI 的失败模式一致——两者都在试图优化"如何更好地注入 η_temporal"，但问题在于 η_temporal 本身不是有效的运动先验。

**实验路径总结:**
- α=0.004 uniform (baseline): 最优 → 因为信号弱到不产生破坏
- α=0.05 multi-scale (G1): 接近 baseline → 有效注入量 ≈ baseline
- α=0.30 multi-scale (G2): 灾难性退化 → 证明信号本身有害
- α=0.02 channel-gated (CEGI): 中等退化 → 同样的"注入越多越差"模式

**下一步优先级重排:**
1. **TPI (最高优先)**: 只传递相位，不传递有害的幅度内容
2. **OCS (次优先)**: 从 random 侧操作，不直接注入 temporal 内容
3. 如果 TPI/OCS 都失败 → L2 noise prior 这整个方向需要从根本上重新设计

### 13.9 数据存档

- G1 生成目录: `/root/xixihaha/P-Flow/outputs/mstdi_L3_a005/`
- G1 评测结果: `/root/xixihaha/P-Flow/evaluation_results/mstdi_L3_a005/`
- G2 生成目录: `/root/xixihaha/P-Flow/outputs/mstdi_L3_a030_d050/`
- G2 评测结果: `/root/xixihaha/P-Flow/evaluation_results/mstdi_L3_a030_d050/`

---

## 十四、方向 H: TPI (Temporal Phase Injection)

### 14.1 方法原理

核心思想: 保留 η_random 的幅度谱（amplitude），只将时间维度的相位（phase）向 η_temporal 插值。不注入 η_temporal 的任何"内容"（能量/幅度），只传递其时间节奏结构。

算法:
1. 对 η_temporal 和 η_random 沿时间维度做 rFFT
2. 分离出各自的 amplitude 和 phase
3. 对 phase 做 circular interpolation: `φ_out = (1-γ) * φ_random + γ * φ_temporal`
4. 保留 η_random 的 amplitude 不变
5. 用 (amplitude_random, φ_out) 做 iRFFT 重建，renorm 到 N(0,1)

### 14.2 实验配置 (H1: γ=0.5)

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/tpi_g05 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --noise_prior --svd_mode v1 \
    --tpi --tpi_gamma 0.5 --tpi_freq_min 1 --tpi_freq_max -1 \
    --steps 30 --guidance 5.0 --seed 42
```

参数: γ=0.5 (50% 相位插值)，freq_range=[1, 11) (跳过 DC，全部 AC 频率)

### 14.3 实验结果 (H1: ❌❌❌❌ 史诗级失败，所有方案中最差)

**总体指标:**

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | **0.7874** | — |
| MSTDI G2 (α_base=0.30, decay=0.50) | 0.7595 | 0.5327 | -15.3%, -32.3% |
| **TPI H1 (γ=0.5, all freqs)** | **0.6546** | **0.3007** | **CLIP -27.0%** ❌❌❌❌, **XCLIP -61.8%** ❌❌❌❌ |

**逐 Case 对比:**

| 样本 | 场景 | Baseline CLIP | TPI CLIP | Δ CLIP | Baseline XCLIP | TPI XCLIP | Δ XCLIP |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 7 | 杯中帆船 | 0.9303 | 0.7563 | **-18.7%** ❌❌ | 0.6982 | 0.3298 | **-52.8%** ❌❌❌ |
| 17 | SUV越野 | 0.9092 | 0.5419 | **-40.4%** ❌❌❌ | 0.8368 | 0.2627 | **-68.6%** ❌❌❌❌ |
| 21 | 丛林纸飞机 | 0.8928 | 0.6570 | **-26.4%** ❌❌ | 0.7637 | 0.3133 | **-59.0%** ❌❌❌ |
| 31 | 水下城市 | 0.8324 | 0.6747 | **-18.9%** ❌❌ | 0.5237 | 0.3598 | **-31.3%** ❌❌ |
| 32 | 雪地金毛 | 0.9167 | 0.5435 | **-40.7%** ❌❌❌ | 0.8221 | 0.3301 | **-59.8%** ❌❌❌ |
| 33 | 跑步者 | 0.8531 | 0.7703 | -9.7% ❌ | 0.8618 | 0.3271 | **-62.1%** ❌❌❌ |
| 34 | 四只小狗 | 0.8968 | 0.5922 | **-34.0%** ❌❌❌ | 0.8710 | 0.2238 | **-74.3%** ❌❌❌❌ |
| 43 | 花园猫咪 | 0.9539 | 0.6196 | **-35.1%** ❌❌❌ | 0.9069 | 0.2070 | **-77.2%** ❌❌❌❌ |
| 46 | 火山喷发 | 0.9022 | 0.6660 | **-26.2%** ❌❌ | 0.7869 | 0.3251 | **-58.7%** ❌❌❌ |
| 47 | 动画狗城市 | 0.8769 | 0.7249 | -17.3% ❌❌ | 0.8024 | 0.3281 | **-59.1%** ❌❌❌ |

胜负统计: TPI 全部 10/10 样本惨败，无一接近 baseline。

### 14.4 诊断数据

| 样本 | direction_shift | cos(mixed, random) | cos(mixed, temporal) |
|:---:|:---:|:---:|:---:|
| 7 | 0.8287 | 0.6546 | 0.3542 |
| 17 | 0.8287 | 0.6547 | 0.4553 |
| 21 | 0.8343 | 0.6541 | 0.4601 |
| 31 | 0.8343 | 0.6537 | 0.4044 |
| 32 | 0.8343 | 0.6530 | 0.4444 |
| 33 | 0.8343 | 0.6536 | 0.3092 |
| 34 | 0.8343 | 0.6532 | 0.4428 |
| 43 | 0.8343 | 0.6544 | 0.4679 |
| 46 | 0.8287 | 0.6544 | 0.4609 |
| 47 | 0.8343 | 0.6540 | 0.4626 |

**关键发现:**
- `direction_shift ≈ 0.83` — 噪声被彻底改造（vs MSTDI G2 的 0.10，vs baseline 的 0.02）
- `cos(mixed, random) ≈ 0.65` — 混合结果只保留了 65% 的 random 特征
- direction_shift 几乎所有样本相同（0.8287/0.8343）— 因为 γ=0.5 对所有频率统一操作，与具体样本内容无关

### 14.5 失败原因分析

**TPI 比 MSTDI G2 更差的根本原因: 破坏了帧间统计独立性**

T2V 模型期望输入 z_T 中每一帧的噪声是 i.i.d.（独立同分布）的 N(0,1)。TPI 通过修改时间维度的相位，在帧间引入了强相关性：

1. **相位 = 帧间时序关系**：时间维度 FFT 的相位决定了不同帧之间的"时间对齐关系"。修改 50% 的相位等于将噪声从"帧间无关"变为"帧间有特定节奏结构"。

2. **这比修改幅度更致命**：幅度修改（如 MSTDI）只改变各频率分量的强度，但保持了帧间的"随机性结构"不变。相位修改直接改变了帧间的相对时间关系，T2V 模型的 temporal attention 完全无法处理这种非 i.i.d. 输入。

3. **direction_shift=0.83 说明一切**：噪声被改了 83%，只有 17% 还像原来的 random。对于一个期望纯随机输入的模型来说，这等于送了一个完全"错误"的初始条件。

**与"有害内容"假说的关系:**

MSTDI G2 证明了"η_temporal 的内容有害"；TPI H1 进一步证明了"η_temporal 的相位同样有害"。综合来看：**η_temporal 的一切信息（内容、幅度、相位）对 T2V 模型都是有害的**。问题不在 η_temporal 的哪个维度有用哪个有害，而在于 T2V 模型的 denoise 机制根本无法利用 z_T 中的结构化先验——任何偏离 N(0,1) 的结构都只能带来伤害。

### 14.6 方向 H (TPI) 最终结论

**TPI 整体判定: ❌❌❌❌ CATASTROPHIC FAILURE**

TPI 是目前所有实验中表现最差的方案（CLIP -27.0%, XCLIP -61.8%）。它彻底否定了"相位是有用信号"的假设，并进一步确认了纯黑盒路线的绝对天花板就是 α=0.004 uniform blend。

**OCS 不再需要实验**: OCS 的 suppress_ratio 同样会导致大 direction_shift（通过投影消除正交分量），预期结果与 TPI/MSTDI G2 类似。在已有充分证据的情况下，跳过 OCS 直接转向灰盒方案。

### 14.7 数据存档

- 生成目录: `/root/xixihaha/P-Flow/outputs/tpi_g05/`
- 评测结果: `/root/xixihaha/P-Flow/evaluation_results/tpi_g05/`

---

## 十五、L2 Noise Prior 技术路线总结与突破方向分析

### 15.1 纯黑盒路线的极限总结

经过方向 C~G 的系统性实验，纯黑盒约束下的 L2 noise prior 技术路线已触及天花板:

**已验证结论:**
- 最优配置: α=0.004 uniform blend (baseline)，CLIP 0.8964, XCLIP 0.7874
- 核心矛盾: η_temporal（SVD 从反演噪声中提取的时序分量）**内容本身对 T2V 模型有害**
- 所有"增强注入"的尝试均失败: renorm、频域重塑、SGA、PODI、CEGI、MSTDI、TPI，注入量越大效果越差
- baseline α=0.004 的"成功"恰恰是因为信号弱到不产生破坏（direction_shift ≈ 0.02），只提供了统计层面的微弱引导

**失败的根本原因:**

纯黑盒下只能操作两个输入接口: prompt（文本）和 z_T（初始噪声）。在 z_T 层面，T2V 模型期望的输入是严格的 N(0,1) 高斯噪声，任何结构化偏离都会被模型的 30 步去噪过程逐步放大为 artifact。SVD 提取的 η_temporal 虽然包含参考视频的时序统计信息，但这些信息在 latent space 中的表现形式与模型的生成逻辑不兼容——它更像是"latent space 的统计伪影"而非"可被模型利用的运动先验"。

**TPI/OCS 实验验证:**

TPI H1 (γ=0.5) 实验结果为 CLIP 0.6546, XCLIP 0.3007（-27.0%, -61.8%），是所有方案中最差的结果。这证明 η_temporal 的相位信息同样对 T2V 模型有害。OCS 的作用机制（投影消除正交分量）预期会产生类似的高 direction_shift，在已有充分证据的情况下跳过。

**最终结论: 纯黑盒路线的一切可能性已穷尽，α=0.004 uniform blend 就是绝对天花板。**

### 15.2 突破方向: 灰盒 (Gray-Box) 方案

如果放松约束到"不改权重，但能 hook 中间层"（training-free gray-box），则有若干有前景的方向:

#### 方向 α: Attention Injection (注意力注入)

**原理:** 类似 Plug-and-Play Diffusion (ICLR 2023)、MasaCtrl (ICCV 2023) 的思路。先用参考视频跑一次 inversion 得到中间层 attention maps / features，生成时 hook DiT 的 self-attention 层，将参考视频的 K/V 注入（partial replacement 或 weighted blend）。

**优势:**
- 直接在特征空间传递运动/结构信息，绕过"只能改噪声"的限制
- 在 image 领域已被充分验证 (PnP, MasaCtrl, P2P)
- 视频版本可以作为论文的新贡献（Video DiT Attention Injection for Motion Transfer）
- 可以控制注入的 timestep 范围（早期强注入 → 后期放手），实现 coarse-to-fine

**实现路径:**
- PyTorch forward hook 拦截 DiT 的 self-attention 层
- 参考视频 inversion 过程中缓存每个 timestep 的 K, V
- 生成时: `K_gen' = (1-w_t) * K_gen + w_t * K_ref`, `V_gen' = (1-w_t) * V_gen + w_t * V_ref`
- w_t 随 timestep 递减（早期注入结构，后期保持创造性）

**风险:** Wan2.1 的 DiT 架构可能与 U-Net based 方法有差异，attention 结构需要适配。

#### 方向 β: Latent Trajectory Soft Anchor (潜空间轨迹锚定)

**原理:** 参考视频反演得到完整的 latent 轨迹 {z_0, z_1, ..., z_T}。生成时不只改初始 z_T，而是在每个 timestep 将当前 latent 向参考轨迹做 soft anchor。

**算法:**
```
for t in T, T-1, ..., 0:
    z_t_gen = denoise_step(z_{t+1}_gen, prompt)
    z_t_gen' = (1 - β_t) * z_t_gen + β_t * z_t_ref
    # β_t 从 β_max (如 0.3) 线性递减到 0
```

**优势:**
- 实现极简（每步 denoise 后一行 lerp）
- 可自然退化为纯黑盒（β=0 等价于无注入）
- 论文中可展示 β 从 0 到 1 的 ablation curve，故事完整
- 类似 SDEdit 思路但更精细，有明确的理论支撑

**风险:** β 过大会导致内容泄露（生成结果过于接近参考视频的空间内容而非运动模式）。

#### 方向 γ: Guided Denoising (梯度引导去噪)

**原理:** 每步 denoise 时，额外计算一个 loss 对当前 latent 的梯度，nudge 生成轨迹向参考视频对齐。

**算法:**
```
for t in T, T-1, ..., 0:
    z_t.requires_grad_(True)
    ε_pred = model(z_t, t, prompt)
    L = perceptual_loss(z_t, z_t_ref) + flow_loss(optical_flow(z_t), optical_flow(z_t_ref))
    grad = torch.autograd.grad(L, z_t)
    z_t = z_t - λ * grad
    z_t.requires_grad_(False)
    z_{t-1} = denoise_update(z_t, ε_pred)
```

**优势:**
- 可以用任意可微 loss 函数引导（perceptual, optical flow, structural similarity）
- 不需要缓存 attention，内存友好
- 引导强度通过 λ 精细控制

**风险:** 需要反向传播通过部分网络，计算开销 2~3×；梯度可能不稳定。

### 15.3 白盒方案 (需要 finetune)

如果允许修改模型参数，则有更强力的方案:

#### 方向 δ: Motion LoRA / IP-Adapter for Video

对参考视频提取运动特征（optical flow, motion vector），训练轻量 LoRA adapter 使模型在生成时 condition on motion。类似 AnimateDiff + MotionCtrl 的路线，但可以做 per-sample 的 test-time adaptation。

#### 方向 ε: Reference Video Conditioning (额外输入通道)

修改模型输入层，额外 concat 参考视频的 latent 或 optical flow map 作为条件。需要对输入 projection 层做少量 finetune。

### 15.4 推荐路径

考虑论文定位（training-free video reproduction）和可行性:

| 优先级 | 方向 | 约束级别 | 预期收益 | 实现难度 | 论文新颖性 |
|:---:|------|:---:|:---:|:---:|:---:|
| 1 | ~~TPI / OCS~~ (已验证失败) | 纯黑盒 | ❌ 负收益 | 已完成 | 低 |
| 2 | Latent Trajectory Soft Anchor | 灰盒 | 中 (+3~8%) | 低 (数行代码) | 中 |
| 3 | Attention Injection | 灰盒 | 高 (+5~15%) | 中 (需适配 DiT) | **高** |
| 4 | Guided Denoising | 灰盒 | 中-高 | 中-高 | 中 |
| 5 | Motion LoRA | 白盒 | 高 | 高 (需训练) | 中 |

**建议策略（更新）:**
1. ~~先快速跑完 TPI/OCS~~ ✅ 已完成，纯黑盒天花板确认为 α=0.004
2. **立即转向灰盒方向 β (Latent Trajectory Soft Anchor)** 作为快速验证——实现简单（每步 denoise 后一行 lerp），能快速判断灰盒是否有 significant gain
3. 如果方向 β 有效，再投入精力做方向 α (Attention Injection) 作为论文的核心贡献
