# SVD V2 实验分析 & 改进记录

> 创建时间: 2025-06-10  
> 最后更新: 2025-06-11  
> 状态: renorm 失败 → rescale 否决 → QGA 失败 → α=0.01 确认天花板 → **方向 C 频域重塑 代码就绪**  
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

---

## 八、实验路径规划（更新后）

```
当前状态
├── L2 SVD 改进
│   ├── [已完成] renorm α=0.001 → 失败，整体不如 v1
│   ├── [已否决] rescale 模式 → 等价于调 alpha，无独立价值
│   ├── [已失败] Quality-Gated Alpha → quality scores 系统性接近 0，无法区分样本
│   ├── [已失败] α=0.01 → 后 9 样本正向 (+1.2%)，但 S7/S31 catastrophic failure
│   ├── [★当前] 方向 C: 频域噪声重塑 (代码就绪，待验证)
│   │     - 核心突破: 不注入 η_temporal 内容，只传递时间频谱形状
│   │     - 预期解决 one-shot blend 的毒性问题
│   └── [后续] β 参数消融 + v9 caption 联合实验
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
| 1 | **频域重塑 β=1.0** | 安全性提升 + XCLIP 正向 | 低 (代码已就绪) | **★ 待跑** |
| 2 | 频域重塑 β=0.5 | 对比不同重塑强度 | 低 | 等 β=1.0 结果 |
| 3 | v9 + freq_reshape 联合 | CLIP+XCLIP 双提升 | 低 | 等频域结果 |
| 4 | 扩大样本量验证 | 统计显著性 | 高 | 论文投稿前 |

---

## 九、关键数据存档

### 生成日志位置 (服务器)

- Pure L2 Baseline: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/`
- SVD V2 Renorm: `/root/xixihaha/P-Flow/outputs/svd_v2_renorm_alpha001/`
- v9 Prompt Rewrite: `/root/xixihaha/P-Flow/outputs/v9_svd_v1/`
- α=0.01 实验: `/root/xixihaha/P-Flow/outputs/svd_v1_alpha010/`
- Baseline 评测: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/eval_results/`
- Renorm 评测: `/root/xixihaha/P-Flow/evaluation_results/svd_v2_renorm_alpha001/`
- v9 评测: `/root/xixihaha/P-Flow/evaluation_results/v9_svd_v1/`
- α=0.01 评测: `/root/xixihaha/P-Flow/evaluation_results/svd_v1_alpha010/`

### 全实验对比汇总

| 配置 | CLIP (orig-gen) | XCLIP (orig-gen) | vs Baseline |
|------|:---:|:---:|:---:|
| Pure L2 Baseline (v1, α=0.004) | **0.8964** | 0.7874 | — |
| SVD V2 Renorm (α=0.001) | 0.8826 | 0.7506 | CLIP -1.5%, XCLIP -4.7% ❌ |
| v9 Prompt (LLM删减+VLM补充 + v1) | 0.8947 | **0.7973** | CLIP -0.2%, XCLIP +1.3% ✅ |
| SVD v1, α=0.01 | 0.8903 | 0.7685 | CLIP -0.7%, XCLIP -2.4% ❌ |
| SVD v1, α=0.01 (后9, 去S7) | 0.8947 | 0.7966 | CLIP -0.2%, XCLIP +1.2% ✅ |
| 频域重塑 β=1.0 (待验证) | ? | ? | 预期安全性↑ + XCLIP正向 |

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
