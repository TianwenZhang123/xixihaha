# SVD V2 Renorm 实验分析 & 下一步计划

> 创建时间: 2025-06-10  
> 状态: SVD renorm 实验完成分析，prompt rewrite v9 待验证  
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

## 四、下一步改进方案

### 方案 A: Direction-Preserving Rescale（推荐，最小改动）

**核心思路**: 不做 (x-mean)/std 的全量归一化，而是做等比缩放，保留方向间相对比例。

```python
# 目标: 让每个样本的有效注入量接近 v1 最优中位数
# v1 中位 effective = sqrt(0.004) * median_std = 0.0632 * 0.37 ≈ 0.0234
target_effective = 0.0234
current_effective = sqrt(alpha) * eta_temporal.std()

if current_effective < target_effective:
    scale = target_effective / current_effective
    eta_temporal = eta_temporal * scale
# 如果已经够强则不动
```

**优点**: 只放大信号不足的样本 (std < 0.37)，不干扰已经足够的样本。保留方向结构。

**实现位置**: `src/svd_filter.py` 新增 `mode="rescale"` 分支

### 方案 B: Quality-Gated Alpha（更灵活）

根据 SVD 方向质量动态调节 alpha:

```python
quality = compute_direction_quality(eta_temporal)
# quality 综合: cos(Δ_first, Δ_last), 空间不均匀度, 帧间余弦均值

effective_alpha = base_alpha * (0.5 + quality)
# quality=0 → alpha 减半 (几乎不注入)
# quality=1 → alpha 1.5x (强注入)
```

**适合论文叙事**: "Sample-Adaptive Prior Injection"

### 方案 C: Conditional Renorm（有选择地 renorm）

仅对 std < 阈值 + 方向质量 > 阈值的样本启用 renorm:

```python
if eta_temporal.std() < 0.32 and direction_quality > 0.5:
    eta_temporal = renormalize(eta_temporal)
    alpha = 0.001  # renorm 后用小 alpha
else:
    alpha = 0.004  # 保持 v1 原样
```

### 建议实验顺序

1. **先跑方案 A (rescale)**: 改动最小、风险最低，能快速验证"保方向缩放"是否优于"全量归一化"
2. **如果 A 有效**: 进一步加 quality gating (方案 B)，针对论文写 ablation
3. **同步进行 v9 prompt rewrite 实验** (见下一章)

---

## 五、Prompt Rewrite v9 — 待验证

### 5.1 状态

代码已写好: `P-Flow/scripts/rewrite_minimal.py` (v9-vlm, 836 行)

策略设计:
- **Step 1: LLM pure subtraction** — 只删不加 (去 preamble + hedging + summary)
- **Step 2: VLM visual supplement** — 用 DashScope VLM 看原始视频，补充视觉细节

解决 v8-minimal 的问题: v8 的 CLIP 0.8915 < Pure L2 的 0.8964，根因是 LLM 虚构了不接地气的 camera sentences。v9 用 VLM 确保每个新增细节都是真实视觉事实。

### 5.2 运行命令

```bash
cd /root/xixihaha/P-Flow

# Step 1: 生成 v9 captions
python scripts/rewrite_minimal.py \
    --input-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir data/captions_v9 \
    --video-dir data/videos \
    --backend dashscope \
    --model qwen-plus \
    --vlm-provider dashscope \
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

### 5.3 预期

v9 的核心假设是: VLM 补充的视觉细节比 LLM 虚构的更准确 → CLIP 不降（甚至提升），XCLIP 至少维持。

如果 v9 caption + SVD v1 超过 Pure L2 baseline，说明 L1 层的正确路径是 "删减 + VLM 补充" 而非 "LLM 改写"。

---

## 六、实验路径规划

```
当前状态
├── L2 SVD 改进
│   ├── [已完成] renorm α=0.001 → 失败，整体不如 v1
│   ├── [下一步] rescale 模式实验 (方案 A)
│   │     - 修改 svd_filter.py 添加 rescale mode
│   │     - 用 α=0.004 跑 10 样本
│   │     - 对比 v1 baseline
│   └── [备选] quality-gated alpha (方案 B)
│
├── L1 Prompt Rewrite
│   ├── [已完成] v8-minimal → 失败 (CLIP 0.8915 < baseline 0.8964)
│   ├── [待验证] v9-vlm (代码就绪)
│   │     - 先单独跑 v9 caption + v1 SVD
│   │     - 如果 v9 > Pure L2，则 v9 + rescale SVD 联合测试
│   └── [长期] 如果 v9 仍然不行 → 放弃 L1 改写，全力做 L2+L3
│
└── L3 Velocity Matching
    └── [暂缓] 等 L1+L2 稳定后再叠加
```

### 优先级排序

| 序号 | 实验 | 预期收益 | 成本 | 理由 |
|:---:|------|:---:|:---:|------|
| 1 | v9 prompt rewrite 验证 | CLIP ≥ baseline | 中 (需 VLM 推理) | 决定 L1 层是否保留 |
| 2 | SVD rescale 模式 | XCLIP +1~3% | 低 (改几行代码) | 修复 renorm 的一刀切问题 |
| 3 | Quality-gated alpha | XCLIP +2~4% | 低 | 在 rescale 基础上进一步优化 |
| 4 | 扩大样本量验证 | 统计显著性 | 高 | 论文投稿前必做 |

---

## 七、关键数据存档

### 生成日志位置 (服务器)

- Pure L2 Baseline: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/`
- SVD V2 Renorm: `/root/xixihaha/P-Flow/outputs/svd_v2_renorm_alpha001/`
- Baseline 评测: `/root/xixihaha/P-Flow/outputs/pure_svd_v1_baseline/eval_results/`
- Renorm 评测: `/root/xixihaha/P-Flow/evaluation_results/svd_v2_renorm_alpha001/`

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
