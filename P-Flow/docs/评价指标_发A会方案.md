# P-Flow 发 A 会评价指标方案

> 本文档定义 P-Flow 投稿 A 类会议（CVPR/ICCV/NeurIPS/AAAI）所需的完整评价指标体系。
> 当前已有指标：CLIP-SIM、X-CLIP。本文档补全缺失维度，确保审稿人无可挑剔。
>
> 整理时间：2025年6月 | 基于 VBench (CVPR 2024)、EvalCrafter (CVPR 2024)、OmniDirector (2026) 等顶会论文的评价标准

---

## 一、指标总览

| 维度 | 指标 | 方向 | 现有？ | 优先级 | 说明 |
|------|------|------|--------|--------|------|
| 语义一致性 | CLIP-SIM | ↑ | ✅ | — | 帧级视觉相似度（原始 vs 生成） |
| 时序语义 | X-CLIP | ↑ | ✅ | — | 时序感知的视频级语义相似度 |
| 分布质量 | FVD | ↓ | ❌ | **P0** | 生成/真实视频集的分布距离 |
| 细粒度一致性 | DINO-Score | ↑ | ❌ | **P0** | 帧间主体身份保持度 |
| 运动保真度 | Flow Consistency (EPE) | ↓ | ❌ | **P0** | 光流场一致性，直接验证 SVD 效果 |
| 感知距离 | LPIPS | ↓ | ❌ | **P1** | 深度感知特征距离 |
| 时序平滑度 | Temporal Flickering | ↓ | ❌ | **P1** | 高频闪烁检测 |
| 动态程度 | Dynamic Degree | ↑ | ❌ | **P2** | 证明保真度提升不以牺牲动态性为代价 |
| 人工评估 | User Study (Win Rate) | — | ❌ | **P0** | A 会必需，审稿人必问 |

**P0** = 必须有（缺了审稿人一定会要求补充）；**P1** = 强烈建议；**P2** = 锦上添花。

---

## 二、P0 指标详细说明

### 2.1 FVD (Fréchet Video Distance) ↓

**用途**：衡量生成视频与真实视频在特征空间的分布距离，是视频生成论文的"标配"指标。

**计算方法**：

```
1. 对每个视频取 16 帧，通过预训练 I3D (Kinetics-400) 提取 400-dim 特征
2. 分别计算真实集 R 和生成集 G 的特征均值 μ 和协方差 Σ
3. FVD = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r·Σ_g)^{1/2})
```

**实现工具**：`pytorch-fvd` 或 `stylegan-v` 的评测模块

**注意事项**：
- 需要足够多的样本（≥256 个视频）才能稳定，我们 200 样本数据集刚好满足
- I3D 对时序扰动不太敏感（已知局限），所以不能只靠 FVD，需配合其他指标

**对 P-Flow 的意义**：证明我们的方法在总体视频分布质量上不逊于（甚至优于）baseline。

---

### 2.2 DINO-Score (Subject Consistency) ↑

**用途**：用 DINOv2 的实例级特征衡量生成视频帧间主体一致性。比 CLIP 更能捕捉细粒度视觉差异。

**计算方法**：

```
1. 对生成视频逐帧提取 DINOv2 ViT-B/14 的 [CLS] token 特征
   f_t = DINOv2(frame_t)

2. 计算相邻帧间余弦相似度：
   sim_t = cos(f_t, f_{t+1})

3. 取所有帧对均值：
   DINO_Score = (1/(T-1)) · Σ sim_t
```

**为什么用 DINO 而非 CLIP**：DINO 自监督训练使其对物体外观/纹理/形状极其敏感（实例级），而 CLIP 偏向语义级。DINO 能检测"同一辆车是否始终是同一辆车"这类身份一致性。

**对 P-Flow 的意义**：Feature Injection (L3) 的核心目标是注入参考视频的外观细节，DINO-Score 直接衡量这一效果。

---

### 2.3 Flow Consistency / EPE (Endpoint Error) ↓

**用途**：直接对比原始视频和生成视频的光流场，量化运动轨迹保真度。**这是验证 SVD Noise Prior (L2) 最直接的指标。**

**计算方法**：

```
方案A：光流场端点误差 (EPE)
1. 用 RAFT 分别估计原始视频和生成视频的光流场：
   F_orig_t = RAFT(orig_t, orig_{t+1})
   F_gen_t  = RAFT(gen_t, gen_{t+1})

2. 计算逐像素端点误差：
   EPE_t = mean_p( ||F_orig_t(p) - F_gen_t(p)||₂ )

3. 取所有帧对均值：
   Flow_EPE = (1/(T-1)) · Σ EPE_t

方案B：光流余弦相似度 (Flow Cosine Similarity) ↑
1. 同上提取光流场
2. 将光流场展平后计算余弦相似度：
   flow_sim_t = cos(flatten(F_orig_t), flatten(F_gen_t))

3. 取均值：
   Flow_Sim = mean(flow_sim_t)
```

**对 P-Flow 的意义**：
- L2 (SVD noise prior) 的设计目标是保留原始视频的运动模式，Flow EPE 直接度量这一目标
- 可以做消融：无 SVD vs 有 SVD 的 Flow EPE 对比，量化 SVD 的运动保留效果
- 比 X-CLIP 更精确——X-CLIP 是语义级的时序理解，Flow EPE 是像素级的运动轨迹对比

**实现**：RAFT (torchvision 已内置) 或 UniFlow

---

### 2.4 User Study (人工评估) — A 会必需

**设计方案**：

```
评估形式：A/B Preference Test (双盲)

评估维度（每维度独立打分）：
  1. Visual Fidelity — "哪个生成视频在视觉外观上更像原始视频？"
  2. Motion Consistency — "哪个生成视频的运动模式更像原始视频？"
  3. Overall Quality — "整体而言，哪个生成视频质量更高？"

对比组：
  - Ours (L1+L2+L3) vs Baseline (Raw Caption)
  - Ours (L1+L2+L3) vs Ablation (L2 only)
  - Ours (L1+L2+L3) vs 同类方法 (如适用)

规模要求：
  - 至少 50 组对比视频
  - 10-20 名评估者（非作者）
  - 报告 win rate (%) + 95% 置信区间或 p-value

展示方式：
  - 同时播放原始视频 + 两个生成视频（A/B 随机排列）
  - 每组评估者看相同视频但 A/B 顺序随机化
```

**报告格式示例**：

| 对比 | Visual Fidelity | Motion Consistency | Overall Quality |
|------|:---:|:---:|:---:|
| Ours vs Raw Caption | 78.2% / 21.8% | 82.4% / 17.6% | 76.0% / 24.0% |
| Ours vs L2 Only | 65.3% / 34.7% | 52.1% / 47.9% | 61.8% / 38.2% |

---

## 三、P1 指标详细说明

### 3.1 LPIPS (Learned Perceptual Image Patch Similarity) ↓

**用途**：逐帧计算生成视频与原始视频对应帧之间的感知距离。比 SSIM/PSNR 更符合人类感知。

**计算方法**：

```
1. 对原始/生成视频逐帧对齐（我们从同一首帧生成，天然对齐）
2. 逐帧计算 LPIPS：
   lpips_t = LPIPS_VGG(orig_t, gen_t)

3. 取所有帧均值：
   LPIPS_avg = (1/T) · Σ lpips_t
```

**实现**：`pip install lpips`，使用 VGG backbone（与人类感知相关性最高）。

**对 P-Flow 的意义**：补充 CLIP-SIM 的不足——CLIP 对全局语义敏感但对局部纹理不敏感，LPIPS 能检测局部视觉失真。

---

### 3.2 Temporal Flickering ↓

**用途**：检测相邻帧之间的高频闪烁（纹理跳动、光照突变）。

**计算方法**：

```
1. 计算相邻帧逐像素绝对差：
   diff_t = |gen_t - gen_{t+1}|

2. 取每帧对的 MAE：
   MAE_t = mean(diff_t)

3. 转换为得分：
   Flicker_Score = mean(MAE_1, ..., MAE_{T-1})
```

**范围**：越低越好（0 = 完全无闪烁）。

**注意**：对高运动区域需配合光流 mask 排除（否则正常运动也会计为"闪烁"）。

---

## 四、P2 指标详细说明

### 4.1 Dynamic Degree ↑

**用途**：证明我们在提升保真度的同时没有牺牲视频的动态性（避免生成"接近静止"的视频）。

**计算方法**：

```
1. 用 RAFT 计算相邻帧光流 F_t
2. 计算每帧对的光流幅度均值：
   flow_mag_t = mean_p(||F_t(p)||₂)

3. 取所有帧对均值：
   Dynamic_Degree = mean(flow_mag_t)
```

**对 P-Flow 的意义**：某些方法（如过强的 Feature Injection）可能导致生成视频"冻住"，Dynamic Degree 能揭示这一问题。我们需要证明 L3 不会抑制运动。

---

## 五、消融实验设计 (Ablation Table)

发 A 会需要完整的消融实验表，每组跑全套指标：

| 配置 | CLIP↑ | X-CLIP↑ | FVD↓ | DINO↑ | Flow EPE↓ | LPIPS↓ |
|------|-------|---------|------|-------|-----------|--------|
| Raw Caption (baseline) | — | — | — | — | — | — |
| L1 only (v9 rewrite) | — | — | — | — | — | — |
| L2 only (SVD α=0.004) | — | — | — | — | — | — |
| L1 + L2 | — | — | — | — | — | — |
| L3 only (FI) | — | — | — | — | — | — |
| L2 + L3 | — | — | — | — | — | — |
| **L1 + L2 + L3 (Full)** | — | — | — | — | — | — |

**关键消融对比**：
- L1 的贡献：对比 "L2" vs "L1+L2"，看 CLIP/X-CLIP 提升
- L2 的贡献：对比 "Raw Caption" vs "L2 only"，看 Flow EPE 大幅下降
- L3 的贡献：对比 "L2" vs "L2+L3"，看 DINO/LPIPS 提升
- 三层叠加的正交性：L1+L2+L3 vs 各种两两组合

---

## 六、与同类工作对比 (Comparison Table)

| 方法 | 类型 | CLIP↑ | X-CLIP↑ | FVD↓ | Flow EPE↓ | User Study↑ |
|------|------|-------|---------|------|-----------|-------------|
| Raw VLM Caption | Baseline | — | — | — | — | — |
| Rewriting Video (CVPR 2026) | Prompt-based | — | — | — | — | — |
| VideoAssembler | Cond. injection | — | — | — | — | — |
| **P-Flow (Ours)** | 3-layer framework | — | — | — | — | — |

---

## 七、实现优先级与工作量估算

### 第一批（P0，1-2 天可完成）

| 指标 | 依赖库 | 估算工作量 | 备注 |
|------|--------|-----------|------|
| FVD | `pytorch-fvd` + I3D 权重 | 4h | 需下载 I3D 预训练模型 |
| DINO-Score | `transformers` (DINOv2) | 2h | 逐帧提取 + 余弦相似度 |
| Flow EPE | `torchvision` (RAFT) | 4h | 提取双边光流 + 计算 EPE |

### 第二批（P1，半天）

| 指标 | 依赖库 | 估算工作量 |
|------|--------|-----------|
| LPIPS | `lpips` | 1h |
| Temporal Flickering | numpy | 0.5h |

### 第三批（P2 + User Study）

| 指标 | 估算工作量 |
|------|-----------|
| Dynamic Degree | 1h（复用 RAFT） |
| User Study 平台搭建 | 1-2 天 |
| User Study 数据收集 | 3-5 天 |

---

## 八、评测代码组织建议

```
P-Flow/
├── evaluation/
│   ├── run_clip_xclip_eval.py      # 已有
│   ├── run_fvd_eval.py             # 新增：FVD 评测
│   ├── run_dino_eval.py            # 新增：DINO-Score 评测
│   ├── run_flow_eval.py            # 新增：光流一致性 (EPE + Cosine)
│   ├── run_lpips_eval.py           # 新增：LPIPS 评测
│   ├── run_temporal_eval.py        # 新增：Temporal Flickering + Dynamic Degree
│   ├── run_all_metrics.py          # 新增：一键跑全套指标
│   └── user_study/
│       ├── generate_pairs.py       # 生成 A/B 对比视频
│       └── collect_results.py      # 统计 win rate
```

---

## 九、参考论文中的指标使用情况

| 论文 | 会议 | 使用指标 |
|------|------|----------|
| Stable Video Diffusion | ICLR 2024 | FVD, FID, CLIP-SIM, LPIPS |
| AnimateAnything | CVPR 2025 | FVD, CLIP-SIM, DINO, LPIPS, User Study |
| OmniDirector | 2026 | RRE, RTE, R-Pre, T-Pre, GSB (Gemini) |
| VBench | CVPR 2024 | 16 维度 (Subject Consistency, Motion Smoothness, Dynamic Degree, ...) |
| FrameBridge | NeurIPS 2025 | FVD, LPIPS, CLIP-SIM, DINO, User Study |
| CogVideoX | ICLR 2025 | FVD, CLIP-Score, VBench 全维度 |
| P-Flow (目标) | A 会 | CLIP, X-CLIP, FVD, DINO, Flow EPE, LPIPS, User Study |

---

## 十、注意事项

1. **样本量**：200 个视频对 FVD 来说偏少（理想 ≥2048），论文中需说明这一点并补充 per-sample 指标（如 CLIP/DINO/LPIPS 取均值 ± std）
2. **统计显著性**：所有指标需报告 mean ± std，消融实验用 paired t-test 证明差异显著
3. **Baseline 公平性**：确保所有方法使用相同的视频生成模型（Wan2.1-T2V-1.3B）、相同 seed、相同步数
4. **FVD 的局限**：在论文中可以讨论 FVD 的 content bias 问题，说明我们同时使用了 Flow EPE / DINO 等互补指标
5. **User Study 规范**：需在论文中说明评估者数量、专业背景、评估界面设计、是否通过 IRB 审批
