# 故事、技术演进与未来方向

> 本文档包含四部分：(1) 通俗故事，(2) 技术架构与原理，(3) 设计演进与简化，(4) 当前状态与优化路线。

---

## 第一部分：通俗故事 — "让 AI 精准重画任意视频"

### 一句话版本

给 AI 一段视频，让它仅凭「文字 + 噪声」重新生成一个尽可能一模一样的版本。我们的方法是一个渐进式框架，通过逐层添加信息（文字→噪声→嵌入），一步步逼近完美复现。

### 完整故事

想象你面前有一位画师（Wan2.1 文生视频模型），他只接受两样东西：一段文字描述和一张满是涂鸦的白纸（初始噪声）。你的目标是：让他画出和你手中参考视频完全一致的作品。

问题在于，一段文字远远不够。你说「一只橘猫在窗台打哈欠」，画师每次画出来的猫都不同——朝向、动作幅度、背景光线全凭自由发挥。文字只是一个模糊的意图传达。

**我们的解法是：用 flag 组合逐层逼近目标。**

第一层（文字优化 `--iter`）—— 先让 VLM 看参考视频，写一段详尽描述，然后反复对比生成结果和原视频，迭代优化措辞。每一轮 VLM 都会发现新的差异并修正 prompt。迭代 10 轮通常能达到文字的极限表达。

第二层（噪声先验 `--inversion --svd --blend`）—— 文字到了极限，画师的「起笔」就成了关键。我们通过 Flow Matching Inversion 从成品倒推起点（η_inv），再用 SVD 分离出纯运动成分（去除外观信息），最后将运动噪声与随机噪声以 α=0.004 混合（约 6:94 的比例）。这种微妙的方向性偏置就像给画师指了一个「大致方向」，不干扰创作自由但确保运动一致。

第三层（速度场匹配 `--velocity`）—— 文字和噪声都搞定后，还剩那些文字无法表达的隐含动态——比如特定的加速曲线、微妙的节奏感。我们通过优化一个 embedding 残差 Δe，让模型在生成每一帧时的「画笔运动方向」都对齐参考视频定义的理想轨迹。Δe 的注入极其轻微（仅 0.02 倍），但足以将生成视频的运动特征锁定在正确方向。

三层组合就像一个分辨率递增的编码系统：文字给出语义轮廓（"什么在动"），噪声先验提供结构引导（"从哪里开始动"），Δe 锁定精确轨迹（"怎么动"）。每一层都是前一层的残差补充，互不冗余。

### 大白话版完整流程（从头到尾发生了什么）

假设你手里有一段参考视频——一只猫从桌子上跳下来。你要让 AI 重新"画"出这个视频。

**第一步：写剧本（Layer 1 — Prompt 改写）**

先让一个"观察员"（VLM）看你的参考视频，写出一段描述："一只橘猫蹲在木桌上，突然起跳，四脚离桌，落地后跑开"。然后让"编辑"（LLM）把这段描述改写得更精准，变成适合指导 AI 画画的话术。

这就是 caption，也就是给画师的"剧本"。

**第二步：配一张有方向感的草稿纸（Layer 2 — Noise Prior）**

AI 画画时需要一张"初始画布"——本质上是一团随机噪声。如果完全随机，画师虽然按照剧本画，但画面构图、运动方式全凭运气。

所以我们反过来操作：从参考视频"倒推"出一张特殊的噪声图（Flow Inversion），这张噪声图里暗含了原视频的运动结构。再用 SVD 把里面的"外观信息"去掉、只留"运动方向"，最后把它和随机噪声按约 6:94 的比例混合（α=0.004，即 √α≈0.063 的运动信号权重）。

这就像给画师一张"带运动暗示的草稿纸"——他不会照着画，但起笔时会不自觉往正确方向走。

**第三步：给画师一个"参考运动轨迹"（Layer 3 — Trajectory Anchor）**

剧本写好了，草稿纸配好了，但画师画到一半可能走偏——运动方向不对、节奏不准。

我们的做法是：提前知道"理想的画画过程"长什么样（通过 inversion 记录参考视频从清晰→噪声的反向轨迹），然后在画师画画时，每画一步都轻轻"拽一下"，让他不要偏离太远。

具体来说，画师画到第 k 步时，我们看一眼"理想轨迹在同一进度下应该在什么位置"（z_ref_t），然后把画师当前位置（z_gen）往那个方向轻轻推一下：`z_new = (1-β)*z_gen + β*z_ref_t`。β 从小到大再到零（warmup → decay），前期让画师自由发挥，中期施加引导，后期松手让他画细节。

**第四步：正式画画（生成）**

万事俱备：
- 草稿纸 = 混合后的 latents（带运动方向暗示，α=0.004）
- 条件 = caption embedding（原始 text embedding）
- 画师 = Wan2.1 DiT 模型（完全冻结，一个参数都没改）
- 参考轨迹 = inversion 过程记录的每步中间状态

启动 30 步 ODE 积分，画师按照条件从草稿纸出发画画，每步画完后我们轻轻拽他一下往参考轨迹靠。

**最终效果**：三层信息叠加——文字告诉画师"画什么"，噪声先验暗示"从哪开始"，轨迹锚定引导"怎么动"。每一层都是上一层的补充，不重复、不冲突。

---

### 关键设计哲学

我们的方法的名字来源于它的核心理念：**Pipeline as Flags**。所有改动点都是可选的、正交的、可以任意组合的。你可以只用文字（baseline），也可以全部叠加（`--full`）。这使得消融实验变成了简单的 flag 排列组合，无需维护多个代码分支。

---

## 第二部分：技术架构

### 2.1 架构概览

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Overview                              │
├──────────────────────────────────────┬─────────────────────────────────┤
│  Step 1: Load Reference Video        │  → ref_video tensor             │
│  Step 2: Caption (VLM if empty)      │  → prompt text                  │
│  Step 3: Noise Prior                 │  → (η_temporal, η_inv_raw)      │
│    ├─ Flow Matching Inversion        │     η_inv = ODE_reverse(z₀)     │
│    ├─ SVD Filtering                  │     η_temporal = SVD(η_inv)     │
│    └─ (returns raw η_inv for Step 3.5) │                               │
│  Step 3.6: Trajectory Cache           │  → ref_trajectory               │
│    └─ Inversion with cache           │     缓存每步中间态               │
│  Step 4: Generate Loop               │  → video frames                 │
│    ├─ Noise Blending (if --blend)    │     η = √α·η_t + √(1-α)·η_r   │
│    ├─ Trajectory Anchor (if --traj)  │     callback: lerp to ref       │
│    ├─ CFG Generation                 │     Wan2.1 diffusers pipeline   │
│    └─ VLM Iteration (if --iter)      │     compare → refine prompt     │
│  Step 5: Output                      │  → final.mp4 + metadata.json   │
└──────────────────────────────────────┴─────────────────────────────────┘
```

### 2.2 Flag 体系

| Flag | 改动点 | 效果 | 计算开销 |
|------|--------|------|----------|
| `--inversion` | Flow Matching Inversion | 从参考视频反演噪声 η_inv | +50 DiT forwards |
| `--svd` | SVD Two-stage Filtering | 空间去内容 + 时间保运动 | 几乎无开销 |
| `--blend` | Noise Prior Blending | η = √α·η_temporal + √(1-α)·η_random | 无开销 |
| `--trajectory_anchor` | Trajectory Anchor | callback 每步 lerp to ref traj | +50 DiT forwards (缓存轨迹) |
| `--iter N` | Iterative VLM Optimization | N轮VLM反馈循环 | N×(生成+VLM推理) |
| `--midpoint` | Midpoint ODE Solver | 二阶精度反演 | +50 DiT forwards |
| `--composite` | Vertical Composite | 三面板拼接送VLM | 无开销 |

### 2.3 计算开销分析

```
Configuration                      DiT Equivalent Forwards    相对 Baseline
────────────────────────────────────────────────────────────────────────────
Baseline (caption only)            ~30                        1.0×
+inversion                         ~80  (50 inv + 30 gen)    2.7×
+inversion +svd +blend             ~80                        2.7×
+inversion +trajectory_anchor      ~130 (50 inv + 50 cache + 30 gen) 4.3×
+all +iter10                       ~430 (130 + 10×30 gen)    14.3×
```

### 2.4 核心数学

**Flow Matching Inversion**: 沿 ODE 反向积分，将视频 latent z₀ 映射回噪声空间

```
η_inv = z₀ + ∫₁⁰ v_θ(x_t, t, e₀) dt    (Euler / Midpoint discretization)
```

**SVD Filtering**: 两阶段频谱分离

```
η_inv ∈ R^(C×F×H×W)

Stage 1 (Spatial): reshape → (C·F, H·W), SVD, 截断前 ρ_s 比例奇异值 → η_motion
Stage 2 (Temporal): reshape → (C·H·W, F), SVD, 保留前 ρ_m 比例奇异值 → η_temporal

效果: 空间去内容（外观信息在大奇异值中）+ 时间保运动（运动信息在大奇异值中）
```

**Trajectory Anchor**: 每步去噪后将 latent 向参考轨迹做 soft lerp

```
ref_trajectory = {t: z_t | t ∈ [0,1]}      (inversion 过程缓存的每步中间态)
z_anchored = (1-β_t) * z_gen + β_t * z_ref_t   (lerp 锚定)
β_t: warmup_decay 调度 (前20% 从 0.5β_max 升到 β_max，后80% cosine decay 到 0)

等价于在 Flow ODE 上加弹性恢复力:
  dz/dt = v_θ(z_t, t, c) + β_t * (z_ref_t - z_t)
```

**Noise Blending**: 运动噪声与随机噪声的凸组合

```
η = √α · η_temporal + √(1-α) · η_random,  α = 0.004
```

---

## 第三部分：设计演进与简化

### 3.1 为什么采用 Trajectory Anchor？

早期实验中，我们尝试过 Velocity Matching（30 步 Adam 优化 Δe）作为 Layer 3。该方案虽然在部分高运动样本上有效，但整体存在以下问题：

1. **耗时过长**：每样本额外 90 DiT forwards（30步×forward+backward），生成时间从 80s 增到 170s
2. **效果边际**：在 L1+L2 基础上仅额外贡献 CLIP +0.3%，XCLIP +0.5%
3. **两极分化**：对低运动样本存在退化风险（#21 -1.7%，#47 -1.8%）
4. **与 L2 融合困难**：SVD-blended z_T 和 η_inv 定义的目标轨迹起点不对齐

因此我们转向 Trajectory Anchor——利用 diffusers 原生 callback 机制，在每步去噪后将 latent 轻轻拉向参考轨迹。优势在于：零额外 forward 开销（只需 lerp 操作）、与 L2 自然兼容、实现简洁。

### 3.2 Trajectory Anchor 的设计选择

**核心机制**: callback_on_step_end 中做 position lerp
- `z_anchored = (1-β_t)*z_gen + β_t*z_ref_t`
- β 调度: warmup_decay（前 20% warmup，后 80% cosine decay）
- 质量门控: η_temporal 帧间 cos 检测运动一致性，混乱样本自动跳过

**当前探索方向**:
- cos-proportional β: `effective_β = β_t × max(0, cos(gen, ref))`，解决 L2+L3 起点不对齐问题
- Velocity Direction Anchor: 不做位置 lerp，改做速度方向参考（学术潜力最大）

---

## 第四部分：版本演进历史

### v1.0 — Baseline (caption → 一次生成)
- 最简形态: VLM 描述参考视频 → T2V 生成
- 建立了统一 pipeline 架构和 flag 体系

### v2.0 — Noise Prior (+inversion +svd +blend)
- 引入 Flow Matching Inversion: 从参考视频反演初始噪声
- SVD 两阶段滤波: 从 η_inv 中分离运动成分
- Noise Blending: α=0.001 混合，提供方向性偏置

### v3.0 — Iterative Optimization (+iter)
- VLM 迭代反馈: 生成→对比→优化 prompt → 重新生成
- Composite 三面板拼接: 让 VLM 同时看到 ref/prev/current

### v4.0 — LLM Rewriting (外部预处理)
- LLM 话术改写: 将 VLM 的「描述性 caption」改写为「指导性 prompt」
- Hybrid 策略: VLM 描述 → LLM 改写 → 位置优化

### v5.0 — Midpoint Solver (+midpoint)
- 二阶中点法 ODE 求解器: 替代一阶 Euler，提升反演精度

### v6.0 — Velocity Matching (+velocity) [已废弃]
- 30 步 Adam 优化 embedding 残差 Δe，通过 text_encoder hook 注入
- 效果边际（CLIP +0.3%，XCLIP +0.5%），耗时 +90 DiT forwards
- 废弃原因：开销大、收益小、与 SVD Noise Prior 起点不对齐

### v7.0 — Position-Aware + RF-Solver [已废弃]
- Position-Aware Gradient Scaling 假设错误（DiT 中 pos 0 非 attention sink）
- RF-Solver 反演/生成 solver mismatch，指标全面退化

### v8.0 — Trajectory Anchor (+anchor) ← 当前版本

彻底替换 Velocity Matching 方案，采用去噪 callback 中的位置 lerp：

- `z_anchored = (1-β_t)*z_gen + β_t*z_ref_t`
- warmup_decay β 调度 + η_temporal 质量门控
- 零额外 forward 开销，与 L2 SVD Noise Prior 自然兼容
- 当前探索：cos-proportional β、Velocity Direction Anchor

---

## 第五部分：当前状态与实验结论

### 已验证的最优组合（L1+L2，10 样本）

| 方案 | CLIP | XCLIP | 相对 Baseline |
|------|------|-------|--------------|
| Baseline（VLM caption 直出） | 0.8703 | 0.7164 | — |
| L1: V4 Hybrid Prompt iter1 | 0.8842 | 0.7430 | CLIP +1.6%, XCLIP +3.7% |
| **L1+L2: + SVD Noise Prior α=0.004** | **0.8953** | **0.7667** | **CLIP +2.9%, XCLIP +7.0%** |

### 各层独立贡献

| 层 | 技术 | CLIP 增量 | XCLIP 增量 | 状态 |
|---|------|----------|-----------|------|
| L1 | V4 Hybrid Prompt Rewrite | +1.6% | +3.7% | ✅ 已收敛 |
| L2 | SVD Noise Prior (α=0.004) | +1.3% | +3.3% | ✅ 已收敛 |
| L3 | Trajectory Anchor | — | — | 🔬 实验中 |

### 关键结论

1. **L1+L2 贡献了绝大部分增益**（CLIP +2.9%，XCLIP +7.0%），框架基础已非常稳固
2. **α=0.004 是最优混合系数**（5 点网格搜索），颠覆了此前 α=0.001 的保守假设
3. **旧 L3（Velocity Matching）已废弃**：收益边际（+0.3%/+0.5%）且耗时翻倍，不适合作为最终方案
4. **新 L3（Trajectory Anchor）在探索中**：零额外开销的位置 lerp 方案，当前验证 warmup_decay 调度 + cos-proportional β

### 下一步方向

Layer 3 Trajectory Anchor 的完善与验证——重点是在不引入额外计算开销的前提下提供稳定的运动一致性增益。

---

## 附录：文件结构

```
P-Flow/
├── run.py                      # CLI 入口 (flag 解析 → PFlowConfig → PFlowPipeline)
├── src/
│   ├── __init__.py             # 版本 & 导出
│   ├── pipeline.py             # 统一管线 (PFlowConfig + PFlowPipeline)
│   ├── trajectory_anchor.py    # Trajectory Anchor (callback position lerp)
│   ├── flow_matching.py        # FlowMatchingInverter (Euler + Midpoint)
│   ├── svd_filter.py           # SVDFilter (空间去内容 + 时间保运动)
│   ├── vlm_client.py           # VLM 客户端 (Local/DashScope/Mock)
│   ├── video_utils.py          # 视频 I/O 工具
│   └── distributed.py          # 单 GPU 推理工具
├── evaluation/
│   ├── run_clip_xclip_eval.py  # CLIP/X-CLIP 评估
│   └── run_stream_eval.py      # StreamEval 流式评估
├── scripts/
│   ├── rewrite_hybrid.py       # LLM 话术改写
│   └── ...
├── docs/
│   ├── P-Flow故事与技术演进.md  # ← 本文档
│   └── ...
└── configs/
    └── batch_experiment.yaml   # 批量实验配置
```
