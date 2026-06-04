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

**第三步：微调画师的理解（Layer 3 — Velocity Matching / Δe）**

剧本写好了，草稿纸配好了，但画师对"怎么动"的理解还可能有偏差——比如猫跳的弧度、落地的节奏，光靠文字说不清。

这时候我们做一件事：反复测试画师。

具体来说，我们已经知道理想答案是什么（参考视频的 latent z₀），也知道起点是什么（反演噪声 η_inv）。从起点到终点的"理想画笔轨迹"就是一条直线，所有时刻的速度都应该是 `v* = z₀ - η_inv`（终点减起点）。

然后我们在这条理想轨迹上随机挑几个时间点，问画师："如果你现在在这个位置，按照我给你的条件 `e₀+Δe`，你下一笔会往哪画？" 画师回答 `v_pred`。如果 `v_pred` 和理想速度 `v*` 有偏差，我们就微调 Δe，让画师的回答更接近正确答案。

反复 30 次后，Δe 就收敛了——它编码了"文字说不清但参考视频里确实存在的运动细节"。

**第四步：正式画画（生成）**

万事俱备：
- 草稿纸 = 混合后的 latents（带运动方向暗示，α=0.004）
- 条件 = `e₀ + 0.02×Δe`（原始 caption embedding + 微量运动修正）
- 画师 = Wan2.1 DiT 模型（完全冻结，一个参数都没改）

启动 30 步 ODE 积分，画师按照修正后的条件，从草稿纸出发，一步步画出最终视频。

**最终效果**：三层信息叠加——文字告诉画师"画什么"，噪声先验暗示"从哪开始"，Δe 修正"怎么动"。每一层都是上一层的补充，不重复、不冲突。

---

### 速度场是什么？

用一句话说：**速度场就是"模型在生成过程中，每个像素每一步往哪个方向变化"的总和。**

更具体一点：Wan2.1 的生成过程是从噪声到清晰画面的连续变换（ODE 积分）。在任意中间时刻 t，模型都有一个预测——"当前这堆像素应该往哪个方向变"。所有像素在所有时刻的"变化方向"组合起来，就构成了速度场 `v_θ(x_t, t, c)`。

速度场完全由条件 c（text embedding）控制。不同的 text embedding 会让速度场指向不同的终点——也就是生成不同的视频。Velocity Matching 做的事就是：微调条件 c（加一个 Δe），让速度场精确指向参考视频对应的终点。

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
│  Step 3.5: Velocity Matching         │  → Δe                           │
│    └─ 30-step Δe optimization        │     v_θ(x_t,t,e₀+Δe) ≈ v*     │
│  Step 4: Generate Loop               │  → video frames                 │
│    ├─ Noise Blending (if --blend)    │     η = √α·η_t + √(1-α)·η_r   │
│    ├─ Embedding Hook (if --velocity) │     e_final = e₀ + 0.02·Δe     │
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
| `--velocity` | Velocity Field Matching | 计算 Δe → embedding hook 注入 | +60 DiT forwards |
| `--position_aware` | Position-Aware Gradient Scaling | U-shape梯度缩放+位置正则化 | 无额外forward |
| `--rfsolver` | RF-Solver (2nd-order Taylor) | 2阶Taylor反演(替代Euler) | 同Euler(+0) |
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
+inversion +velocity               ~170 (50 inv + 90 vel + 30 gen)  5.7×
+inversion +svd +blend +velocity   ~170                       5.7×
+all +iter10                       ~470 (170 + 10×30 gen)    15.7×
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

**Velocity Field Matching**: 优化 Δe 使模型速度场对齐目标轨迹

```
v* = z₀ - η_inv                          (目标速度场: 从噪声直达视频的理想方向)
x_t = (1-t)·η_inv + t·z₀                 (ground-truth 轨迹上的中间状态)
L_vel = E_{t~U[0,T_m]} [ || v_θ(x_t, t, e₀+Δe) - v* ||² ]

优化: 30步 Adam + Cosine Annealing, lr=1e-3
注入: e_final = e₀ + 0.02·Δe (通过 text_encoder hook)
```

**Noise Blending**: 运动噪声与随机噪声的凸组合

```
η = √α · η_temporal + √(1-α) · η_random,  α = 0.004
```

---

## 第三部分：设计演进与简化

### 3.1 为什么采用轻量化设计？

早期实验中，完整的 velocity matching 流水线开销巨大（900+ DiT forwards），且代码复杂（Extract/Apply 分离、三层独立脚本）。我们的方法作为轻量统一框架，在保留核心优化目标的同时大幅简化了架构。

### 3.2 核心设计选择

我们的方法在 Velocity Field Matching 模块上做了以下设计取舍：

**保留的**:
- 核心优化目标 `L_vel = ||v_pred - v*||²`
- Embedding 残差 `Δe` 的优化策略（Adam + Cosine Annealing）
- Hook-based 注入方式（保留 CFG 正常运行）
- 注入参数 `embed_strength=0.02`（经理论推导确定，详见实验记录）

**去掉的**（实现简化）:
- Content Disentanglement（Δe 的内容/运动分离正则化）— 条件反演已天然产生更纯净的 η_inv，无需额外分离
- Position-Aware Gradient Scaling（动态 U-shape 权重）— 增加复杂度但实验中收益有限
- Token Decoding（将 Δe 解码为文本）— 已证明有害，直接舍弃

**改进的**:
- 优化步数: 100→30（使用条件反演 η_inv，起点质量更高，更快收敛）
- 反演方式: 无条件反演 → 条件反演（`_encode_prompt(prompt)` vs `_encode_prompt("")`），实验表明条件反演的 η_inv 质量更高
- 梯度管理: velocity matching 需要梯度 → `run()` 不再用 `@torch.no_grad()`，改为各子步骤自行管理

### 3.3 为什么 30 步就够？

早期方案需要 100 步的根本原因是：它使用**无条件反演**（`prompt=""`)，得到的 η_inv 与有条件生成的轨迹有较大偏差，需要更多步来弥补。

我们的方法使用**条件反演**（`prompt=caption`），反演出的 η_inv 已经处在"正确条件"的 ODE 轨迹上，因此 Δe 只需做小幅修正即可。这就是为什么 30 步足够——我们不是从零开始搜索，而是在一个好的起点附近做精细调优。

### 3.4 运行时对比

```
早期方案 (100步 velocity + token decode):
  100 × (forward + backward) = 300 DiT equiv
  + token decode (385 forward) = 385 DiT equiv
  总计: ~685 DiT equivalent forwards

我们的方法 --velocity (30步, 无 token decode):
  30 × (forward + backward) = 90 DiT equiv
  总计: ~90 DiT equivalent forwards

加速比: 685 / 90 ≈ 7.6× 加速，且去掉了有害的 token decode
```

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

### v6.0 — Velocity Matching (+velocity)
- Velocity Field Matching 的轻量实现
- 30 步 Δe 优化 + text_encoder hook 注入
- embed_strength: 初始 0.005，后经理论推导优化为 0.02（最优值）

### v7.0 — Position-Aware + RF-Solver (+position_aware +rfsolver) [已废弃]

尝试引入 Position-Aware Gradient Scaling 和 RF-Solver 2nd-order Taylor inversion。

**废弃原因**（2025-06-05）：
1. **Position-Aware 假设错误**：Attention 分析证明 Wan2.1 DiT 中 pos 0 仅 2.01x mean（需 >3x 才算 attention sink），实际 high-attention 在中后段 padding positions。梯度缩放方向完全错误。
2. **RF-Solver solver mismatch**：反演用 2 阶但生成用 Euler，导致 η_inv 不在生成 ODE 的有效轨迹上，指标全面退化。

两个 flag 已从代码中删除。

### v8.0 — Velocity Matching v2 (Stratified + PadMask + MotionWeight) ← 当前版本

基于 60 步实验暴露的三个问题（梯度方差大、能量分散到 padding、静态区域浪费优化），实现三项互补改进：

1. **分层多时间步采样** (`--velocity_K 4`)：将 [0, T_m] 分成 K=4 bin，每步 4 次 forward+backward 后平均梯度，覆盖密度提升 4×，方差下降 ~2×。
2. **Padding-Aware 梯度 Mask**：tokenize caption 获取有效长度，每步 backward 后将 padding 位置梯度置零，能量集中在有语义的 token 上。
3. **运动感知 Loss 加权** (`--velocity_motion_weight 1.0`)：计算 v* 帧间方差作为 per-pixel motion weight，高运动区域获得更高 loss 权重。

**实验结果**（10 样本全量验证）：

| 指标 | v1 | v2 | 增量 |
|------|------|------|------|
| CLIP | 0.8981 | 0.8998 | +0.2% |
| XCLIP | 0.7705 | 0.7736 | +0.4% |

v2 在高运动样本上效果显著（#31 XCLIP +5.4%），但对低运动样本存在轻微退化风险（#21 -1.7%, #47 -1.8%）。消融实验证明 motion_weight 不是退化主因，退化来自 stratified sampling 或 padding mask 对特定样本的负面交互。

**当前最优配置**: `--velocity_full --alpha 0.004 --embed_strength 0.02 --velocity_K 4 --velocity_motion_weight 1.0`

---

## 第五部分：当前状态与实验结论

### 已验证的最优组合（本周实验，10 样本）

| 方案 | CLIP | XCLIP | 相对 Baseline |
|------|------|-------|--------------|
| Baseline（VLM caption 直出） | 0.8703 | 0.7164 | — |
| L1: V4 Hybrid Prompt iter1 | 0.8842 | 0.7430 | CLIP +1.6%, XCLIP +3.7% |
| L1+L2: + SVD Noise Prior α=0.004 | 0.8953 | 0.7667 | CLIP +2.9%, XCLIP +7.0% |
| L1+L2+L3v1: + Velocity es=0.02 | 0.8981 | 0.7705 | CLIP +3.2%, XCLIP +7.6% |
| **L1+L2+L3v2: + Stratified/PadMask/MW** | **0.8998** | **0.7736** | **CLIP +3.4%, XCLIP +8.0%** |

### 各层独立贡献

| 层 | 技术 | CLIP 增量 | XCLIP 增量 | 状态 |
|---|------|----------|-----------|------|
| L1 | V4 Hybrid Prompt Rewrite | +1.6% | +3.7% | ✅ 已收敛 |
| L2 | SVD Noise Prior (α=0.004) | +1.3% | +3.3% | ✅ 已收敛 |
| L3v1 | Velocity Matching (es=0.02) | +0.3% | +0.5% | ✅ 稳定 |
| L3v2 | + Stratified + PadMask + MW | +0.2% | +0.4% | ⚠️ 边际，有退化风险 |

### 关键结论

1. **L1+L2 贡献了 88% 的总增益**，L3 的边际贡献较小但方向正确
2. **α=0.004 是最优混合系数**（5 点网格搜索），颠覆了此前 α=0.001 的保守假设
3. **es=0.02 是最优注入强度**（理论推导 + 网格验证），对应注入量 ||es×Δe||≈0.17
4. **L3v2 存在两极分化**：高运动样本显著受益（#31 +5.4%），低运动样本轻微退化（#21 -1.7%）
5. **60 步实验失败**：更多步数使 ||Δe|| 增大导致注入过强，问题在效率而非步数

### 下一步优化方向

详见 `docs/TODO_优化方向.md`

---

## 附录：文件结构

```
P-Flow/
├── run.py                      # CLI 入口 (flag 解析 → PFlowConfig → PFlowPipeline)
├── src/
│   ├── __init__.py             # 版本 & 导出 (v6.0.0)
│   ├── pipeline.py             # 统一管线 (PFlowConfig + PFlowPipeline)
│   ├── velocity_matching.py    # 轻量版 VelocityMatcher (30步 Δe 优化)
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
