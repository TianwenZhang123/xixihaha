# Prompt Inversion 四条主流路线与 VMAD 定位

> 本文档梳理"从视频/图像逆向找到 prompt 表示"这一问题的所有主流技术路线，并阐明 VMAD 在技术版图中的位置。

---

## 问题定义

给定一个已训练好的生成模型 G 和一段目标视频 V_ref，目标是找到某种"输入表示" c*，使得 G(c*) ≈ V_ref。不同方法的区别在于 **c* 的形式** 和 **求解 c* 的方式**。

---

## 路线一：梯度优化 Embedding（VMAD 所在路线）

### 核心思路

冻住模型参数 θ，初始化一个可学习的 embedding（或 embedding 残差 Δe），通过某种重建损失对 Δe 做梯度下降，让生成结果逼近目标。

### 代表工作

| 工作 | 会议 | 优化对象 | 损失函数 | 模型类型 |
|------|------|---------|---------|---------|
| Textual Inversion | ICLR 2023 | pseudo-token embedding | denoising loss (ε-prediction MSE) | T2I (SD) |
| Promptus | AAAI 2025 Oral | per-frame prompt embedding | 逐帧 denoising loss | T2I (SD) → 视频 |
| MotionPrompt | CVPR 2025 | learnable motion token | motion-aware denoising loss | T2V |
| TPSO | arXiv 2025 | token + prompt dual space | dual-space reconstruction loss | T2I |
| **VMAD (ours)** | — | embedding residual Δe | **velocity field matching loss** | **T2V (Flow Matching)** |

### 各方法损失函数对比

```
Textual Inversion:  L = E_{t,ε} [ ||ε_θ(x_t, t, e₀+Δe) - ε||² ]
                    (标准 DDPM denoising loss)

Promptus:           L = (1/N) Σᵢ E_{t,ε} [ ||ε_θ(x_t^i, t, eᵢ) - εᵢ||² ]
                    (逐帧 denoising loss，每帧独立优化)

VMAD:               L = E_{t~U[0,1]} [ ||v_θ(x_t, t, e₀+Δe) - v*||² ]
                    (velocity field matching，v* = z₀ - η_inv)
                    其中 x_t = (1-t)·η_inv + t·z₀
```

### VMAD 相对于同路线方法的优势

1. **时序一致性**：Promptus 逐帧优化，无法保证帧间连贯；VMAD 在 T2V 模型的 velocity field 上直接优化，天然保持时序一致
2. **目标速度场有封闭解**：Flow Matching 的直线轨迹使 v* = z₀ - η_inv 精确成立，无需近似；DDPM 框架下需要通过采样估计目标
3. **效率**：100 步 DiT 前向即可收敛（vs Textual Inversion 通常需要 3000-5000 步）

### 优缺点

- 优点：保真度最高，理论上可无损还原；不需要额外训练 encoder
- 缺点：每个视频需要单独跑梯度循环（100-1000步），推理时间长

---

## 路线二：训练 Encoder 直接预测（前馈式）

### 核心思路

训练一个独立的 encoder 网络（通常是 CLIP/BLIP 视觉编码器 + 映射层），输入图像/视频，一次前向传播直接输出 embedding，无需梯度迭代。

### 代表工作

| 工作 | 会议 | Encoder架构 | 输出形式 |
|------|------|------------|---------|
| BLIP-Diffusion | NeurIPS 2023 | BLIP-2 Q-Former | subject embedding |
| IP-Adapter | arXiv 2023 | CLIP image encoder + linear | image embedding |
| ELITE | ICLR 2024 | CLIP + mapping network | token-level embedding |
| EDITOR | arXiv 2025 | BLIP caption + embedding-to-text | 可解释文本 prompt |
| Emu2 | CVPR 2024 | visual encoder + regressor | unified multimodal embedding |

### 做法

离线训练大量 (image/video, prompt) 配对数据，让 encoder 学会从视觉内容直接预测对应的 text/image embedding。推理时只需一次前向传播（几十毫秒）。

### 优缺点

- 优点：推理极快（单次前向）；可批量处理；适合工业部署
- 缺点：保真度有上限 —— encoder 的表达能力有限，无法逐像素还原；需要大量训练数据；换模型需重新训练

### 与 VMAD 的关系

VMAD 不采用此路线作为核心方法，因为目标是"高保真还原"而非"快速近似"。但此路线可作为 VMAD 的未来加速方向 —— 先用梯度优化获得大量 (video, Δe) 训练对，再训练 encoder 做 amortized inference。

---

## 路线三：噪声/轨迹反演（不改 prompt，改初始噪声）

### 核心思路

不修改 prompt embedding，而是寻找一个"正确的初始噪声" η*，使模型从 η* 出发、用原始 prompt 就能精确还原目标。做法是将生成过程的 ODE/SDE 反向积分（从 x₀ 到 x_T）。

### 代表工作

| 工作 | 会议 | 模型框架 | 反演方法 |
|------|------|---------|---------|
| DDIM Inversion | — | DDPM/DDIM | 确定性逆采样 |
| Null-Text Inversion | CVPR 2023 | SD (CFG) | null-text 优化 + DDIM inv |
| RF-Inversion | ICLR 2025 | Rectified Flow (FLUX) | 随机微分方程 + LQR 控制 |
| RF-Solver | ICML 2025 | Rectified Flow | 高阶 Taylor ODE 求解器 |
| PnP Inversion | ICLR 2024 | SD | 3行代码修正反演偏差 |

### 核心公式

```
DDIM Inversion:     x_{t+1} = √(α_{t+1}/α_t) · x_t + (√(1-α_{t+1}) - √(α_{t+1}/α_t)·√(1-α_t)) · ε_θ(x_t, t)

Flow Matching Inv:  η_inv = z₀ + ∫₀¹ v_θ(x_t, t, c) dt   (Euler 离散化 N=50步)
                    即沿着 ODE 从 z₀ "倒退" 到噪声空间
```

### 优缺点

- 优点：数学严格；理论上完美重建（如果ODE足够精确）；无需训练或优化循环
- 缺点：得到的是一个巨大噪声张量（与视频同尺寸），**不可解释、不可编辑、不可压缩、不可迁移**。本质是"记住了全部信息"但没有任何抽象

### 与 VMAD 的关系

VMAD 将此路线作为 **Phase A 的前置步骤**：
1. Flow Matching Inversion 提供 η_inv，用于构建梯度优化的目标速度场 v* = z₀ − η_inv
2. SVD 分解后的 η_motion 作为 Layer 3 的噪声先验，在生成时混入初始噪声提供空间引导

简言之：路线三为路线一提供"锚点"，但 VMAD 的核心资产是 Δe（紧凑、可迁移），不是 η_inv 本身。

---

## 路线四：VLM / Captioning 直接描述（语言级别）

### 核心思路

用视觉语言模型直接看视频，输出一段自然语言描述，然后把这段文字当作 prompt 去生成。纯推理，无任何优化。

### 代表工作

| 工作/模型 | 类型 | 特点 |
|-----------|------|------|
| Qwen2-VL / Qwen2.5-VL | VLM | 视频理解 + 详细描述 |
| InternVL-2.5 | VLM | 多模态理解 |
| GPT-4o | VLM | 通用视觉推理 |
| CogAgent | VLM | GUI/视觉理解 |
| Video Recaptioning (OpenSora) | Pipeline | 自动生成训练用 caption |

### 做法

VLM(V_ref) → text_description → T2V_model(text_description) → V_gen

### 优缺点

- 优点：最快、最可解释、最可迁移（纯文本，任何模型通用）
- 缺点：保真度最差 —— 自然语言无法精确描述时序节奏、加速度曲线、微妙的运动细节。实测 VLM caption 的天花板约 CLIP=0.88

### 与 VMAD 的关系

VLM caption 在 VMAD 中扮演两个辅助角色：
1. 提供初始描述 e₀ = TextEnc(VLM(V_ref))，作为梯度优化的起点
2. 最终资产的 Layer 1（可解释文本层），用于跨模型迁移和人工编辑

---

## 总结对比表

| 维度 | ① 梯度优化 Embedding | ② Encoder 前馈预测 | ③ 噪声/轨迹反演 | ④ VLM Captioning |
|------|---------------------|-------------------|----------------|-----------------|
| 优化对象 | text embedding (Δe) | 训练映射网络 | 初始噪声 η | 自然语言描述 |
| 需要梯度 | 是 (100步) | 推理时不需要 | 不需要 | 不需要 |
| 保真度 | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |
| 推理速度 | 慢 (分钟级) | 极快 (毫秒级) | 快 (秒级) | 极快 (秒级) |
| 资产大小 | 小 (L×D tensor) | 小 | 巨大 (=视频尺寸) | 极小 (文本) |
| 可解释性 | 中 | 低 | 无 | ★★★★★ |
| 可迁移性 | 中 (同 text encoder) | 低 (模型绑定) | 无 (模型+seed绑定) | ★★★★★ |
| 可编辑性 | 中 | 低 | 无 | ★★★★★ |

---

## VMAD 在技术版图中的位置

```
                    保真度 ↑
                    │
         ③噪声反演 ─┼──── ① 梯度优化 Embedding ← VMAD核心
                    │              │
                    │              │ (Token Decode)
                    │              ↓
         ② Encoder ─┤         ④ VLM Caption ← VMAD辅助
                    │
                    └──────────────────────→ 可迁移性/可解释性
```

VMAD 的独特贡献是 **将路线①③④有机组合为三层编码**，以路线①的梯度优化为核心引擎，路线③提供优化锚点和高频补充，路线④提供可解释性入口。而在路线①内部，VMAD 的创新在于：

1. **首次将 Velocity Field Matching 用于视频 prompt inversion**（vs Promptus 的逐帧 denoising loss）
2. **利用 Flow Matching 直线轨迹的封闭解**构建精确目标 v*（无需采样估计）
3. **Position-Aware 梯度缩放**利用 DiT attention sink 特性加速收敛
4. **三层频谱互补编码**兼顾保真度与可迁移性

---

## 整体框架图

```
┌─────────────────────────────────────────────────────────────────┐
│                    VMAD: Video Prompt Inversion                  │
│                via Velocity Field Matching                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入: 原始视频 V_ref                                            │
│    │                                                            │
│    ├── VAE Encode ──→ z₀ (视频 latent)                          │
│    ├── VLM Caption ──→ text ──→ T5 Encode ──→ e₀               │
│    └── Flow Matching Inversion (Euler 50步) ──→ η_inv           │
│              │                                                  │
│              ├── SVD频谱分解 ──→ η_motion (Layer 3 噪声先验)      │
│              │                                                  │
│    ┌─────────▼──────────────────────────────────────────┐       │
│    │  ★ 核心: Velocity Field Matching (梯度优化循环)     │       │
│    │                                                    │       │
│    │  目标: v* = z₀ − η_inv                             │       │
│    │  优化: min_Δe E_t[||v_θ(x_t,t,e₀+Δe) − v*||²]    │       │
│    │  约束: x_t = (1-t)·η_inv + t·z₀                   │       │
│    │  方法: Adam + Cosine LR + Position-Aware Scaling   │       │
│    │  步数: 100步 DiT forward                           │       │
│    │                                                    │       │
│    │  输出: Δe (紧凑 embedding 残差)                     │       │
│    └────────────────────────────────────────────────────┘       │
│              │                                                  │
│              └── Token Decode ──→ motion_text (Layer 1 文本)     │
│                                                                 │
│  输出: Motion Asset = (motion_text, Δe, η_motion)               │
│         Layer 1 (文本)  Layer 2 (核心) Layer 3 (噪声)            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  应用 (还原视频):                                                │
│    e_final = e_content + Δe                                     │
│    η_init  = √α·η_motion + √(1-α)·η_random                    │
│    V_gen   = T2V_model(e_final, η_init) ≈ V_ref                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 大白话版本

**问题**：你有一个视频，你想让 AI 生成模型重新画出一模一样的。但 AI 只认文字指令（prompt），而文字根本描述不了视频的全部细节。

**四种解法**：

1. **让 AI 反复试，试到对为止（路线①）**—— 从一个初始描述开始，每次生成后看哪里不像，通过梯度告诉"描述"往哪个方向改，改 100 次就很像了。VMAD 走的就是这条路。

2. **训练一个"识图写描述"的专家（路线②）**—— 看一眼图就直接说出最佳描述，速度极快，但准确度有限。

3. **记住 AI 画画时的起笔位置（路线③）**—— 如果你知道 AI 从哪个随机噪声开始画的，再画一遍就一模一样。但这个"起笔位置"跟视频一样大，没法压缩，也没法给别的 AI 用。

4. **让另一个 AI 看视频写作文（路线④）**—— 最直觉的办法，但作文再长也描述不了"猫的爪子在第 3 秒加速划下去"这种微妙细节。

**VMAD 的聪明之处**：把 1+3+4 组合起来。先用路线④写个初稿，再用路线③算出"理想答案长什么样"（作为参照物），最后用路线①的梯度优化慢慢逼近理想答案。最终得到一个又小又准的"prompt 资产"。
