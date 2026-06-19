# P-Flow: 基于三层正交分解的参考视频引导视频生成方法

> **摘要**：本文档系统性总结了 P-Flow 三层架构（L1: Prompt Rewrite, L2: SVD Noise Prior, L3: Feature Injection）的完整技术方案，涵盖从设计动机、理论基础、数学公式到工程实现的所有细节。三层操作分别作用于文本条件空间、噪声空间和特征空间，构成正交互补的参考视频信息注入体系。在 Wan2.1-T2V-1.3B 模型上的实验表明，该方法在不修改模型参数的前提下，实现了 CLIP +4.4%、XCLIP +14.5% 的一致性提升。

---

## 1. 问题定义与总体框架

### 1.1 问题陈述

给定参考视频 $V_{\text{ref}}$ 及其文本描述 $c$，目标是引导预训练 T2V 模型生成视频 $V_{\text{gen}}$，使其在外观和运动模式上尽可能接近参考视频，同时保持生成多样性和自然性。核心约束条件为：**不修改 T2V 模型参数**（zero-training），仅通过操作模型的输入接口实现引导。

### 1.2 基础生成模型

P-Flow 基于 **Wan2.1-T2V-1.3B**，其核心结构为 DiT (Diffusion Transformer) 配合 Flow Matching 采样。生成过程遵循如下 ODE：

$$
\frac{dx_t}{dt} = v_\theta(x_t, t, c), \quad t \in [0, 1]
$$

其中 $x_t$ 为 $t$ 时刻的隐状态，$v_\theta$ 为 DiT 预测的速度场，$c$ 为文本条件。Flow Matching 采用线性插值路径：

$$
x_t = (1 - t) \cdot \varepsilon + t \cdot x_1 \tag{1}
$$

其中 $\varepsilon \sim \mathcal{N}(0, I)$ 为初始噪声，$x_1$ 为数据分布样本。对应速度场为：

$$
v_\theta(x_t, t, c) \approx x_1 - \varepsilon \tag{2}
$$

生成时从 $t=0$（纯噪声）出发，通过 Euler 积分向 $t=1$（数据）演进：

$$
x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c), \quad \Delta t = \frac{1}{N_{\text{steps}}} \tag{3}
$$

### 1.3 三个操作接口

Wan2.1 Pipeline 对外暴露三个可操作接口，恰好对应三层正交空间：

| 层级 | 操作接口 | 操作空间 | 信息注入方式 |
|------|----------|----------|------------|
| **L1** | `prompt` (文本字符串) | 文本条件空间 | UMT5 编码 → cross-attention 条件 |
| **L2** | `latents` (初始噪声张量) | 噪声空间 (ODE 起点) | 替换默认随机噪声为结构化初始条件 |
| **L3** | `register_forward_hook` | 特征空间 (DiT 中间层) | 前向传播内部特征混合 |

三层的正交性体现为：L1 影响"生成什么内容"的语义方向；L2 影响"从哪里出发"的 ODE 起点；L3 影响"途中怎么走"的中间表示。三者在不同空间操作，不存在直接冲突。

### 1.4 完整 Pipeline 数据流

```
参考视频 V_ref
    │
    ├──[VAE Encode]──→ z_0 ∈ ℝ^{1×16×21×60×104}
    │
    ├──[VLM Caption]──→ 原始文本 c_raw ──[L1: Rewrite]──→ 优化文本 c
    │
    ├──[Flow Matching Inversion]──→ η_inv (反演噪声)
    │       │
    │       ├──[SVD Stage 1]──→ 去外观 → η_filtered
    │       │
    │       ├──[SVD Stage 2]──→ 保运动 → η_temporal
    │       │
    │       └──[L2: Noise Blend]──→ z_T = √α·η_temporal + √β·η_spatial + √(1-α-β)·η_random
    │
    ├──[FI Cache]──→ {h_ref^{(l,t)}} (参考特征字典)
    │
    └──[Generation with L3 Hook]──→ V_gen
            每步: h_injected = (1-λ_eff)·h_gen + λ_eff·h_ref
```

---

## 2. Layer 1: Prompt Rewrite（文本条件优化）

### 2.1 设计动机

VLM（如 InternVL2、Qwen2.5-VL）对参考视频生成的原始 caption 存在以下问题：(1) 含有大量无信息前缀（"The video depicts..."）；(2) 尾部冗余总结句占据 token 位置但不提供视觉信息；(3) 运动描述模糊或缺失。直接使用原始 caption 作为 T2V 条件，生成质量受限。

然而，过度改写（如精确添加运动描述）会与 L2 的 SVD 运动先验产生**语义冲突**——当 prompt 明确指定运动方向而 SVD 注入另一个方向时，模型被迫在两个矛盾信号间折中，导致效果反而下降。

### 2.2 理论基础：DiT Cross-Attention 的 U 型位置权重分布

通过在 Wan2.1 的 `WanAttnProcessor` 上挂载 hook，手动计算注意力权重 $\text{softmax}(QK^T/\sqrt{d})$，发现了关键的位置效应：

**实验结论**：

- 位置 0（第一个 token）的注意力权重是中间位置的 **10-15 倍**
- 最后一个 token 的权重与位置 0 几乎相等（~0.029-0.030）
- 中间所有位置几乎完全均匀（~0.001）
- 从位置 0 到位置 1，权重立即下降 96%，无渐进衰减

**根源机制**：该效应源自 UMT5 Text Encoder 的 `relative_position_bias`（`relative_attention_num_buckets=32`, `relative_attention_max_distance=128`）。DiT 的 cross-attention 直接继承了 UMT5 输出中位置 0 和最后位置的统计偏置。

**设计推论**：仅 prompt 的**首词（subject noun）**和**末词（vivid keyword）**对生成有显著影响，中间内容对生成质量的边际贡献极小。

### 2.3 V10 策略：首尾关键词替换 + VLM 事实性校正

基于上述 U 型分布发现，v10 采用最小编辑策略（edit ratio ~5-8%），确保与 L3 Feature Injection 的兼容性：

**Step 1 — LLM 首尾替换**：

- 若开头为无意义前缀（"The video depicts/shows/features..."），替换为主体名词短语
- 若结尾为笼统总结句（"The overall atmosphere/mood..."），替换为 1-3 个已在中间出现的视觉/运动关键词
- 中间内容 100% 保持原文不变
- 输出长度与输入相差不超过 ±5%

**Step 2 — VLM 事实性校正**：

- VLM 观看视频帧，与当前 caption 对比
- 仅修正事实性错误（颜色、数量、物体识别等）
- 最多 3 处单词级替换，不增不减
- 输出长度与输入完全相同

**约束设计原则**：

$$
\text{edit\_ratio} = \frac{|\text{changed\_tokens}|}{|\text{total\_tokens}|} \leq 8\% \tag{4}
$$

该约束确保 L1 改写不会导致 UMT5 编码后的 prompt embedding 产生显著偏移，从而维持与 L3 预缓存的参考特征 $h_{\text{ref}}$ 的语义对齐：

$$
\cos(E(c_{\text{rewritten}}), E(c_{\text{original}})) \geq 0.95 \tag{5}
$$

### 2.4 L1 与 L2 的结构性矛盾及解决

**核心矛盾**：SVD 噪声先验本质是"运动信息补充器"——当 prompt 不描述运动时，SVD 的时序方向偏置可以引导模型产生与参考视频相似的运动模式。一旦 prompt 已精确描述运动（如 v7e 策略添加了 dolly-in、pan-left 等镜头语），prompt 指定的运动方向与 SVD 注入的方向可能互相矛盾，实测 XCLIP 反降 4.1%。

**解决方案**：v10 策略的核心约束是**不修改中间的运动描述**，仅操作首尾位置。运动语义由 L2 (SVD) 全权负责，L1 只负责将高注意力位置的无效 token 替换为有效视觉概念词。

---

## 3. Layer 2: SVD Noise Prior（结构化噪声先验）

### 3.1 设计动机

在标准 T2V 生成中，初始噪声 $z_T \sim \mathcal{N}(0, I)$ 不携带任何参考视频信息。P-Flow 的核心思想是：通过 Flow Matching Inversion 将参考视频映射回噪声空间，经 SVD 滤波提取其中的时序运动结构，以极小比例混入初始噪声。这提供了一个确定性的运动方向偏置，被后续 30 步 ODE 积分过程累积放大。

**在 P-Flow 原论文的迭代优化框架中**，SVD 的核心作用不是直接引导运动方向，而是提供**跨迭代的稳定锚点**——使每轮生成结果的变化主要归因于 prompt 变化（可优化目标）而非噪声随机性（不可控因素），从而稳定优化器的收敛方向。

**在我们的单次生成实验中**，SVD 的作用退化为提供运动方向的微弱统计偏置，使生成视频在运动模式上与参考视频更相似。

### 3.2 Flow Matching Inversion（反演）

反演是将参考视频从数据空间映射回噪声空间的逆过程。给定 VAE 编码后的参考 latent $z_0 = \text{Enc}(V_{\text{ref}})$，反演沿 ODE 从 $t=1$ 向 $t=0$ 积分：

$$
z_{t-\Delta t} = z_t - \Delta t \cdot v_\theta(z_t, t, c), \quad \Delta t = \frac{1}{N_{\text{inv}}} \tag{6}
$$

时间步序列 $t \in \{1.0, 1-\Delta t, 1-2\Delta t, \ldots, 0\}$，最终得到反演噪声 $\eta_{\text{inv}} = z_0$（$t=0$ 处的噪声估计）。

**Midpoint 二阶方法**（可选，更高精度）：

$$
k_1 = v_\theta(z_t, t, c) \tag{7a}
$$
$$
z_{\text{mid}} = z_t + \frac{\Delta t}{2} \cdot k_1 \tag{7b}
$$
$$
k_2 = v_\theta(z_{\text{mid}}, t + \frac{\Delta t}{2}, c) \tag{7c}
$$
$$
z_{t+\Delta t} = z_t + \Delta t \cdot k_2 \tag{7d}
$$

**默认参数**：$N_{\text{inv}}=50$, $\text{guidance\_scale}=1.0$（无 CFG，避免引入 CFG 偏差）。

**输出**：$\eta_{\text{inv}} \in \mathbb{R}^{1 \times 16 \times 21 \times 60 \times 104}$，其中 16 为通道维，21 为时间维（81帧经 VAE 时间压缩4×），60×104 为空间维。

### 3.3 SVD 两阶段滤波

反演噪声 $\eta_{\text{inv}}$ 同时编码了参考视频的外观信息和运动信息。两阶段 SVD 的目标是分离并提取纯运动分量。

#### Stage 1: Spatial Decontenting（去外观/内容）

将 $\eta_{\text{inv}}$ reshape 为空间矩阵进行 SVD 分解：

$$
\eta_{\text{inv}} \in \mathbb{R}^{C \times F \times H \times W} \xrightarrow{\text{reshape}} M_s \in \mathbb{R}^{(C \cdot F) \times (H \cdot W)} \tag{8}
$$

$$
M_s = U_s \cdot \text{diag}(\sigma_s) \cdot V_s^T \tag{9}
$$

找到最小的 $k_s$ 使得去除 top-$k_s$ 奇异值后的剩余能量占比不低于 $\rho_s$：

$$
k_s = \arg\min_k \left\{ \frac{\sum_{i=k+1}^{r} \sigma_{s,i}^2}{\sum_{i=1}^{r} \sigma_{s,i}^2} \geq \rho_s \right\} \tag{10}
$$

去除空间主成分（外观信息集中在前几个奇异值中）：

$$
\eta_{\text{filtered}} = M_s - \sum_{i=1}^{k_s} \sigma_{s,i} \cdot u_{s,i} \cdot v_{s,i}^T \tag{11}
$$

同时保留外观分量（用于可选的 β 混合）：

$$
\eta_{\text{spatial}} = \sum_{i=1}^{k_s} \sigma_{s,i} \cdot u_{s,i} \cdot v_{s,i}^T \tag{12}
$$

**默认参数**：$\rho_s = 0.1$（仅去除能量最集中的约 10% 空间成分）。

#### Stage 2: Temporal Retention（保运动）

对 Stage 1 输出进行时间维度 SVD：

$$
\eta_{\text{filtered}} \in \mathbb{R}^{C \times F \times H \times W} \xrightarrow{\text{reshape}} M_m \in \mathbb{R}^{(C \cdot H \cdot W) \times F} \tag{13}
$$

$$
M_m = U_m \cdot \text{diag}(\sigma_m) \cdot V_m^T \tag{14}
$$

保留 top-$k_m$ 时间主成分（运动信息集中在前几个时间模式中）：

$$
k_m = \arg\min_k \left\{ \frac{\sum_{i=1}^{k} \sigma_{m,i}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2} \geq \rho_m \right\} \tag{15}
$$

$$
\eta_{\text{temporal}} = \sum_{i=1}^{k_m} \sigma_{m,i} \cdot u_{m,i} \cdot v_{m,i}^T \tag{16}
$$

**默认参数**：$\rho_m = 0.9$（保留 90% 能量的时间运动模式）。

**物理直觉**：Stage 1 消除了"每帧看起来像什么"的静态外观信息（对应高能量空间主成分），Stage 2 保留了"帧与帧之间如何变化"的时序运动结构（对应高能量时间主成分）。最终 $\eta_{\text{temporal}}$ 只携带运动方向偏置，不含具体外观内容。

### 3.4 Noise Blending（噪声混合）

将 SVD 提取的运动先验与随机噪声混合，构造结构化初始噪声：

**基础二路混合**（仅运动）：

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{1-\alpha} \cdot \eta_{\text{random}}, \quad \eta_{\text{random}} \sim \mathcal{N}(0, I) \tag{17}
$$

**三路混合**（运动 + 外观）：

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}} \tag{18}
$$

其中外观分量需要**量级匹配**（renorm）后再混合：

$$
\hat{\eta}_{\text{spatial}} = \eta_{\text{spatial}} \cdot \frac{\text{std}(\eta_{\text{temporal}})}{\text{std}(\eta_{\text{spatial}})} \tag{19}
$$

**参数设定**：
- $\alpha = 0.004$（运动注入系数，实际方向偏移约 2%）
- $\beta = 0.001$（外观注入系数，默认可选开启）

**为何 α 必须极小**：实验发现 T2V 模型对 $z_T$ 中的结构化偏离极度敏感。$\alpha = 0.004$ 时 $\sqrt{\alpha} \approx 0.063$，而 $\eta_{\text{temporal}}$ 的 std 约 0.28-0.41，实际注入能量仅为 $0.063 \times 0.35 \approx 0.022$，远小于随机噪声的 unit std。10+ 组消融实验证明，α 越大效果越差——信号弱到"不产生破坏"是其有效的前提条件。

### 3.5 TSR: 自适应 α（Temporal Signal Reliability）

不同视频的最优 α 差异巨大：物体运动型视频（动物奔跑）对 α 不敏感；场景型视频（室内镜头推移）对 α 极度敏感（α 从 0.001 增到 0.002，XCLIP 暴跌 0.075）。TSR 机制实现 per-sample 自适应 α 调节。

**TSR 由两个分量组成**：

1. **TCR (Temporal Concentration Ratio)**：第一奇异值的能量集中度

$$
\text{TCR} = \frac{\sigma_{m,1}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2} \tag{20}
$$

2. **TAC (Temporal Autocorrelation)**：相邻帧平均余弦相似度

$$
\text{TAC} = \frac{1}{F-1} \sum_{i=1}^{F-1} \frac{\langle \eta_{\text{temporal}}^{(i)}, \eta_{\text{temporal}}^{(i+1)} \rangle}{\|\eta_{\text{temporal}}^{(i)}\| \cdot \|\eta_{\text{temporal}}^{(i+1)}\|} \tag{21}
$$

**综合 TSR 计算**：

$$
\text{TCR}_{\text{norm}} = \sigma(10 \cdot (\text{TCR} - 0.1)) \tag{22a}
$$
$$
\text{TAC}_{\text{norm}} = \max(0, \text{TAC}) \tag{22b}
$$
$$
\text{TSR} = \text{TCR}_{\text{norm}} \times \text{TAC}_{\text{norm}} \tag{22c}
$$

**自适应 α 计算**：

$$
\alpha_{\text{adaptive}} = \alpha_{\min} + \text{TSR} \cdot (\alpha_{\max} - \alpha_{\min}) \tag{23}
$$

物理含义：TSR 高（运动信号清晰、方向一致）→ α 大 → 注入运动先验多；TSR 低（运动弱或混乱）→ α 小 → 几乎不注入，避免破坏。

### 3.6 量级匹配的必要性（β 注入的教训）

直接注入 $\eta_{\text{spatial}}$ 会导致全线崩溃（CLIP -8.3%, XCLIP -15.2%），因为：

$$
\text{std}(\eta_{\text{spatial}}) \approx 0.9 \sim 1.2, \quad \text{std}(\eta_{\text{temporal}}) \approx 0.28 \sim 0.41
$$

未 renorm 时：

$$
\sqrt{\alpha} \cdot \text{std}(\eta_{\text{temporal}}) = 0.063 \times 0.35 = 0.022 \quad (\text{运动})
$$
$$
\sqrt{\beta} \cdot \text{std}(\eta_{\text{spatial}}) = 0.032 \times 1.0 = 0.032 \quad (\text{外观}) \leftarrow \text{反而更大!}
$$

Renorm 后外观注入量正确降到运动注入量的 $\beta/\alpha$ 倍。

---

## 4. Layer 3: Feature Injection（特征空间注入）

### 4.1 设计动机与理论基础

L2 的操作仅影响 ODE 的**起点** $z_T$，信息在 30 步积分过程中逐渐被模型的生成prior冲淡。实验证明纯黑盒下 α=0.004 是绝对天花板——进一步增大 α 只会破坏生成质量而非增强引导。

Feature Injection 绕过了"只能改起点"的限制，通过 PyTorch `register_forward_hook` 在 DiT 前向传播的**内部**修改中间层特征。这等价于在 ODE 积分路径的每一步施加来自参考视频的语义约束，类比于 zero-training ControlNet 的工作方式，但无需任何额外网络或训练。

**与 L2 的正交性**：L2 影响的是噪声空间的位置（"从哪里出发"），L3 影响的是特征空间的表示（"途中怎么走"）。FI 不依赖 L2 的 $z_T$ 起点对齐，天然可叠加。实测 L2+L3 联合使用时 XCLIP +13.6%，远超单独 L2 (+5.1%) 或单独 L3 (+7.0%)。

### 4.2 参考特征缓存

在 Flow Matching Inversion 过程中，通过 hook 同步缓存 DiT 指定层的中间激活：

$$
h_{\text{ref}}^{(l, t)} = \text{DiT\_Layer}_l(z_t, t, c), \quad l \in \mathcal{L}_{\text{target}}, \; t \in \{t_0, t_1, \ldots, t_{N-1}\} \tag{24}
$$

其中 $\mathcal{L}_{\text{target}}$ 为目标注入层集合（默认 mid=层10~19，共10层），每层每步缓存一个特征张量。

**缓存模式**：注入 cross-attention 输出特征（`fi_cache_mode=attention`），即 DiT 中 cross-attention 层产出的特征激活。

### 4.3 核心注入公式

在生成过程的每步去噪中，对目标层的前向传播输出进行混合：

$$
h_{\text{injected}}^{(l,t)} = (1 - \lambda_{\text{eff}}) \cdot h_{\text{gen}}^{(l,t)} + \lambda_{\text{eff}} \cdot h_{\text{ref}}^{(l,t)} \tag{25}
$$

其中 $h_{\text{gen}}^{(l,t)}$ 为当前生成的中间层激活，$h_{\text{ref}}^{(l,t)}$ 为预缓存的参考特征，$\lambda_{\text{eff}}$ 为经门控调制后的实际注入强度。

### 4.4 λ 时间调度策略

注入强度随去噪步 $t$ 变化，采用 middle_peak 调度：

$$
\lambda(t) = \lambda_{\max} \cdot \sin\left(\frac{\pi \cdot t}{N-1}\right) \tag{26}
$$

**物理直觉**：去噪初期（$t \approx 0$）和末期（$t \approx N$），模型分别处理全局结构和高频细节，两端应给予模型更多自由度；中间步骤决定语义内容和运动模式，此时注入参考特征效果最大。

其他可选调度：

- constant: $\lambda(t) = \lambda_{\max}$
- warmup_decay: 前 20% 线性升温，后 80% 余弦衰减
- cosine_decay: $\lambda(t) = \lambda_{\max} \cdot \cos(\frac{\pi}{2} \cdot \frac{t}{N-1})$

### 4.5 三重门控机制

L3 Feature Injection 的实际注入强度 $\lambda_{\text{eff}}$ 由三重门控联合决定，在**样本级**、**步级**和**层级**三个粒度上自适应调节：

$$
\lambda_{\text{eff}}^{(l,t)} = \lambda(t) \cdot \text{QS}_{\text{eff}} \cdot g(h_{\text{gen}}^{(l,t)}, h_{\text{ref}}^{(l,t)}) \tag{29}
$$

其中：
- $\lambda(t) = \lambda_{\max} \cdot \sin(\pi t / (N-1))$：middle-peak 时间调度（§4.4）
- $\text{QS}_{\text{eff}}$：经 M_d 修正后的质量门控分数（**样本级**，§4.6）
- $g(h_c, h_r)$：自适应特征对齐门控（**步×层级**，§4.7）

**三重门控的协作逻辑**：
1. $\text{QS}_{\text{eff}}$ 在生成开始前一次性确定该样本的 FI 整体强度——运动可靠的物体运动样本获得完整注入，场景类样本减半
2. $g(h_c, h_r)$ 在每步每层动态判断——特征已对齐时自动降低注入，偏离时增强
3. $\lambda(t)$ 控制注入的时间节奏——两端自由、中间集中

#### 门控决策流程

```
FI 启动
  │
  ├── 计算 QS_signal（帧间余弦相似度 → 0.1~1.0）
  │
  ├── M_d 修正: QS_eff = QS_signal × max(M_d, fi_qs_md_floor)
  │   ├── M_d=1.0（物体运动）→ QS_eff = QS_signal × 1.0（完全不衰减）
  │   └── M_d<1（场景/环境）→ QS_eff = QS_signal × max(M_d, 0.5)（最多减半）
  │
  ├── if QS_eff < 1e-6 → 跳过 FI，走标准生成（完全跳过注入）
  │
  └── 每步每层:
      ├── Adaptive Gate: cos(h_gen, h_ref)
      │   ├── cos > 0.5 → gate 小 → 少注入（特征已对齐，无需引导）
      │   └── cos < 0.5 → gate 大 → 多注入（特征偏离，需要纠正）
      │
      └── λ_final = λ_schedule[t] × QS_eff × gate
```

### 4.6 Quality Scale × M_d（场景感知质量门控）

并非所有样本都适合特征注入——当参考视频本身运动混乱（如纯噪声）时，注入其特征只会引入干扰。Quality Scale 基于 $\eta_{\text{temporal}}$ 的帧间一致性评估注入可靠性：

$$
\text{mean\_cos} = \frac{1}{F-1} \sum_{i=1}^{F-1} \cos(\eta_{\text{temporal}}^{(i)}, \eta_{\text{temporal}}^{(i+1)}) \tag{30}
$$

$$
\text{QS}_{\text{signal}} = 0.1 + 0.9 \cdot \sigma(20 \cdot (\text{mean\_cos} - \theta)) \tag{31}
$$

其中 $\theta = 0.05$ 为阈值。当 $\text{mean\_cos} \gg \theta$（运动一致性高）时 $\text{QS}_{\text{signal}} \approx 1.0$；当 $\text{mean\_cos} < \theta$（帧间无连贯运动）时 $\text{QS}_{\text{signal}} \approx 0.1$，大幅压制注入。

**M_d 场景感知修正**：$\text{QS}_{\text{signal}}$ 仅度量 SVD 时间信号的统计可靠性，无法区分"物体运动"和"环境动态"（如蒲公英飘动、水蒸汽上升等环境动态也会导致高 TSR 和高 QS）。引入运动明确度 $M_d$（§3.3）对 QS 做语义修正：

$$
\text{QS}_{\text{eff}} = \text{QS}_{\text{signal}} \times \max(M_d, \text{fi\_qs\_md\_floor}) \tag{31a}
$$

其中 $\text{fi\_qs\_md\_floor} = 0.5$ 为保底参数，确保 $M_d = 0$ 时 FI 不会被完全关闭（保留至少 50% 的 FI 强度）。

| $M_d$ | 场景 | $\max(M_d, 0.5)$ | QS 衰减 | 物理含义 |
|:-----:|------|:----------------:|:-------:|----------|
| 1.0 | 物体运动（人/动物/车辆） | 1.0 | 无衰减 | FI 全力注入 |
| 0.3 | 环境/相机动态 | 0.5 | 减半 | FI 谨慎注入 |
| 0.0 | 静态场景 | 0.5 | 减半 | FI 保底注入 |

**跳过机制**：若 $\text{QS}_{\text{eff}} < 10^{-6}$（质量极差），直接跳过 FI，退化为标准生成路径。

### 4.7 自适应特征对齐门控

为防止在生成特征已与参考特征高度对齐时过度注入（引入不必要的约束），设计余弦相似度门控：

$$
\text{sim} = \frac{\langle h_{\text{gen}}, h_{\text{ref}} \rangle}{\|h_{\text{gen}}\| \cdot \|h_{\text{ref}}\|} \tag{27}
$$

$$
g(h_c, h_r) = 1 - \sigma(\tau \cdot (\text{sim} - 0.5)) \tag{28}
$$

其中 $\tau = 5.0$ 为温度参数。

**门控逻辑**：

- 当 $\text{sim} > 0.5$（特征已对齐）→ $g \approx 0$ → 不注入（无需引导）
- 当 $\text{sim} < 0.5$（特征偏离）→ $g \approx 1$ → 强注入（需要纠正方向）
- 当 $\text{sim} = 0.5$（中等对齐）→ $g = 0.5$ → 半量注入

**物理直觉**：该门控实现了"需要时注入，不需要时自动跳过"的智能行为。在去噪初期（特征混沌）和末期（特征趋同），门控自动降低注入；在中间阶段（语义形成期），门控保持较高值，确保参考信息被有效传递。

### 4.8 EMA 特征平滑

跨步参考特征可能存在跳变（因反演过程中的数值误差），通过 EMA 平滑减少注入信号的不连续性：

$$
\tilde{h}_{\text{ref}}^{(t)} = \gamma \cdot \tilde{h}_{\text{ref}}^{(t-1)} + (1 - \gamma) \cdot h_{\text{ref}}^{(t)} \tag{32}
$$

**默认参数**：$\gamma = 0.7$（EMA 衰减系数）。

### 4.9 FI 不修改 ODE 路径的性质

FI 仅修改 DiT 内部中间层的**输出表示**，不改变 ODE solver 看到的 $x_t$ 序列。从 ODE 的角度，FI 等价于在每步略微修改速度场 $v_\theta$ 的计算过程（通过改变中间特征），但不改变 ODE 的状态变量。这意味着 FI 与 L2 的 SVD 初始噪声完全独立——L2 选择 ODE 起点，L3 修改 ODE 路径上每步的速度估计方向。

---

## 5. 三层联合工作机制

### 5.1 正交互补关系

$$
V_{\text{gen}} = \text{ODE}_{t=0}^{t=1}\big(z_T^{(L2)}, \; v_\theta(\cdot, \cdot, c^{(L1)}) + \Delta v^{(L3)}\big) \tag{33}
$$

- $c^{(L1)}$：经 v10 首尾关键词替换的优化 prompt
- $z_T^{(L2)}$：SVD 结构化初始噪声
- $\Delta v^{(L3)}$：FI 引起的速度场隐式修正

三层信息在不同维度提供引导：

| 维度 | L1 贡献 | L2 贡献 | L3 贡献 |
|------|---------|---------|---------|
| **语义内容** | ✅ 主要 | ❌ 无 | ✅ 辅助 |
| **运动方向** | ❌ 不碰 | ✅ 主要 | ✅ 辅助 |
| **外观细节** | ✅ 辅助 | ⚠️ 可选(β) | ✅ 主要 |
| **时序一致性** | ❌ 无 | ✅ 主要 | ✅ 主要 |

### 5.2 冲突避免设计

L1 与 L2 的矛盾通过 v10 的"不碰运动描述"原则解决：L1 仅操作首尾的高注意力位置词汇（利用 U 型分布），不改动中间的运动描述，避免与 L2 的运动方向先验产生语义拉扯。

L2 与 L3 的兼容通过"空间正交"实现：L2 操作 ODE 起点（噪声空间），L3 操作 ODE 路径中间（特征空间），两者在不同抽象层级工作，无直接干扰。实测叠加效果（+13.6%）远超各自独立贡献之和（+5.1% + +7.0% = +12.1%），呈现微弱的正向协同。

L1 与 L3 的兼容性通过 v10 的极低编辑率（≤8%）保证：FI 在反演时以 $c_{\text{original}}$ 缓存特征，生成时以 $c_{\text{rewritten}}$ 为条件。若改写幅度过大（如 v9 的 ~18%），$h_{\text{gen}}$ 的分布与 $h_{\text{ref}}$ 偏移过大，cos_sim 下降导致 gate 全开、注入过强，反而有害。v10 的 ≤8% 编辑率确保 prompt embedding 偏移极小，FI 特征对齐度维持。

### 5.3 逐步数学描述

**第 $t$ 步生成的完整计算**（$t = 0, 1, \ldots, N-1$）：

1. DiT 前向传播：
$$
h_{\text{gen}}^{(l,t)} = f_l(x_t, t, c^{(L1)}), \quad l \in \mathcal{L}_{\text{target}}
$$

2. FI 注入：
$$
\text{sim} = \cos(h_{\text{gen}}^{(l,t)}, \tilde{h}_{\text{ref}}^{(l,t)})
$$
$$
g = 1 - \sigma(\tau \cdot (\text{sim} - 0.5))
$$
$$
\lambda_{\text{eff}} = \lambda_{\max} \cdot \sin\left(\frac{\pi t}{N-1}\right) \cdot g \cdot \text{QS}_{\text{eff}}
$$
$$
h_{\text{out}}^{(l,t)} = (1 - \lambda_{\text{eff}}) \cdot h_{\text{gen}}^{(l,t)} + \lambda_{\text{eff}} \cdot \tilde{h}_{\text{ref}}^{(l,t)}
$$

3. ODE 步进（使用修改后的特征完成剩余层计算，得到速度场）：
$$
x_{t+1} = x_t + \Delta t \cdot v_\theta^{(L3)}(x_t, t, c^{(L1)})
$$

4. 初始条件来自 L2：
$$
x_0 = z_T^{(L2)} = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}}
$$

---

## 6. 实验结果汇总

### 6.1 核心指标定义

- **CLIP (orig_gen_clip)**：逐帧计算生成视频与参考视频的 CLIP 图像 embedding 余弦相似度，取帧均值。衡量静态外观保真度。
- **XCLIP (orig_gen_xclip)**：将整段视频输入 X-CLIP 模型计算时序语义 embedding 相似度。衡量运动模式和时序语义一致性。

### 6.2 主要消融结果

| 配置 | CLIP | XCLIP | vs Baseline |
|------|------|-------|-------------|
| Baseline (纯 VLM caption, 随机噪声) | 0.8753 | 0.7491 | — |
| L2 only (SVD v1, α=0.004) | 0.8964 | 0.7874 | +2.4% / +5.1% |
| L1+L2 (v10 改写 + SVD) | 0.8947 | 0.7973 | +2.2% / +6.4% |
| L2+L3 (SVD + FI 三重门控) | **0.9042** | **0.8138** | **+3.3% / +8.6%** |
| L1+L2+L3 (v10 + SVD + FI + β) | **0.9086** | **0.8221** | **+3.8% / +9.7%** |

### 6.3 场景类型适应性

| 场景类型 | 样本数 | SVD 增益 | FI 增量 | 推荐 α | 推荐 λ |
|----------|--------|----------|---------|--------|--------|
| 物体运动（动物/人类） | 10 | +0.0466 | +0.0117 | 0.003-0.005 | 0.05 |
| 场景/镜头运动 | 16 | +0.0083 | +0.0100 | 0.0005-0.001 | 0.02-0.03 |
| 异常活动 | 4 | +0.0370 | -0.0253 | 0.001-0.002 | 0.03-0.04 |

---

## 7. 关键设计决策与 Ablation 发现

### 7.1 SVD 的核心限制

> **α=0.004 是纯黑盒的绝对天花板。**

10+ 组消融实验（频域重塑、通道集中注入、多尺度分解、相位插值、SGA 自适应等）全部失败。$\eta_{\text{temporal}}$ 的一切信息（内容、幅度、相位）在强注入时对 T2V 模型均有害。原因在于 Flow Matching 模型对 $z_T$ 分布的假设极其刚性——任何偏离 $\mathcal{N}(0, I)$ 的结构化扰动，哪怕很小，都会使 ODE 积分路径偏离训练时的数据流形。

### 7.2 Renorm 的有害性

对 $\eta_{\text{temporal}}$ 做 renorm（将 std 归一化到 1.0）会抹杀其"隐性自适应"——不同样本的 $\eta_{\text{temporal}}$ 天然有不同的 std（0.28-0.41），std 小的样本运动信号弱（合理地注入少），renorm 强行拉平后过度注入这些低信号样本。实测 renorm 版本 XCLIP -4.7%。

### 7.3 负面 Prompt 的有害性

Wan2.1-1.3B 的 UMT5 encoder 对负面 prompt 极度敏感。实验发现使用 `negative_prompt` 会导致 XCLIP 下降约 5.9%——负面 prompt 在 CFG 减法操作中干扰了正向语义编码。P-Flow 使用硬编码的极简 negative prompt 或完全不使用。

### 7.4 L1 改写中"不碰运动"的必要性

v7e（精确运动描述 + SVD）的 XCLIP -4.1% vs baseline 是核心反例。当 prompt 说"dolly-in"而 SVD 注入的运动方向是"pan-left"时，模型在两个矛盾信号间产生混乱的折中运动，XCLIP 严重下降。

### 7.5 FI 与 SVD 的协同机理

L2 提供运动方向的宏观偏置（影响 ODE 起点），L3 提供语义保真的微观约束（影响每步的特征表示）。两者互补的关键在于：

- SVD 仅能传递**时序结构**信息（哪些帧之间有关联），不能传递**语义内容**（具体是什么在动）
- FI 传递完整的**语义特征**（外观+运动语义），但不改变 ODE 路径的宏观几何
- 联合使用时：SVD 确保 ODE 起点在正确的运动流形附近，FI 确保路径上每步都朝参考视频的语义方向演进

---

## 8. 超参数汇总

| 参数 | 默认值 | 含义 | 敏感度 |
|------|--------|------|--------|
| `num_inference_steps` | 30 | Flow Matching 去噪步数 | 低 |
| `guidance_scale` | 5.0 | CFG 强度 | 中 |
| `α` (alpha) | 由门控决定 | SVD temporal 混合系数（由 M_d×TSR 融合门控自适应计算） | 🔴 极高 |
| `alpha_floor` | 0.004 | α 保底值（M_d=1.0 时 α 最低值 = 旧版固定值） | 🔴 极高 |
| `alpha_max` | 0.006 | α 融合上限（高 TSR+M_d=1.0 时 α_fusion 上限） | 高 |
| `alpha_md_floor` | 0.3 | α 保底中 M_d 的下限，防止 M_d=0 时 α 完全归零 | 高 |
| `β` (beta) | 0.001 | SVD spatial 混合系数 | 中 |
| `ρ_s` | 0.1 | Stage 1 去外观能量阈值 | 低 |
| `ρ_m` | 0.9 | Stage 2 保运动能量阈值 | 低 |
| `λ_max` (fi_lambda) | 0.05 | FI 最大注入强度 | 中 |
| `fi_layers` | mid (10-19) | FI 目标层 | 中 |
| `fi_schedule` | middle_peak | FI 时间调度 | 低 |
| `fi_adaptive_temp` (τ) | 5.0 | 自适应门控温度 | 低 |
| `fi_ema_decay` (γ) | 0.7 | EMA 平滑系数 | 低 |
| `QS_threshold` (θ) | 0.05 | Quality Scale 阈值 | 中 |
| `fi_qs_md_floor` | 0.5 | FI QS 中 M_d 的下限，防止 M_d=0 时 FI 完全关闭 | 中 |
| `N_inv` | 50 | 反演步数 | 低 |

---

## 9. 总结与展望

### 9.1 方法贡献

P-Flow 三层架构在不修改模型参数的前提下，通过操作三个正交空间（文本、噪声、特征）将参考视频信息注入预训练 T2V 模型。核心贡献包括：

1. **U 型位置权重发现**驱动的最小编辑 prompt 策略（v10），仅改首尾高权重位置
2. **两阶段 SVD 滤波**分离运动与外观，配合极小 α 实现安全的运动先验注入
3. **三重门控 Feature Injection**：样本级 QS×M_d 场景感知门控 + 步×层级自适应特征对齐门控 + 时间调度，实现"需要时注入、不需要时自动跳过"的智能行为
4. **三层正交设计**消除层间冲突，实现超越单层贡献之和的协同效果

### 9.2 当前局限

1. 场景型视频（镜头运动、FPV 穿越）的 L2 α 增益仍不稳定，M_d 门控将其降至保底值但可能丢失空间信息正增益
2. L3 FI 的 M_d 门控对 M_d<1 样本一律减半注入，粒度仍偏粗，未来可考虑 M_d 连续调制
3. FI 的 Quality Scale 依赖 $\eta_{\text{temporal}}$ 帧间余弦，该指标与实际注入效果的相关性有限

### 9.3 未来方向

1. **SSR → β 门控**：72 号（静态场景）的旧版正增益来自 SVD 空间信息而非时间信号，可通过实现 SSR 门控让 β（空间分量）在场景类样本上自动启用
2. **M_d 连续调制**：当前 L3 FI 对 M_d<1 样本一律使用 $\max(M_d, 0.5)$ 保底减半，未来可探索 M_d 的连续调制曲线（如 sigmoid 软过渡）
3. **分层 FI**：对不同层注入不同类型的信息（early 层注入外观细节，mid 层注入语义结构，late 层注入高频纹理）
4. **迭代优化的引入**：结合 VLM 反馈实现 prompt 迭代优化，充分发挥 SVD 作为"跨迭代稳定锚点"的原始设计意图

---

*文档版本: v2.0 | 更新日期: 2026-06-20 | 基于 P-Flow 场景感知门控 v3.5 实验数据*
