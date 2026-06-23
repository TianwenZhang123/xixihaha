# P-Flow: 基于正交三层分解的参考视频引导视频生成

## 3. 方法

给定参考视频 $V_{\text{ref}} \in \mathbb{R}^{T \times H \times W \times 3}$ 及其文本描述 $c$，我们的目标是引导预训练文本到视频（T2V）扩散模型生成视频 $V_{\text{gen}}$，使其保持 $V_{\text{ref}}$ 的外观和运动模式，同时不修改任何模型参数（零训练）。我们将此形式化为操控 T2V 模型的三个正交输入接口：文本条件空间（§3.2）、噪声空间（§3.3）和中间特征空间（§3.4）。我们首先在 §3.1 中介绍预备知识，然后详细阐述每一层，最后在 §3.5 中给出统一公式。

---

### 3.1 预备知识

#### 3.1.1 用于视频生成的 Flow Matching

骨干模型 Wan2.1-T2V-1.3B 是一个采用 Flow Matching 目标训练的 Diffusion Transformer (DiT)。与定义离散噪声调度的 DDPM 模型不同，Flow Matching 通过噪声与数据之间的线性插值构造连续概率路径：

$$
x_t = (1 - t) \cdot \varepsilon + t \cdot x_1, \quad t \in [0, 1], \quad \varepsilon \sim \mathcal{N}(0, I)
\tag{1}
$$

其中 $x_1$ 表示来自数据分布 $p_{\text{data}}$ 的样本，$\varepsilon$ 为标准高斯噪声。模型 $v_\theta$ 被训练以预测速度场：

$$
v_\theta(x_t, t, c) \approx x_1 - \varepsilon = \frac{x_1 - x_t}{1 - t}
\tag{2}
$$

生成过程通过 Euler 积分从 $t = 0$（纯噪声）到 $t = 1$（数据）求解 ODE：

$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c), \quad \Delta t = \frac{1}{N}
\tag{3}
$$

其中 $N = 30$ 为推理步数，$c$ 表示由冻结的 UMT5 文本编码器编码的文本条件。

#### 3.1.2 DiT 架构与输入接口

Wan2.1 DiT 由 $L = 30$ 个 Transformer 块组成，每个块包含自注意力、交叉注意力（以 UMT5 嵌入为条件）和前馈层。我们识别出三个非侵入性接口，通过它们可以注入参考信息：

- **文本接口**：prompt 字符串 $c$，由 UMT5 编码为 token 嵌入 $E(c) \in \mathbb{R}^{S \times d}$，在交叉注意力中作为键/值。
- **噪声接口**：初始隐状态 $z_T \sim \mathcal{N}(0, I)$，作为 ODE 的起始点。
- **特征接口**：层 $l$ 和时间步 $t$ 处的中间激活 $h^{(l,t)}$，可通过 PyTorch 的 `register_forward_hook` 访问。

这三个接口在不同的向量空间中操作——文本嵌入空间、噪声（隐）空间和特征（激活）空间——确保了**正交**操作而无直接干扰。

#### 3.1.3 隐空间表示

参考视频 $V_{\text{ref}} \in \mathbb{R}^{1 \times 3 \times T \times H \times W}$（其中 $T = 81$，$H = 480$，$W = 832$）首先由预训练 3D-VAE 编码为紧凑的隐表示：

$$
z_1 = \mathcal{E}(V_{\text{ref}}) \in \mathbb{R}^{1 \times C \times F \times H' \times W'}
\tag{4}
$$

其中 $C = 16$ 为隐通道维度，VAE 进行 $4\times$ 时间下采样和 $8\times$ 空间下采样，得到 $F = 21$，$H' = 60$，$W' = 104$。

---

### 3.2 Layer 1: 最小化 Prompt 重写

#### 3.2.1 动机：U 型位置注意力分布

文本条件通过交叉注意力层进入 DiT。一个自然的问题是：**prompt 中哪些 token 位置对生成质量贡献最大？** 我们通过在 `WanAttnProcessor` 上安装注意力权重提取 hook 来实证回答这个问题。

**观察（U 型分布）。** 令 $\mathbf{A} = \text{softmax}(QK^\top / \sqrt{d}) \in \mathbb{R}^{N_v \times S}$ 表示交叉注意力权重矩阵，其中 $N_v$ 是视觉 token 数量，$S$ 是 prompt 序列长度。我们计算每个文本位置 $j$ 接收到的平均注意力：

$$
\bar{a}_j = \frac{1}{N_v} \sum_{i=1}^{N_v} A_{ij}
\tag{5}
$$

在所有层和时间步中，我们观察到一致的 **U 型分布**：

$$
\bar{a}_0 \approx \bar{a}_{S-1} \approx 0.029, \quad \bar{a}_j \approx 0.001 \;\; \forall j \in \{1, \ldots, S-2\}
\tag{6}
$$

即第一个和最后一个 token 位置接收到的注意力大约是内部位置的 $10$–$15$ 倍，从位置 0 到位置 1 立即下降 $96\%$。

**根本原因。** 这一现象源于 UMT5 编码器的相对位置偏置（`relative_attention_num_buckets=32`，`relative_attention_max_distance=128`），它在输出嵌入中引入了边界偏向的统计特性。DiT 的交叉注意力直接继承了这些位置偏置。

**设计启示。** 只有 prompt 的**头部**（主语名词）和**尾部**（生动关键词）对生成有显著影响；内部 token 的边际贡献很小。

#### 3.2.2 首尾关键词替换策略

基于 U 型分布发现，我们设计了一种最小编辑策略，仅修改高注意力的边界位置，同时保留所有内部内容：

**步骤 1：LLM 首尾替换。** 给定原始 VLM 生成的 caption $c_{\text{raw}}$，我们应用以下规则：

- 如果头部包含无信息前缀（例如 "The video depicts/shows/features..."），则替换为主语名词短语。
- 如果尾部包含笼统总结句（例如 "The overall atmosphere/mood..."），则替换为已在内部出现的 1-3 个视觉/运动关键词。
- 内部内容 100% 保持原文不变。

**步骤 2：VLM 事实性校正。** 视觉语言模型（Qwen2.5-VL-7B）查看视频帧并通过单词替换最多修正 3 个事实性错误（错误的颜色、物体数量、识别错误的物体）。

组合编辑比率限制为：

$$
r_{\text{edit}} = \frac{|\text{changed tokens}|}{|\text{total tokens}|} \leq 8\%
\tag{7}
$$

该约束确保重写 prompt 的 UMT5 编码保持接近原始编码：

$$
\cos\big(E(c_{\text{rewrite}}),\; E(c_{\text{raw}})\big) \geq 0.95
\tag{8}
$$

这对于维持与 Layer 3（§3.4）的兼容性至关重要，因为参考特征是使用原始 prompt 缓存的。

#### 3.2.3 避免与 Layer 2 的语义冲突

Layer 1 的一个关键设计原则是**不能修改运动相关描述**。理由如下：Layer 2（SVD 噪声先验）提供了从参考视频时序结构中提取的运动方向偏置。如果 prompt 明确指定不同的运动方向（例如，当 SVD 编码 "pan-left" 时 prompt 说 "dolly-in"），两个信号会对 ODE 轨迹产生矛盾约束，导致模型生成混乱的运动模式。

实证上，我们早期的 v7e 策略（添加精确运动描述）结合 SVD 导致 X-CLIP 比基线下降 $-4.1\%$——证实 prompt 中的显式运动与 SVD 先验冲突。我们的首尾替换策略刻意仅操作具有非运动语义的边界位置，将运动引导完全委托给 Layer 2。

---

### 3.3 Layer 2: 基于 SVD 的运动先验注入

#### 3.3.1 概述

Layer 2 的核心思想是将参考视频的时间动态编码为结构化的初始噪声 $z_T$，从而偏置 ODE 轨迹以生成相似的运动模式。这通过三个步骤实现：(1) Flow Matching 反演将 $V_{\text{ref}}$ 映射回噪声空间，(2) SVD 滤波从空间（外观）组件中分离时间（运动）组件，(3) 将运动信号受控地混入随机初始噪声。

#### 3.3.2 Flow Matching 反演

为了获得参考视频在噪声空间中的表示，我们沿反方向求解 ODE，从 $t = 1$（数据）到 $t = 0$（噪声）：

$$
z_{t - \Delta t} = z_t - \Delta t \cdot v_\theta(z_t, t, c), \quad \Delta t = \frac{1}{N_{\text{inv}}}
\tag{9}
$$

以 $z_1 = \mathcal{E}(V_{\text{ref}})$ 为初始条件，$N_{\text{inv}} = 50$ 个反演步。引导尺度设为 $1.0$（无分类器自由引导），以避免将 CFG 引入的偏差引入反演噪声。最终输出 $\eta_{\text{inv}} = z_0$ 表示参考视频在噪声空间中的编码。

为了获得更高的重建精度，我们可选地采用二阶中点法：

$$
k_1 = v_\theta(z_t, t, c)
\tag{10a}
$$
$$
z_{\text{mid}} = z_t - \frac{\Delta t}{2} \cdot k_1
\tag{10b}
$$
$$
k_2 = v_\theta\left(z_{\text{mid}},\; t - \frac{\Delta t}{2},\; c\right)
\tag{10c}
$$
$$
z_{t - \Delta t} = z_t - \Delta t \cdot k_2
\tag{10d}
$$

这将截断误差从 $\mathcal{O}(\Delta t^2)$ 降低到 $\mathcal{O}(\Delta t^3)$，代价是每步模型评估次数加倍。

#### 3.3.3 两阶段 SVD 滤波

反演噪声 $\eta_{\text{inv}} \in \mathbb{R}^{C \times F \times H' \times W'}$ 同时编码外观信息（物体外观）和运动信息（物体运动方式）。我们设计了两阶段 SVD 分解来解耦这些组件。

**阶段 1：空间去内容化。** 通过沿空间维度执行 SVD 来去除主导的空间（外观）组件：

$$
\eta_{\text{inv}} \xrightarrow{\text{reshape}} M_s \in \mathbb{R}^{(C \cdot F) \times (H' \cdot W')}
\tag{11}
$$
$$
M_s = U_s \Sigma_s V_s^\top
\tag{12}
$$

我们确定最小秩 $k_s$，使得去除前 $k_s$ 个奇异向量后仍保留至少 $\rho_s$ 比例的总能量：

$$
k_s = \min\left\{k : \frac{\sum_{i=k+1}^{r} \sigma_{s,i}^2}{\sum_{i=1}^{r} \sigma_{s,i}^2} \geq \rho_s \right\}
\tag{13}
$$

滤波后的噪声和空间残差为：

$$
\eta_{\text{filtered}} = M_s - \sum_{i=1}^{k_s} \sigma_{s,i} \cdot \mathbf{u}_{s,i} \mathbf{v}_{s,i}^\top
\tag{14}
$$
$$
\eta_{\text{spatial}} = \sum_{i=1}^{k_s} \sigma_{s,i} \cdot \mathbf{u}_{s,i} \mathbf{v}_{s,i}^\top
\tag{15}
$$

**直觉。** 外观信息（静态场景内容、物体纹理）在时间维度上高度相关，表现为主导的空间奇异向量。去除这些后主要留下帧间变化——运动信号。

**阶段 2：时间保留。** 我们通过沿时间维度执行 SVD 将剩余信号进一步集中在主导的时间模式上：

$$
\eta_{\text{filtered}} \xrightarrow{\text{reshape}} M_m \in \mathbb{R}^{(C \cdot H' \cdot W') \times F}
\tag{16}
$$
$$
M_m = U_m \Sigma_m V_m^\top
\tag{17}
$$

我们保留捕获 $\rho_m$ 比例时间能量的前 $k_m$ 个时间奇异向量：

$$
k_m = \min\left\{k : \frac{\sum_{i=1}^{k} \sigma_{m,i}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2} \geq \rho_m \right\}
\tag{18}
$$
$$
\eta_{\text{temporal}} = \sum_{i=1}^{k_m} \sigma_{m,i} \cdot \mathbf{u}_{m,i} \mathbf{v}_{m,i}^\top
\tag{19}
$$

**直觉。** 主导的时间模式捕获了帧间变化的主要方向——全局运动模式。次要时间组件对应于噪声或局部抖动。仅保留主导模式可产生干净的运动先验。

**默认参数：** $\rho_s = 0.1$，$\rho_m = 0.9$。为了计算效率，阶段 1 使用随机 SVD（`torch.svd_lowrank`，秩 $q = \min(\min(CF, H'W'), \max(50, 0.3 \cdot \min(CF, H'W')))$），因为空间维度较大；阶段 2 使用完整 SVD，因为 $F = 21$ 较小。

#### 3.3.4 噪声混合

提取的运动先验与随机噪声混合以构造结构化初始隐状态：

**两组件混合**（仅运动）：

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{1 - \alpha} \cdot \eta_{\text{random}}, \quad \eta_{\text{random}} \sim \mathcal{N}(0, I)
\tag{20}
$$

**三组件混合**（运动 + 外观）：

$$
z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1 - \alpha - \beta} \cdot \eta_{\text{random}}
\tag{21}
$$

其中空间组件需要**量级匹配**（重归一化）以确保 $\alpha$ 和 $\beta$ 具有可比的语义重要性：

$$
\hat{\eta}_{\text{spatial}} = \eta_{\text{spatial}} \cdot \frac{\text{std}(\eta_{\text{temporal}})}{\text{std}(\eta_{\text{spatial}})}
\tag{22}
$$

**重归一化的动机。** 经验上，$\text{std}(\eta_{\text{spatial}}) \approx 0.9\text{--}1.2$ 而 $\text{std}(\eta_{\text{temporal}}) \approx 0.28\text{--}0.41$。不进行重归一化时，系数为 $\sqrt{\beta}$ 的空间组件会注入 $\sqrt{\beta} \times 1.0 \approx 0.032$ 的能量，而时间组件仅注入 $\sqrt{\alpha} \times 0.35 \approx 0.022$ 的能量——尽管系数较小，外观信号仍会占据主导。

**关键设计约束。** 混合系数必须极小：$\alpha = 0.003$，产生 $\sqrt{\alpha} \approx 0.055$。实际注入的能量约为 $0.055 \times 0.35 \approx 0.019$，这比单位方差随机噪声的 $2\%$ 还少。大量消融（超过 10 种配置，包括频域重塑、通道集中注入、多尺度分解、相位插值和 SGA 自适应）证实，**$\alpha \approx 0.003$ 是黑盒注入的绝对上限**。原因是根本性的：Flow Matching 模型在严格假设 $z_T \sim \mathcal{N}(0, I)$ 下训练；任何偏离此分布的结构化偏差都会导致 ODE 轨迹偏离学习的数据流形，降低生成质量。

#### 3.3.5 时间信号可靠性（TSR）

不同视频表现出截然不同的运动特性：物体运动视频（如动物奔跑）对 $\alpha$ 不敏感，而场景运动视频（如缓慢的室内镜头平移）对 $\alpha$ 极度敏感（$\alpha$ 从 0.001 增加到 0.002 导致 X-CLIP 下降 0.075）。我们提出 TSR 实现逐样本自适应 $\alpha$ 缩放。

TSR 由两个互补信号组成：

**时间集中度比（TCR）。** 衡量时间能量集中在第一个奇异模式中的程度：

$$
\text{TCR} = \frac{\sigma_{m,1}^2}{\sum_{i=1}^{r} \sigma_{m,i}^2}
\tag{23}
$$

高 TCR 表示单一主导运动方向（可靠信号）；低 TCR 表示分散、可能有噪声的时间结构。

**时间自相关（TAC）。** 衡量时间信号的帧间一致性：

$$
\text{TAC} = \frac{1}{F-1} \sum_{i=1}^{F-1} \frac{\langle \eta_{\text{temporal}}^{(i)}, \eta_{\text{temporal}}^{(i+1)} \rangle}{\|\eta_{\text{temporal}}^{(i)}\| \cdot \|\eta_{\text{temporal}}^{(i+1)}\|}
\tag{24}
$$

其中 $\eta_{\text{temporal}}^{(i)} \in \mathbb{R}^{C \cdot H' \cdot W'}$ 是第 $i$ 帧展平的时间组件。

**TSR 计算。** 我们对两个信号进行归一化并以乘法方式组合：

$$
\text{TCR}_{\text{norm}} = \sigma\big(s_{\text{tcr}} \cdot (\text{TCR} - \mu_{\text{tcr}})\big)
\tag{25}
$$
$$
\text{TAC}_{\text{norm}} = \max(0, \text{TAC})
\tag{26}
$$
$$
\text{TSR} = \text{TCR}_{\text{norm}} \times \text{TAC}_{\text{norm}}
\tag{27}
$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数，$s_{\text{tcr}} = 10.0$（斜率），$\mu_{\text{tcr}} = 0.1$（中心）。

**自适应 $\alpha$。** 逐样本混合系数为：

$$
\alpha_{\text{adaptive}} = \alpha_{\min} + \text{TSR} \cdot (\alpha_{\max} - \alpha_{\min})
\tag{28}
$$

其中 $\alpha_{\min} = 0.0$ 和 $\alpha_{\max} = 0.003$，并受下限约束 $\alpha_{\text{eff}} \geq \alpha_{\text{floor}} = 0.001$。

**语义。** 当时间信号既集中（清晰的主导方向）又连贯（帧间一致）时 TSR 较高——表明运动先验可靠，可以容忍更强的注入。当运动较弱、分散或不连贯时 TSR 较低——表明应最小化注入以避免引入噪声。

#### 3.3.6 PNA 高斯门控

TSR 依赖于 SVD 信号本身的统计特性，但无法直接评估 $\eta_{\text{temporal}}$ 对模型生成方向的实际效用。我们提出 **PNA（Prompt-Noise Alignment）高斯门控**，通过模型自身的一次前向探测来在线评估 $\eta_{\text{temporal}}$ 的方向是否有利。

**动机。** 给定文本条件 $c$ 和初始噪声 $z_T$，不同的 $\eta_{\text{temporal}}$ 方向可能导致截然不同的生成质量。PNA 通过比较"包含运动先验的噪声"和"纯随机噪声"在模型中间层产生的特征差异，判断运动方向是否与文本条件对齐。

**探测过程。** 在 $t = 0.95$（接近纯噪声）执行单步前向传播：

$$
\text{feat}_{\text{mixed}} = h^{(l)}_{\text{mid}}\big(\sqrt{\alpha_{\text{test}}} \cdot \eta_{\text{temporal}} + \sqrt{1-\alpha_{\text{test}}} \cdot \eta_{\text{random}},\; t,\; E(c)\big)
\tag{29a}
$$
$$
\text{feat}_{\text{random}} = h^{(l)}_{\text{mid}}\big(\eta_{\text{random}},\; t,\; E(c)\big)
\tag{29b}
$$

其中 $l = 15$（中间层），$\alpha_{\text{test}} = 0.004$。

**特征差异分析。** 计算两个关键指标：

$$
\Delta h = \text{feat}_{\text{mixed}} - \text{feat}_{\text{random}}
\tag{30}
$$
$$
\text{relative\_impact} = \frac{\|\Delta h\|}{\|\text{feat}_{\text{random}}\|}
\tag{31}
$$

将 $\Delta h$ 沿序列维度分为前半部分和后半部分，计算方向一致性：

$$
\text{cos\_consistency} = \cos\big(\Delta h_{\text{first\_half}},\; \Delta h_{\text{second\_half}}\big)
\tag{32}
$$

**高斯映射。** 将原始 PNA 信号映射到 $\alpha$ 的有效范围：

$$
\text{pna\_raw} = \text{relative\_impact} \times (0.5 + 0.5 \times \text{cos\_consistency})
\tag{33}
$$
$$
\alpha_{\text{eff}} = \alpha_{\text{floor}} + (\alpha_{\text{max}} - \alpha_{\text{floor}}) \cdot \exp\left(-\frac{(\text{pna\_raw} - \mu_{\text{pna}})^2}{2\sigma_{\text{pna}}^2}\right)
\tag{34}
$$

其中 $\mu_{\text{pna}} = 0.003$，$\sigma_{\text{pna}} = 0.001$，$\alpha_{\text{floor}} = 0.0005$，$\alpha_{\text{max}} = 0.006$。

**标准差门控保护。** 为了防止 $\eta_{\text{temporal}}$ 方向不稳定时过度注入，当时间信号的标准差过低时施加额外约束：

$$
\text{std}_{\text{temporal}} = \text{std}(\eta_{\text{temporal}})
\tag{35}
$$

当 $\text{std}_{\text{temporal}} < 0.33$ 时，将 $\alpha_{\text{eff}}$ 限制为 $\alpha_{\text{pna\_min}} \times 2.0$。

**PNA 诊断分类。** 系统自动将探测结果分为 9 类（信号强度 × 方向一致性），用于调试和分析：

| 信号强度 | 方向一致性 | 诊断类别 |
|----------|------------|----------|
| WEAK | LOW_COS | 可能有害 |
| WEAK | MID_COS | 不确定 |
| WEAK | HIGH_COS | 保守注入 |
| MODERATE | LOW_COS | 方向冲突 |
| MODERATE | MID_COS | 正常情况 |
| MODERATE | HIGH_COS | 理想情况 |
| STRONG | LOW_COS | 信号混乱 |
| STRONG | MID_COS | 需谨慎 |
| STRONG | HIGH_COS | 最佳情况 |

#### 3.3.7 关于 $\eta_{\text{temporal}}$ 重归一化的危害性

自然的预处理步骤是在混合前将 $\eta_{\text{temporal}}$ 重归一化为单位方差。然而，这会破坏重要的**隐式自适应性**：不同样本产生的 $\eta_{\text{temporal}}$ 具有天然不同的标准差（$0.28$–$0.41$）。运动信号较弱的样本具有较小的标准差（适当地接受较少注入），而运动信号较强的样本具有较大的标准差。重归一化强制所有样本接受相等的注入能量，导致对弱运动样本的过度注入。实证上，重归一化使 X-CLIP 下降 $-4.7\%$。

---

### 3.4 Layer 3: 自适应特征注入

#### 3.4.1 动机

Layer 2 只能影响 ODE 的**起点** $z_T$，注入的信息在 $N = 30$ 个积分步骤中随着模型的生成先验逐渐被稀释。消融实验确认 $\alpha = 0.003$ 是绝对上限——增加它会降低质量而不是增强引导。特征注入通过在 ODE 轨迹**内部**操作来规避这一限制，在每个去噪步骤修改 DiT 的中间表示。

概念上，如果 Layer 2 决定"ODE 从哪里开始"，Layer 3 则决定"ODE 如何演化"——提供逐步的语义约束，引导生成朝向参考视频的外观和结构，类似于零训练的 ControlNet 但无需任何辅助网络。

#### 3.4.2 参考特征缓存

在 Flow Matching 反演过程中（§3.3.2），我们通过注册在目标层 $\mathcal{L}_{\text{target}}$ 上的前向 hook 同时缓存 DiT 的中间激活：

$$
h_{\text{ref}}^{(l, t)} = \text{CrossAttn}_l\big(z_t, t, E(c)\big), \quad l \in \mathcal{L}_{\text{target}},\; t \in \{t_0, \ldots, t_{N-1}\}
\tag{36}
$$

其中 $\mathcal{L}_{\text{target}} = \{10, 11, \ldots, 19\}$（30 层 DiT 的中间 10 层）。我们特别缓存**交叉注意力输出**，因为它代表了与内容和运动引导最相关的文本条件语义特征。

**缓存索引。** 由于反演使用 $N_{\text{inv}} = 50$ 步而生成使用 $N_{\text{gen}} = 30$ 步，我们将生成步骤 $i$ 映射到最近的反演时间步：

$$
t_{\text{gen}}^{(i)} = \frac{i + 1}{N_{\text{gen}}}, \quad h_{\text{ref}}^{(l, i)} = h_{\text{ref}}^{(l,\; \text{nearest}(t_{\text{gen}}^{(i)}))}
\tag{37}
$$

#### 3.4.3 自适应门控注入

在每个生成步骤 $i$ 和目标层 $l$，当前生成激活 $h_{\text{gen}}^{(l,i)}$ 与缓存的参考特征混合：

$$
h_{\text{out}}^{(l,i)} = (1 - \lambda_{\text{eff}}^{(l,i)}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}}^{(l,i)} \cdot \tilde{h}_{\text{ref}}^{(l,i)}
\tag{38}
$$

其中 $\tilde{h}_{\text{ref}}$ 表示 EMA 平滑的参考特征（§3.4.6），$\lambda_{\text{eff}}$ 是通过三个调制阶段计算的有效注入强度：

**阶段 (a)：时间调度。** 基础注入强度随去噪步骤变化，遵循正弦（中峰）曲线：

$$
\lambda_{\text{base}}(i) = \lambda_{\max} \cdot \sin\left(\frac{\pi \cdot i}{N - 1}\right)
\tag{39}
$$

**原理。** 早期步骤（$i \approx 0$）决定全局结构；晚期步骤（$i \approx N$）细化高频细节。两个极端都受益于更多的模型自由度。决定语义内容和运动模式的中间步骤从参考引导中受益最大。

**阶段 (b)：余弦相似度门控。** 当生成特征已经与参考特征对齐良好时，我们抑制注入：

$$
s^{(l,i)} = \frac{\langle h_{\text{gen}}^{(l,i)},\; \tilde{h}_{\text{ref}}^{(l,i)} \rangle}{\|h_{\text{gen}}^{(l,i)}\| \cdot \|\tilde{h}_{\text{ref}}^{(l,i)}\|}
\tag{40}
$$
$$
g^{(l,i)} = 1 - \sigma\big(\tau \cdot (s^{(l,i)} - 0.5)\big)
\tag{41}
$$

其中 $\tau = 5.0$ 是温度参数。当相似度高（$s > 0.5$）时，门关闭（$g \to 0$），防止不必要的过度约束。当相似度低（$s < 0.5$）时，门打开（$g \to 1$），施加校正注入。

**阶段 (c)：质量尺度。** 并非所有参考视频都提供可靠的引导。运动不连贯（如噪声主导的场景）的视频产生不可靠的缓存特征。我们基于 $\eta_{\text{temporal}}$ 的时间连贯性计算逐样本质量分数：

$$
\bar{s}_{\text{frame}} = \frac{1}{F-1} \sum_{i=1}^{F-1} \cos\big(\eta_{\text{temporal}}^{(i)},\; \eta_{\text{temporal}}^{(i+1)}\big)
\tag{42}
$$
$$
\text{QS} = 0.1 + 0.9 \cdot \sigma\big(20 \cdot (\bar{s}_{\text{frame}} - \theta)\big)
\tag{43}
$$

其中 $\theta = 0.05$。当运动连贯性高时，$\text{QS} \approx 1.0$；当运动不连贯时，$\text{QS} \approx 0.1$，显著抑制注入。

**组合有效强度：**

$$
\lambda_{\text{eff}}^{(l,i)} = \lambda_{\text{base}}(i) \cdot g^{(l,i)} \cdot \text{QS}
\tag{44}
$$

#### 3.4.4 层选择策略

30 个 DiT Transformer 块在不同抽象级别编码信息：

- **早期层（0–9）：** 低级空间特征、边缘、纹理。
- **中间层（10–19）：** 语义结构、物体身份、运动模式。
- **晚期层（20–29）：** 高频细节、精细纹理。

我们注入中间层（$\mathcal{L}_{\text{target}} = \{10, \ldots, 19\}$），因为它们编码了与保持参考视频身份和运动最相关的语义级信息，同时允许模型在早期/晚期层自由处理低级重建和精细细节。

#### 3.4.5 替代时间调度

除了默认的正弦调度，我们还支持：

- **常数：** $\lambda_{\text{base}}(i) = \lambda_{\max}$
- **预热衰减：** 在前 20% 步线性预热（$\lambda_{\text{base}} = \lambda_{\max} \cdot (0.5 + 0.5 \cdot p)$，其中 $p = i / (0.2N)$），然后余弦衰减。
- **余弦衰减：** $\lambda_{\text{base}}(i) = \lambda_{\max} \cdot \cos\left(\frac{\pi}{2} \cdot \frac{i}{N-1}\right)$

消融显示 `middle_peak` 总体表现最佳，因为它将注入集中在语义关键的中间去噪步骤。

#### 3.4.6 EMA 特征平滑

由于反演过程中的数值误差或反演与生成之间的离散时间步不匹配，缓存的参考特征可能存在时间不连续性。我们应用跨去噪步骤的指数移动平均（EMA）平滑：

$$
\tilde{h}_{\text{ref}}^{(l, i)} = \gamma \cdot \tilde{h}_{\text{ref}}^{(l, i-1)} + (1 - \gamma) \cdot h_{\text{ref}}^{(l, i)}
\tag{45}
$$

衰减因子 $\gamma = 0.7$。这抑制了特征突变并产生更平滑的引导信号。

#### 3.4.7 与 ODE 动力学的关系

特征注入不直接修改 ODE 状态变量 $x_t$。相反，它改变了速度场预测 $v_\theta$ 内部的中间计算。形式上，步骤 $i$ 处的修正速度可以写为：

$$
\tilde{v}_\theta(x_t, t, c) = v_\theta(x_t, t, c) + \Delta v^{(L3)}(x_t, t, c, \{h_{\text{ref}}\})
\tag{46}
$$

其中 $\Delta v^{(L3)}$ 表示由中间层特征替换引起的隐式速度扰动。该公式明确表明 Layer 3 在每一步修改 ODE 轨迹的**方向**，而 Layer 2 修改**起点**——两者在 ODE 动力学的互补方面操作。

---

### 3.5 统一三层公式

#### 3.5.1 完整生成过程

结合所有三层，完整的 P-Flow 生成可以写为：

$$
V_{\text{gen}} = \mathcal{D}\left(\text{ODE-Solve}_{t=0}^{t=1}\Big(z_T^{(L2)},\; \tilde{v}_\theta(\cdot, \cdot, E(c^{(L1)}); \{h_{\text{ref}}\})\Big)\right)
\tag{47}
$$

其中：

- $c^{(L1)}$ 是最小重写的 prompt（§3.2），
- $z_T^{(L2)}$ 是 SVD 结构化的初始噪声（§3.3），
- $\tilde{v}_\theta$ 是特征注入的速度场（§3.4），
- $\mathcal{D}$ 是 VAE 解码器。

#### 3.5.2 逐步计算

在每个去噪步骤 $i \in \{0, 1, \ldots, N-1\}$：

1. **计算目标层的生成特征：**
$$
h_{\text{gen}}^{(l,i)} = \text{CrossAttn}_l\big(x_{t_i}, t_i, E(c^{(L1)})\big), \quad l \in \mathcal{L}_{\text{target}}
$$

2. **计算自适应门控：**
$$
s^{(l,i)} = \cos(h_{\text{gen}}^{(l,i)}, \tilde{h}_{\text{ref}}^{(l,i)}), \quad g^{(l,i)} = 1 - \sigma(\tau \cdot (s^{(l,i)} - 0.5))
$$

3. **计算有效注入强度：**
$$
\lambda_{\text{eff}}^{(l,i)} = \lambda_{\max} \cdot \sin\left(\frac{\pi i}{N-1}\right) \cdot g^{(l,i)} \cdot \text{QS}
$$

4. **注入参考特征：**
$$
h_{\text{out}}^{(l,i)} = (1 - \lambda_{\text{eff}}^{(l,i)}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}}^{(l,i)} \cdot \tilde{h}_{\text{ref}}^{(l,i)}
$$

5. **ODE 步进**（使用注入特征后的修正速度）：
$$
x_{t_{i+1}} = x_{t_i} + \Delta t \cdot \tilde{v}_\theta(x_{t_i}, t_i, c^{(L1)})
$$

6. **Layer 2 的初始条件：**
$$
x_{t_0} = z_T^{(L2)} = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}}
$$

#### 3.5.3 正交性与协同

三层表现出正交操作和协同增强：

**正交性。** 每层在不同的数学空间中操作：

- L1：文本嵌入空间 $\mathbb{R}^{S \times d}$（修改交叉注意力键/值）
- L2：噪声空间 $\mathbb{R}^{C \times F \times H' \times W'}$（修改 ODE 初始条件）
- L3：特征空间 $\mathbb{R}^{B \times N_v \times d_{\text{model}}}$（修改中间激活）

这些空间之间没有直接的代数耦合。

**协同。** 尽管独立操作，这些层实现了超加性性能。单独 L2 实现 +5.1% X-CLIP，单独 L3 实现 +7.0%。组合使用时实现 +8.6%——超过单独贡献的总和，表明存在正向相互作用。机制是：

- L2 将 ODE 起点偏向参考运动流形，确保早期去噪步骤已经产生几何上接近参考的特征。
- L3 利用这种接近性：当 $h_{\text{gen}}$ 起始更接近 $h_{\text{ref}}$（由于 L2）时，自适应门控需要较少的激进干预，产生更平滑和稳定的注入。

**冲突避免。** L1–L2 冲突（运动描述 vs. SVD 先验）通过首尾策略的显式约束解决：不修改内部运动描述。L1–L3 兼容性通过 $\leq 8\%$ 编辑率维持，确保 prompt 嵌入漂移足够小，使得在 $c_{\text{raw}}$ 下缓存的特征与在 $c_{\text{rewrite}}$ 下生成的特征保持对齐。

---

### 3.6 算法总结

我们在算法 1 中展示完整的 P-Flow 算法。

---

**算法 1：** P-Flow：参考引导的视频生成

---

**输入：** 参考视频 $V_{\text{ref}}$，caption $c_{\text{raw}}$，模型 $v_\theta$，VAE 编码器 $\mathcal{E}$，解码器 $\mathcal{D}$

**输出：** 生成的视频 $V_{\text{gen}}$

**超参数：** $N=30$, $N_{\text{inv}}=50$, $\alpha_{\max}=0.003$, $\lambda_{\max}=0.1$, $\rho_s=0.1$, $\rho_m=0.9$, $\tau=5.0$, $\gamma=0.7$, $\theta=0.05$

---

**// Layer 1：最小化 Prompt 重写**

1: $c \leftarrow \text{HeadTailReplace}(c_{\text{raw}})$ $\quad$ // LLM 首尾关键词替换

2: $c \leftarrow \text{VLMCorrect}(c, V_{\text{ref}})$ $\quad$ // VLM 事实性校正（$\leq 3$ 处编辑）

**// Layer 2：SVD 噪声先验**

3: $z_1 \leftarrow \mathcal{E}(V_{\text{ref}})$ $\quad$ // VAE 编码

4: $\eta_{\text{inv}} \leftarrow \text{Invert}(z_1, v_\theta, c, N_{\text{inv}})$ $\quad$ // Flow Matching 反演（公式 9）

5: $\eta_{\text{filtered}}, \eta_{\text{spatial}} \leftarrow \text{SVD-Stage1}(\eta_{\text{inv}}, \rho_s)$ $\quad$ // 空间去内容化（公式 11–15）

6: $\eta_{\text{temporal}} \leftarrow \text{SVD-Stage2}(\eta_{\text{filtered}}, \rho_m)$ $\quad$ // 时间保留（公式 16–19）

7: $\text{TSR} \leftarrow \text{ComputeTSR}(\eta_{\text{temporal}})$ $\quad$ // 自适应信号可靠性（公式 23–27）

8: $\alpha \leftarrow \alpha_{\min} + \text{TSR} \cdot (\alpha_{\max} - \alpha_{\min})$ $\quad$ // 自适应混合系数（公式 28）

9: $z_T \leftarrow \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{\beta} \cdot \hat{\eta}_{\text{spatial}} + \sqrt{1-\alpha-\beta} \cdot \eta_{\text{random}}$ $\quad$ // 噪声混合（公式 21）

**// Layer 3：特征注入准备**

10: $\{h_{\text{ref}}^{(l,t)}\} \leftarrow \text{CacheFeatures}(z_1, v_\theta, c, \mathcal{L}_{\text{target}}, N_{\text{inv}})$ $\quad$ // 反演期间缓存（公式 36）

11: $\text{QS} \leftarrow \text{QualityScale}(\eta_{\text{temporal}}, \theta)$ $\quad$ // 逐样本质量（公式 42–43）

**// 三层引导生成**

12: $x_0 \leftarrow z_T$

13: **for** $i = 0$ **to** $N - 1$ **do**

14: $\quad$ $t_i \leftarrow i / N$

15: $\quad$ **for** $l \in \mathcal{L}_{\text{target}}$ **do**

16: $\quad\quad$ $h_{\text{gen}}^{(l,i)} \leftarrow \text{CrossAttn}_l(x_{t_i}, t_i, E(c))$

17: $\quad\quad$ $\tilde{h}_{\text{ref}}^{(l,i)} \leftarrow \gamma \cdot \tilde{h}_{\text{ref}}^{(l,i-1)} + (1-\gamma) \cdot h_{\text{ref}}^{(l,i)}$ $\quad$ // EMA 平滑

18: $\quad\quad$ $s \leftarrow \cos(h_{\text{gen}}^{(l,i)}, \tilde{h}_{\text{ref}}^{(l,i)})$

19: $\quad\quad$ $\lambda_{\text{eff}} \leftarrow \lambda_{\max} \cdot \sin(\pi i / (N{-}1)) \cdot [1 - \sigma(\tau(s{-}0.5))] \cdot \text{QS}$

20: $\quad\quad$ $h_{\text{out}}^{(l,i)} \leftarrow (1 - \lambda_{\text{eff}}) \cdot h_{\text{gen}}^{(l,i)} + \lambda_{\text{eff}} \cdot \tilde{h}_{\text{ref}}^{(l,i)}$

21: $\quad$ **end for**

22: $\quad$ $x_{t_{i+1}} \leftarrow x_{t_i} + \Delta t \cdot \tilde{v}_\theta(x_{t_i}, t_i, c)$

23: **end for**

24: $V_{\text{gen}} \leftarrow \mathcal{D}(x_1)$ $\quad$ // VAE 解码

25: **return** $V_{\text{gen}}$

---

### 3.7 讨论

#### 3.7.1 与基于训练方法的比较

与需要训练适配器模块的 IP-Adapter、ControlNet 或 VideoComposer 不同，P-Flow 完全在推理时通过操作现有模型接口运行。这种零训练特性提供三个优势：(1) 无额外训练成本，(2) 可立即应用于任何 Flow Matching T2V 模型，(3) 无灾难性遗忘或过拟合到训练数据分布的风险。

#### 3.7.2 敏感性分析

三层表现出显著不同的敏感性特征：

- **L1（Prompt）：** 对实现选择鲁棒；关键约束是维持 $\leq 8\%$ 编辑率。
- **L2（SVD 噪声）：** 对 $\alpha$ 极度敏感；有效范围为 $[0.001, 0.005]$，超出此区间会导致严重退化。
- **L3（特征注入）：** 对 $\lambda_{\max}$ 和层选择中度敏感；自适应门控机制提供自我调节，降低超参数敏感性。

不对称的敏感性促使 TSR（自适应 $\alpha$）和 Layer 3 中多阶段门控的设计——两者都旨在使系统在不同视频类型上更加鲁棒。

#### 3.7.3 计算开销

相比标准 T2V 生成的额外计算包括：(1) 一次反演（$N_{\text{inv}} = 50$ 次模型评估），(2) SVD 分解（GPU 上 $<1$ 秒），(3) 逐步特征混合（可忽略的 FLOP 开销）。总生成时间相比标准生成增加约 $1.7$ 倍（由反演主导），GPU 内存除特征缓存（10 层 $\times$ 30 步，约 2GB）外无增加。

---

*方法章节结束*