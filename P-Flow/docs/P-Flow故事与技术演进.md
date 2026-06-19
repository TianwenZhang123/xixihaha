# P-Flow 三层架构：理论、故事与演进

> 本文档是 P-Flow 项目的核心理论文档，包含：(1) 论文叙述——学术严谨的三层架构原理，(2) 大白话叙述——直觉理解，(3) 精进方向——Layer 2/3 的优化与融合路径，(4) 后续实验计划。

---

## 第一部分：论文叙述——三层正交互补架构

### 1.1 问题定义

给定参考视频 $V_{\text{ref}}$，使用冻结的文本到视频（T2V）模型 $\mathcal{M}$ 重新生成一个运动一致的视频 $V_{\text{gen}}$，使得 $V_{\text{gen}}$ 在保持 $V_{\text{ref}}$ 的运动结构的同时，由 $\mathcal{M}$ 自主渲染视觉细节。核心挑战在于：T2V 模型的可控接口极其有限——仅有文本 prompt、初始噪声 $z_T$ 和去噪过程的回调钩子。

### 1.2 三层正交互补架构

我们提出一个三层渐进式信息注入框架，在三个正交的操作空间中依次注入参考信息：

$$V_{\text{gen}} = \mathcal{M}\bigl(\underbrace{c_{\text{text}}}_{\text{L1}},\ \underbrace{z_T(\eta_{\text{temporal}})}_{\text{L2}},\ \underbrace{\{h^{\ell}_{\text{ref}}\}_{\ell \in \mathcal{L}}}_{\text{L3}}\bigr)$$

**Layer 1 — Semantic Distillation（语义蒸馏 Prompt）**

操作空间：文本语义空间（UMT5 encoder → cross-attention）

L1 的核心不是"创造新描述"，而是一个**两阶段语义蒸馏**流程——先做减法去噪声，再做加法补视觉事实：

1. **LLM 纯删减**（Step 1）：对 VLM baseline caption 执行纯粹的减法操作——删除开场白（preamble）、模糊修饰（hedging）、结尾总结（summary），不添加任何内容。输出约为原文的 70-85%。
2. **VLM 视觉补充**（Step 2）：Qwen2.5-VL-7B 观看原始视频，同时阅读删减后的 caption，补充缺失的视觉细节（颜色、材质、光照、空间关系）。每一条补充都是 VLM 实际观察到的视觉事实（grounded），而非 LLM 凭空想象。

关键约束：**绝对不修改运动描述**——这是 v9 策略与早期版本的根本区别。LLM 只做删减，VLM 只补视觉细节，两者都不碰运动相关的句子。这为 L2（SVD 运动偏置）和 L3（特征注入）留出了完整的运动信息空间。实验证明，修改运动描述（如 v7e 的精确运动 prompt）与 SVD 运动偏置冲突，导致 XCLIP -4.1%。

设计动机：v8-minimal 实验表明，LLM 凭空添加的相机运动句占用 UMT5 token 但缺乏视觉根据（CLIP 0.8915 < 纯 L2 的 0.8964）。v9 的解决方案是让补充操作完全由 VLM 视觉感知驱动，确保每个新增 token 都有视觉事实支撑。

**Layer 2 — SVD Noise Prior（统计运动偏置注入）**

操作空间：latent 噪声空间（$z_T \in \mathbb{R}^{C \times F \times H \times W}$）

L2 在生成开始前，构造一个携带参考视频运动统计偏置的初始噪声：

$$z_T = \sqrt{\alpha} \cdot \eta_{\text{temporal}} + \sqrt{1 - \alpha} \cdot \eta_{\text{random}}, \quad \alpha = 0.004$$

其中 $\eta_{\text{temporal}}$ 通过两阶段 SVD 滤波从反演噪声 $\eta_{\text{inv}}$ 中提取：

- **Stage 1（空间去内容）**：$\eta_{\text{inv}} \xrightarrow{\text{reshape}} (CF, HW)$，SVD 分解，去除 top-$k_s$ 空间主成分（承载外观/内容信息），保留 $\geq \rho_s = 10\%$ 的残余能量。
- **Stage 2（时间保运动）**：残余信号 $\xrightarrow{\text{reshape}} (CHW, F)$，SVD 分解，保留 top-$k_m$ 时间主成分（承载运动动态），使得累积能量 $\geq \rho_m = 90\%$。

L2 的理论极限：由于 T2V 模型对 $z_T$ 的结构化偏离零容忍，$\alpha = 0.004$ 对应的有效方向偏移仅约 2%（$\sqrt{\alpha} \times \sigma_{\eta_t}/\sigma_{\eta_r} \approx 0.063 \times 0.3 \approx 0.019$）。8 个增强方向（频域重塑/SGA/PODI/CEGI/MSTDI/TPI/OCS/QGA）的系统性消融实验表明，**$\alpha = 0.004$ 是纯黑盒噪声注入的绝对天花板**。

L2 的双重角色：（1）在 P-Flow 原论文的迭代优化 pipeline 中，SVD 提供跨迭代的确定性锚点，稳定 prompt 优化收敛；（2）在我们的单次生成设定中，SVD 提供运动方向的微弱统计偏置，在 30 步 flow matching 积分中被自然放大。

**Layer 3 — Feature Injection with Scene-Aware Triple Gating（场景感知三重门控特征注入）**

操作空间：DiT 中间层特征空间（cross-attention 输出 $h^{\ell} \in \mathbb{R}^{B \times N \times d}$）

L3 在生成过程中，通过 PyTorch forward hook 在 DiT 的中间层注入参考视频的缓存特征：

$$h^{\ell}_{\text{injected}} = (1 - \lambda^{\ell}_{\text{eff}}) \cdot h^{\ell}_{\text{current}} + \lambda^{\ell}_{\text{eff}} \cdot h^{\ell}_{\text{ref}}$$

其中 $\lambda^{\ell}_{\text{eff}}$ 由三重门控联合确定，在**样本级**、**步级**、**层级**三个粒度上自适应调节：

$$\lambda^{\ell}_{\text{eff}} = \lambda_t \cdot \text{QS}_{\text{eff}} \cdot g(h^{\ell}_{\text{current}}, h^{\ell}_{\text{ref}})$$

三重门控的分工：

1. **$\lambda_t = \lambda_{\max} \cdot \sin(\pi t / (N-1))$**：middle-peak 时间调度——两端自由、中间集中
2. **$\text{QS}_{\text{eff}} = \text{QS}_{\text{signal}} \times \max(M_d, \text{fi\_qs\_md\_floor})$**：场景感知质量门控（**样本级**）
   - $\text{QS}_{\text{signal}} \in [0.1, 1.0]$：来自 $\eta_{\text{temporal}}$ 帧间余弦相似度的软门控，度量 SVD 时间信号的统计可靠性
   - $M_d$ 修正：物体运动样本（$M_d=1.0$）完全不衰减；场景/环境样本（$M_d<1$）最多减半（$\text{fi\_qs\_md\_floor}=0.5$）
   - 若 $\text{QS}_{\text{eff}} < 10^{-6}$，直接跳过 FI，退化为标准生成
3. **$g(h_c, h_r) = 1 - \text{sigmoid}(\tau \cdot (\cos(h_c, h_r) - 0.5))$**：自适应特征对齐门控（**步×层级**）
   - $\cos > 0.5$ → 特征已对齐 → gate 小 → 不注入
   - $\cos < 0.5$ → 特征偏离 → gate 大 → 强注入

**直觉理解**：三重门控实现了"需要时注入，不需要时自动跳过"的智能行为。$\text{QS}_{\text{eff}}$ 在生成前一次性判断"这个样本值不值得注入 FI"；$g(h_c, h_r)$ 在每步每层实时判断"这步这层需不需要注入"；$\lambda(t)$ 控制注入的时间节奏。

**两阶段生命周期**：

1. **特征缓存**（生成前）：沿反演轨迹 $\{z_t\}_{t=0}^{T}$ 的每个关键步，对 DiT 做前向传播，通过 hook 捕获 mid-layers（$\ell \in [10, 19]$）的 cross-attention 输出，存为 $\{h^{\ell}_{\text{ref}}(t)\}$。
2. **注入式生成**（生成时）：在 DiT 前向传播中，hook 将 $h^{\ell}_{\text{current}}$ 替换为 $\lambda_{\text{eff}}$ 加权的插值。生成完成后 hook 移除。

关键优势：
- **不修改 ODE 积分路径**：latent $z_t$ 的演化不被干预，只修改 DiT 内部的特征表示
- **零训练**：不需要微调模型任何参数
- **场景自适应**：物体运动全力注入，环境动态谨慎注入，质量极差时自动跳过
- **步级智能**：特征已对齐时自动降低注入，避免过度约束
- **不要求起点对齐**：与 L2(SVD blend) 的 $z_T$ 不在同一轨迹上也能生效

### 1.3 三层正交互补性分析

| 维度 | L1 (Prompt) | L2 (SVD Noise) | L3 (Feature Injection) |
|------|-------------|-----------------|------------------------|
| 操作空间 | 文本语义空间 | latent 噪声空间 | DiT 特征空间 |
| 作用时机 | 生成前（一次性） | 生成前（$z_T$ 构造） | 生成中（每步前向传播） |
| 信息类型 | "什么物体长什么样"（不含运动） | "运动的统计节奏" | "运动的精确结构" |
| 信息带宽 | 中（受文本表达力限制） | 极低（$\alpha \leq 0.004$） | 高（特征空间自由度大） |
| 修改模型 | ❌ | ❌ | 仅 Hook（可逆） |

**互补机制**：

1. **空间正交**：L1 操作文本编码、L2 操作初始噪声、L3 操作中间特征——三者在模型的前向传播路径上处于不同位置，信息注入互不干扰。
2. **时序互补**：L2 在生成起点施加偏置，L3 在生成中期的关键阶段注入（middle-peak 调度），L1 在全局提供语义约束。
3. **信息互补**：L1 提供"什么物体长什么样"的视觉语义（刻意不含运动）、L2 提供"运动的节奏"的统计偏置、L3 提供"怎么动"的精确结构。

### 1.4 实验验证

10 样本（IDs: 7, 17, 21, 31, 32, 33, 34, 43, 46, 47）消融实验：

| 配置 | CLIP | XCLIP | vs Baseline |
|------|------|-------|------------|
| Baseline（裸 VLM caption，无优化） | 0.8703 | 0.7164 | — |
| L1+L2（裸 caption + SVD v1 α=0.004） | 0.8964 | 0.7874 | +3.0% / +9.9% |
| L1+L2+L3（SVD v1 + FI adaptive λ=0.05） | **0.9042** | **0.8138** | **+3.9% / +13.6%** |

---

## 第二部分：大白话叙述——"让 AI 精准重画任意视频"

### 一句话版本

给 AI 一段视频，让它仅凭「剧本 + 草稿纸 + 过程引导」重新生成一个尽可能一样的版本。

### 完整故事

想象你面前有一位画师（Wan2.1 文生视频模型），他只接受三样东西：一段文字描述、一张初始画布、以及画画过程中你可以在他耳边轻声提醒。你的目标是让他画出和你手中参考视频完全一致的作品。

#### 第一步：精炼剧本（Layer 1 — 语义蒸馏 Prompt）

Layer 1 不是"写一段好描述"——它是一个先做减法、再做加法的两步蒸馏流程。

**第一步：请编辑做减法**。观察员（VLM）之前已经写了一段 baseline 描述，但里面有很多废话——"In this video, we can observe that..."这样的开场白、"appears to be"这样的模糊修饰、结尾的总结段落。让 LLM 把这些通通删掉，一个字不加，只做纯粹的删减。删完后文本大约剩原来的 70-85%。

**第二步：请观察员补视觉细节**。让 VLM（Qwen2.5-VL-7B）再次观看原始视频，同时阅读删减后的精简描述，发现"这里还缺颜色信息""那里光照方向没提"，然后把这些**真正看到的**视觉事实补回去。

**最关键的约束**：绝对不碰运动描述。LLM 删减时跳过运动句，VLM 补充时只补视觉细节（颜色、材质、光照、空间位置）不补运动。为什么？因为运动的精确信息要留给 Layer 2（草稿纸暗示运动方向）和 Layer 3（过程中精确引导运动）。如果你在剧本里说"猫从左往右跳"，同时 Layer 2 的草稿纸暗示"运动偏左"，画师就懵了——实验证明这会导致严重的运动冲突。

**为什么不是 LLM 加视觉细节？** 因为 LLM 没看过视频，它添加的"cinematic lighting""smooth camera pan"是凭空想象，占用了宝贵的 token 却没有视觉根据。v8 版本就是这样失败的（CLIP 0.8915 < 纯 L2 的 0.8964）。v9 的解决方案：只让**看过视频的 VLM** 来补充，确保每个新增 token 都是真实的视觉事实。

#### 第二步：配一张有方向感的草稿纸（Layer 2 — SVD Noise Prior）

AI 画画时需要一张"初始画布"——本质上是一团随机噪声。如果完全随机，画师虽然按剧本画，但画面运动全凭运气。

所以从参考视频"倒推"出一张特殊的噪声（Flow Inversion），然后做两步清洗：
- **洗掉外观**：用 SVD 分解，去除那些编码"猫长什么样"的成分（空间主成分），只留下运动相关的残余
- **保留运动**：再用 SVD，保留那些编码"猫怎么动"的成分（时间主成分）

最后，把清洗后的运动噪声和纯随机噪声以极小比例混合（约 0.4% 的运动信号 + 99.6% 的随机）。

这就像给画师一张"带微弱运动暗示的草稿纸"——他不会照着画，但起笔时会不自觉往正确方向偏一点。

**为什么只有 0.4%？** 因为 AI 极度敏感——稍微多一点结构化信号，它就崩了。这不是我们的算法不好，是"修改 AI 的起笔"这件事本身就有极低的带宽。0.4% 是做了 8 种不同增强方案后确认的天花板。

#### 第三步：在画画过程中轻声提醒（Layer 3 — Feature Injection）

剧本写好了，草稿纸也配好了，但画师画到一半可能走偏——运动的节奏、幅度、方向不对。

我们的做法是：**提前录下"参考画师"画画时脑子里的每一步想法**（通过 inversion 缓存 DiT 中间层特征），然后在画师画画时，通过"耳语"（PyTorch Hook）把参考想法注入给他。

具体来说，画师画到第 $t$ 步、思考到中间层 $\ell$ 时，我们偷偷把他的想法 $h_{\text{current}}$ 往参考方向拉一点：

$$h_{\text{injected}} = 0.95 \times h_{\text{current}} + 0.05 \times h_{\text{ref}}$$

**聪明之处**：
- 如果画师已经走对了（$h_{\text{current}}$ 和 $h_{\text{ref}}$ 很接近），自动减少提醒——别打扰他
- 如果画师走偏了（$h_{\text{current}}$ 和 $h_{\text{ref}}$ 差很远），自动增强提醒——拉他一把
- 提醒力度随画画进程变化：前期（纯噪声）不提醒，中期（结构形成时）最积极，后期（画细节时）逐渐收手

**为什么 Layer 3 比 Layer 2 强这么多？** 因为 Layer 2 只能在起笔时给一个暗示，而 Layer 3 可以在画画的每一步都注入精确指导——信息传递的通道宽了几个数量级。Layer 2 像是"起点的一声低语"，Layer 3 像是"全程的同声传译"。

#### 三层合力

- **Layer 1**：给画师一份精炼的剧本——删掉废话，补上真实的视觉细节，但绝不提运动方向
- **Layer 2**：给画师一张"偏向猫跳方向"的草稿纸——运动的节奏和趋势
- **Layer 3**：在画师每一步思考时轻声提醒"注意，猫此时应该是这个姿态"——运动的精确结构

每一层回答一个不同的问题，在不同空间、不同时机注入信息，互不干扰、互为补充。L1 负责"画什么样子"，L2 和 L3 负责"怎么动"——正是因为 L1 克制住了不去描述运动，L2/L3 才有空间精确传递运动信息。

---

## 第三部分：Layer 2 与 Layer 3 的精进方向

### 3.1 Layer 2 SVD Noise Prior 的精进

**核心困境**：$\alpha = 0.004$ 是天花板，8 个方向的增强全部失败。问题不在 SVD 算法，在于 $z_T$ 这个接口的信道容量不足。

**精进方向不是"更强的 SVD"**，而是重新定位 L2 的角色：

#### 方向 1：L2 作为 FI 的质量先验（当前已实现）

L2 的 $\eta_{\text{temporal}}$ 帧间余弦相似度已被用作 FI 的 quality_scale。运动一致性高的样本允许 FI 更强注入，运动混乱的样本自动降低注入。

这个角色已经够了——L2 不需要自己传递多少信息，它帮 L3 判断"该不该传"就够了。

#### 方向 2：验证 SVD 是否冗余（最关键的实验）

如果 FI + 纯随机 $z_T$ ≈ FI + SVD $z_T$（差异 < 1%），则 SVD 是冗余的，论文可以简化为两层架构。这个实验的结论决定论文讲什么故事。

#### 方向 3（如 SVD 不冗余）：理解 SVD 与 FI 的协同机制

如果 FI + SVD 显著优于 FI + 纯随机，说明 SVD 的 2% 方向偏移在 FI 的特征引导下被放大了——这本身是一个有趣的发现。

### 3.2 Layer 3 Feature Injection 的精进

FI 是当前最大的杠杆，有明确的精进空间：

#### 精进 1：统一门控框架（置信度加权注入）

当前有两个独立门控（quality_scale + adaptive_gate），可以统一为一个"置信度加权注入"（Confidence-Weighted Injection）：

$$\lambda^{\ell}_{\text{eff}} = \lambda_t \cdot \underbrace{\text{Confidence}(h^{\ell}_{\text{current}}, h^{\ell}_{\text{ref}}, V_{\text{ref}})}_{\text{统一的置信度}}$$

其中置信度综合了：
- 样本级：$V_{\text{ref}}$ 的运动质量（$\eta_{\text{temporal}}$ 帧间一致性）
- 特征级：$h^{\ell}_{\text{current}}$ 与 $h^{\ell}_{\text{ref}}$ 的对齐度（余弦相似度）

在论文中可以讲成一个干净的数学框架，而不是"两个 ad-hoc 门控"。

#### 精进 2：从线性插值到投影注入

当前注入方式：$h = (1-\lambda) h_c + \lambda h_r$（线性插值）

问题：当 $h_c$ 和 $h_r$ 方向差很大时，线性插值可能产生"中间态"特征。

精进方向：
```python
# 方案 A：残差注入（只加偏移，不替换）
h = h_current + λ_eff * (h_ref - h_current)

# 方案 B：投影注入（只注入正交分量，不改变当前方向）
delta = h_ref - h_current
delta_perp = delta - proj(delta, h_current)  # 去掉与 h_current 平行的分量
h = h_current + λ_eff * delta_perp
```

方案 B 与 VDA 的"只注入正交分量"思想一致，但在特征空间做比在 latent 空间做更鲁棒。

#### 精进 3：跨步特征平滑

当前每步独立取 ref_features[step_idx]，相邻步的特征可能不连续。可以做时序平滑：
```python
h_ref_smooth = 0.7 * h_ref_prev + 0.3 * h_ref_current  # 指数移动平均
```

### 3.3 Layer 2 + Layer 3 的融合路径

#### 当前融合方式

L2 和 L3 已经在代码中融合了：
1. L2 构造 $z_T$（带 SVD 偏置）
2. L3 在生成时注入参考特征
3. L2 的 $\eta_{\text{temporal}}$ 帧间余弦相似度作为 L3 的 quality_scale

但这个融合是"松耦合"的——L2 和 L3 各自独立运行，quality_scale 是唯一的联系。

#### 更深度的融合方向

**方向 A：L2 的质量诊断驱动 L3 的注入策略**

当前 quality_scale 是一个标量，只控制 $\lambda$ 的整体缩放。可以更精细地利用 L2 的信息：
- $\eta_{\text{temporal}}$ 的空间能量分布 → 控制 FI 在哪些空间区域更强注入
- $\eta_{\text{temporal}}$ 的时间变化模式 → 控制 FI 的 middle-peak 调度如何微调

**方向 B：FI 的注入信息回馈 L2**

当前 L2 是一次性的（$z_T$ 构造后不再改变）。如果 FI 在某些层发现注入效果差（gate 持续接近 0），可以动态回退到"更弱 SVD"甚至"无 SVD"模式。

**方向 C：统一的注意力机制**

L2 和 L3 都在"引导模型关注参考视频的运动"，只是操作空间不同。可以设计一个统一的注意力机制，在特征空间同时考虑 SVD 运动先验和 FI 参考特征：
$$h^{\ell}_{\text{out}} = \text{CrossAttn}(h^{\ell}_{\text{current}},\ [h^{\ell}_{\text{ref}},\ \text{SVD\_embedding}])$$

---

## 第四部分：Layer 1 的深度理解

### L1 不只是 Prompt 生成

从实验中可以看到，L1（v9 策略）做了以下工作：

1. **LLM 纯删减**：对 VLM baseline caption 做减法——删除开场白、模糊修饰、结尾总结，一个字不加
2. **VLM 视觉补充**：Qwen2.5-VL-7B 观看原始视频 + 阅读删减后 caption，补充缺失的视觉事实（颜色、材质、光照、空间关系）
3. **运动描述守恒原则**：绝对不碰运动描述——LLM 删减时跳过运动句，VLM 补充时只补视觉细节不补运动。这是 L1 与 L2/L3 分工的基石

L1 的真正角色是**语义蒸馏与锚定**——通过"先减噪、再补事实"，让每个 token 都承载有效信息，同时为 L2/L3 留出完整的"怎么动"的信息空间。

早期的 V4 策略（subject-first + temporal chain + visual keyword）采用"手术式修改"思路，但仍有 LLM 凭空添加信息的问题。v9 的核心洞察：**LLM 不应该添加任何东西**——它只负责删除噪声；所有新增内容必须来自 VLM 的视觉感知，确保信息有视觉根据（grounded）。

### L1 与 L3 的深层交互

L1 的 caption 被 UMT5 编码后，通过 cross-attention 注入 DiT。而 FI 正是 hook 在 cross-attention 的输出上。这意味着：

> L1 决定了 $h_{\text{current}}$ 的语义基底，L3(FI) 决定了 $h_{\text{current}}$ 向 $h_{\text{ref}}$ 的偏移量。当 L1 的 caption 描述的运动方向与 $h_{\text{ref}}$ 不一致时，adaptive_gate 会自动降低 $\lambda_{\text{eff}}$——FI 的自适应门控天然解决了"L1 精确 vs 模糊"的矛盾。

---

## 第五部分：论文故事凝练

### 故事方案 A：三层正交互补

> **标题**：From Noise to Features: Multi-Scale Reference Injection for Zero-Shot Video Motion Transfer
>
> **核心叙事**：给定参考视频，如何在 T2V 模型的三个可控接口中注入运动信息？L1 在语义空间、L2 在噪声空间、L3 在特征空间——三者的信息注入正交不冲突，每一层都是上一层的残差补充。

### 故事方案 B：Feature Injection 为核心

> **标题**：Feature Injection: Training-Free Reference Guidance for Video Diffusion Models
>
> **核心叙事**：（如果 FI-without-SVD 实验证明 SVD 冗余）聚焦 FI 本身——零训练、自适应、模型无关的参考信息注入。对比 ControlNet（需训练）、Prompt-to-Prompt（改 attention map）、SVDInit（只改起点），FI 在特征空间做残差注入，信道容量最大、最鲁棒。

### 故事方案 C：信息论视角

> **标题**：Where to Inject? Information-Theoretic Analysis of Reference Guidance for Video Diffusion Models
>
> **核心叙事**：从信息论角度统一理解三层架构。L2 的失败不是算法问题，是 latent 噪声空间的信道容量不足；FI 成功是因为特征空间的信道容量远大于噪声空间。三层架构的本质是在 T2V 模型的不同信息通道中选择信道容量最大的位置注入。

---

## 第六部分：后续实验计划

### 🔴 优先级 1：FI-without-SVD 消融（30 分钟）

**目的**：确定 SVD 是否冗余，决定论文架构

**命令**：
```bash
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/FI_noSVD_10samples \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --inversion --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose
```

注意：**不加** `--svd --blend --alpha 0.004`，$z_T$ 使用纯随机噪声。

**判定标准**：
- FI+纯随机 ≈ FI+SVD（差异 < 1%）→ SVD 冗余，论文选故事 B
- FI+SVD 显著优于 FI+纯随机 → 三层正交互补成立，论文选故事 A

### 🟡 优先级 2：FI λ 超参数搜索（1.5 小时）

**目的**：确定 λ 的最优区间，画出 λ-XCLIP 曲线

**实验矩阵**：

| λ | 预期 |
|---|------|
| 0.02 | 更保守 |
| 0.05 | 当前设置（基线） |
| 0.10 | 更激进 |
| 0.20 | 极限测试 |

每个 λ 跑 10 样本，共 3 组额外实验（λ=0.02, 0.10, 0.20）。

### 🟡 优先级 3：FI 注入层位置消融（1.5 小时）

**目的**：证明 mid 是最优选择

| 配置 | 命令 |
|------|------|
| early（层 0~9） | `--fi_layers early` |
| mid（层 10~19） | `--fi_layers mid`（当前） |
| late（层 20~29） | `--fi_layers late` |

### 🟢 优先级 4：异常样本 31 分析

样本 31 是唯一退步的（XCLIP -15.3%），水下城市+鲸鱼场景。需要：
- 对比 baseline 样本 31 的 XCLIP
- 检查注入统计日志中该样本的 gate 分布
- 确认是 FI 干扰还是场景本身困难

### 🟢 优先级 5：统一门控框架（代码改动）

将 quality_scale 和 adaptive_gate 统一为置信度加权注入框架，做代码重构 + 实验验证。

### 🔵 优先级 6：多分镜长视频案例

选一个 30 秒+ 的长视频，做场景切分 → 每个分镜独立 FI 生成 → 跨分镜一致性。这是与 P-Flow 原论文最大的差异化贡献。

### 🔵 优先级 7：与基线方法对比

| 方法 | 难度 | 说明 |
|------|------|------|
| SVDInit（直接用 η_inv 做 $z_T$） | 低 | 30 分钟 |
| 纯 inversion（无 SVD 滤波） | 低 | 30 分钟 |
| ControlNet | 高 | 需要 Wan2.1 的 ControlNet |

---

## 附录：版本演进历史

| 版本 | 核心改动 | 状态 |
|------|---------|------|
| v1.0 | Baseline（caption → 一次生成） | ✅ 基础 |
| v2.0 | +Flow Matching Inversion + SVD 滤波 + Noise Blend | ✅ L2 完成 |
| v3.0 | +VLM 迭代优化 + Composite 三面板 | ✅ L1 迭代 |
| v4.0 | +LLM 结构化改写（V4 策略：subject-first + temporal chain + visual keyword） | ⚠️ 已被 v9 取代 |
| v9.0-vlm | +LLM 纯删减 + VLM 视觉补充（不碰运动） | ✅ 当前 L1 策略 |
| v5.0 | +Midpoint ODE Solver | ✅ 反演精度 |
| v6.0 | +Velocity Matching（Δe 优化） | ❌ 废弃：开销大收益小 |
| v7.0 | +Position-Aware + RF-Solver | ❌ 废弃：假设错误 |
| v8.0 | +Trajectory Anchor（position lerp） | ❌ 废弃：L2+L3 不兼容 |
| v9.0-vda | +VDA（速度方向锚定） | ⚠️ 部分有效：angle>90° 限制 |
| **v10.0** | **+Feature Injection + 自适应门控** | **✅ 当前 SOTA：XCLIP +13.6%** |

---

## 附录：计算开销

| 配置 | DiT Forwards | 相对 Baseline | 单样本耗时 (A800) |
|------|-------------|--------------|------------------|
| Baseline | ~30 | 1.0× | ~30s |
| +L2 (inversion+SVD+blend) | ~80 | 2.7× | ~80s |
| +L2+L3 (FI, 含特征缓存) | ~110 | 3.7× | ~110s |
| +L1+L2+L3 (含 VLM 改写) | ~110 | 3.7× | ~120s |

---

*最后更新：2026-06-16 | 基于 P-Flow v10.0 (Feature Injection) | L1 策略更新为 v9-vlm*
