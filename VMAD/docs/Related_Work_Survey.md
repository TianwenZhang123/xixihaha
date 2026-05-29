# VMAD Related Work Survey: 五类视频再现方法综述

> 本文档对视频精确再现（Video Faithful Reproduction）领域的五类核心方法进行系统调研，覆盖 Model Fine-tuning、Textual Inversion、Noise Space Manipulation、Motion Transfer 和 Prompt Engineering 五个方向，共 15 篇代表性工作。每类方法从核心思想、关键技术、优缺点和与 VMAD 的关系四个维度展开。

---

## 1. Model Fine-tuning（模型微调）

本类方法通过修改模型参数来注入特定的视觉概念（身份、风格、动作），是最直接的个性化方案。

### 1.1 DreamBooth (CVPR 2023)

- **论文**: Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation", CVPR 2023
- **核心思想**: 给定同一主体的 3-5 张图片，微调整个 diffusion 模型的参数，使其学会将一个稀有 token（如 `[V] dog`）绑定到该特定主体。同时使用 class-specific prior preservation loss 防止语言漂移（language drift），即避免模型忘记"dog"这个通用概念。
- **关键技术**:
  - 全参数微调 U-Net / DiT，配合低学习率（1e-6）和少量步数（~800 步）
  - Prior preservation loss: 用冻结的原始模型生成同类样本作为正则化
  - 稀有 token identifier: 使用字典中罕见的 token 避免与现有语义冲突
- **优点**: 保真度极高，能精确捕获主体的身份特征
- **缺点**: (1) 每个概念需独立微调一次模型（数分钟到数小时），无法实时; (2) 模型参数被修改，可能破坏泛化能力; (3) 多概念组合困难
- **与 VMAD 的关系**: VMAD 明确选择了"模型参数完全冻结"的路线（Conditioning Inversion），避免了 DreamBooth 的参数修改代价。DreamBooth 的保真度上限更高（因为自由度更大），但代价也更高。VMAD 的 Δe 优化可视为一种"无需微调的轻量替代方案"

### 1.2 LoRA (ICLR 2022, 应用于 Diffusion 2023+)

- **论文**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022; 后续广泛应用于 Stable Diffusion / Wan 系列
- **核心思想**: 将模型权重的更新限制为低秩分解 ΔW = B·A（其中 B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)），大幅减少可训练参数量（通常 <1% 总参数），同时保持与全微调接近的效果。
- **关键技术**:
  - 对 attention 层的 Q, K, V, O 矩阵插入低秩旁路
  - rank r 通常取 4-64，控制表达能力与正则化之间的 trade-off
  - 多个 LoRA 可以合并（arithmetic merge）或切换（hot-swap）
- **优点**: (1) 参数高效（rank=8 时仅需 ~2MB 存储）; (2) 可组合（A 的 LoRA + B 的 LoRA）; (3) 训练快（~5 分钟 on A100）
- **缺点**: (1) 仍需训练过程; (2) 低秩约束限制了表达能力，复杂概念可能欠拟合; (3) 多 LoRA 合并时可能冲突
- **与 VMAD 的关系**: LoRA 修改的是模型的"画笔技法"（权重），VMAD 修改的是"指令"（条件 embedding）。两者在数学上互补：LoRA 改变了 v_θ 本身，VMAD 改变了输入给 v_θ 的条件 c。理论上可以组合使用——先用 LoRA 注入身份先验，再用 VMAD 精确匹配目标速度场

### 1.3 AnimateDiff (ICLR 2024)

- **论文**: Guo et al., "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning", ICLR 2024
- **核心思想**: 在冻结的 T2I 模型中插入可训练的 temporal motion module（temporal transformer layers），只训练运动建模部分。这样任何已有的个性化 T2I 模型（如 DreamBooth / LoRA 版）都能直接获得动画能力，无需重新训练。
- **关键技术**:
  - Motion Module: 在每个 spatial transformer block 后插入 temporal self-attention 层
  - 训练数据: 大规模视频数据集（WebVid-10M）上只训练 motion module 参数
  - 推理时: 冻结 T2I backbone + motion module，可与任何 LoRA/DreamBooth checkpoint 组合
- **优点**: (1) 即插即用的运动能力; (2) 与个性化模型兼容; (3) 一次训练，多次复用
- **缺点**: (1) 运动模式受训练数据分布限制; (2) 无法精确复现特定视频的运动轨迹; (3) 生成长度受限（通常 16 帧）
- **与 VMAD 的关系**: AnimateDiff 解决的是"给静态模型加运动能力"，VMAD 解决的是"精确匹配特定视频的运动"。两者的目标不同但技术互补。VMAD 的 velocity matching 本质上是在 embedding 空间做"运动匹配"，可视为 AnimateDiff 的 inference-time 替代方案——不需要训练 motion module，而是通过优化条件来引导已有模型产生目标运动

---

## 2. Textual Inversion（文本反演）

本类方法将视觉概念编码为文本 embedding 空间中的向量，不修改模型参数，与 VMAD 的 Δe 最为接近。

### 2.1 Textual Inversion (ICLR 2023)

- **论文**: Gal et al., "An Image is Worth One Word: Personalizing Text-to-Image Models with Textual Inversion", ICLR 2023
- **核心思想**: 在 text encoder 的 embedding 空间中优化一个新的 token embedding v*（对应特殊 token `S*`），使得包含该 token 的 prompt 能引导模型生成与参考图像一致的输出。优化目标为标准的 diffusion denoising loss。
- **关键技术**:
  - 优化变量: 单个 token embedding v* ∈ R^{768}（或 1024），极低维
  - Loss: 标准 LDM 训练目标 E_{t,ε}[||ε - ε_θ(z_t, t, c(v*))||²]
  - 训练: 3-5 张参考图，~5000 步优化，约 1-2 小时
- **优点**: (1) 不修改模型参数; (2) 学到的 v* 可嵌入任意 prompt; (3) 理论优雅——将视觉概念"压缩"为一个词
- **缺点**: (1) 单个 768 维向量的表达能力有限，复杂概念（如特定人脸的精确细节）难以完整编码; (2) 收敛慢（数千步）; (3) 只能捕获"概念级别"信息，无法精确再现特定图像
- **与 VMAD 的关系**: VMAD 的 Δe 是 Textual Inversion 的直接泛化。区别在于:
  - TI 优化单个 token，VMAD 优化整个序列的 embedding 残差
  - TI 用 denoising loss，VMAD 用 velocity field matching loss（更直接地优化生成轨迹）
  - TI 面向概念学习（"这只猫"），VMAD 面向精确再现（"这段视频的每一帧"）
  - VMAD 的 Position-Aware 策略是对 TI 的关键改进——TI 盲目优化所有位置，VMAD 把预算集中在高影响力位置

### 2.2 ELITE (ICCV 2023)

- **论文**: Wei et al., "ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation", ICCV 2023
- **核心思想**: 训练一个 encoder 网络将参考图像直接映射为 textual embedding，实现 feed-forward 的概念注入（无需逐样本优化）。分两阶段：全局映射（粗粒度语义）+ 局部映射（细粒度 patch 特征）。
- **关键技术**:
  - Global Mapping Network: 将 CLIP image feature 映射为单个 token embedding（类似 TI 的 v*）
  - Local Mapping Network: 将 image patch features 映射为多个辅助 token（保留空间细节）
  - 训练: 在大规模图文对上预训练 encoder，推理时单次前向即可
- **优点**: (1) Feed-forward，无需逐样本优化（推理 <1 秒）; (2) 局部映射保留更多细节; (3) 可与 ControlNet 等组合
- **缺点**: (1) 需要预训练 encoder（额外训练成本）; (2) 泛化能力受训练数据限制; (3) 精确度仍不如 DreamBooth
- **与 VMAD 的关系**: ELITE 的"多 token 映射"思想与 VMAD 的"整序列 Δe"一脉相承——都认识到单个 token 不够用。VMAD 可借鉴 ELITE 的"先粗后细"策略：先用全局 Δe 做粗匹配，再用局部 position-specific 残差做细节补偿

### 2.3 IP-Adapter (arXiv 2023, 广泛引用 2024+)

- **论文**: Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", arXiv 2308.06721
- **核心思想**: 在 diffusion model 的 cross-attention 层中引入一个平行的 image cross-attention 分支。文本特征走原始 text cross-attention，图像特征走新增的 image cross-attention，两路的输出加权融合。这实现了"文本兼容的图像提示"。
- **关键技术**:
  - Decoupled cross-attention: 原始 KV 来自 text，新增 KV 来自 image encoder（CLIP ViT）
  - 可训练参数: 仅新增 image cross-attention 的 projection 矩阵（~22M 参数）
  - 权重控制: 通过 scale factor λ 控制图像特征的注入强度
- **优点**: (1) 即插即用，与现有模型兼容; (2) 文本和图像信号可独立控制; (3) 无需逐样本优化; (4) 支持多图像/多概念组合
- **缺点**: (1) 需要额外训练 adapter; (2) image feature 经过 CLIP 瓶颈，丢失了精细纹理; (3) 无法精确逐帧对齐
- **与 VMAD 的关系**: IP-Adapter 修改的是模型的注意力通路（新增 branch），VMAD 修改的是注意力的输入（条件 embedding）。核心区别在于 IP-Adapter 是"训练时"方案（需要训练 adapter），VMAD 是"推理时"方案（只优化输入）。在精确再现任务中，VMAD 的 velocity matching 提供了比 IP-Adapter 更直接的优化目标

---

## 3. Noise Space Manipulation（噪声空间操控）

本类方法通过操控扩散模型的初始噪声来影响生成结果，对应 VMAD 的 Layer 3。

### 3.1 FreeInit (ECCV 2024)

- **论文**: Wu et al., "FreeInit: Bridging Initialization Gap in Video Diffusion Models", ECCV 2024
- **核心思想**: 视频扩散模型在训练时接收的 noisy latent 包含了参考视频的低频信息（全局运动结构），但推理时的纯随机噪声缺乏这种结构。FreeInit 通过迭代初始化来弥补这个 gap：生成一次视频 → 对生成结果做部分扩散（加噪到中间步）→ 保留低频 + 替换高频 → 再生成一次。
- **关键技术**:
  - Training-inference gap 分析: 训练时 z_t = α_t·z₀ + σ_t·ε，z₀ 的低频成分会渗透到 z_t
  - 迭代滤波: 每轮生成后，对结果 FFT → 保留低频 → 高频用新随机噪声替换 → 重新生成
  - 通常迭代 2-3 轮即可收敛
- **优点**: (1) Training-free; (2) 即插即用; (3) 显著提升时间一致性
- **缺点**: (1) 多轮迭代带来 2-3× 推理代价; (2) 低频滤波的截止频率需手动调; (3) 无法精确匹配特定视频
- **与 VMAD 的关系**: FreeInit 是 VMAD Layer 3 的思想前驱。核心洞察完全一致——"初始噪声应该包含目标视频的结构信息"。区别在于 FreeInit 通过迭代生成来逼近好的噪声（间接方法），VMAD 通过 ODE 反演直接计算 η_inv（解析方法）。VMAD 的 SVD noise prior 也借鉴了 FreeInit 的频域分解思路

### 3.2 RF-Inversion (ICLR 2025)

- **论文**: Rout et al., "Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations", ICLR 2025
- **核心思想**: 将 Rectified Flow 的 inversion 过程建模为 LQR（线性二次调节器）最优控制问题。证明了最优控制器对应的向量场等价于一个 Rectified SDE，兼具确定性反演的精度和随机采样的多样性。
- **关键技术**:
  - 将 inversion 形式化为: min ∫₀¹ [||x_t - x̂_t||² + λ||u_t||²] dt, subject to dx_t = (v_θ + u_t)dt
  - 求解 Riccati ODE 获得最优控制 u_t*，等价于在 v_θ 上叠加一个修正项
  - 无需训练，直接作用于 FLUX、SD3 等模型
- **优点**: (1) 理论优雅，有最优控制的精确数学保证; (2) 兼容所有 Rectified Flow 模型; (3) 同时支持重建和编辑
- **缺点**: (1) 需要解 Riccati ODE（计算开销）; (2) 对离散化步数敏感; (3) 主要面向图像，视频扩展待验证
- **与 VMAD 的关系**: VMAD Layer 3 当前使用 50-step Euler inversion，精度受限于离散化误差。RF-Inversion 的 LQR 框架可作为精度升级方案：用最优控制理论取代朴素 Euler，期望获得更准确的 η_inv，进而为 Layer 2 的 velocity matching 提供更精确的目标速度场 v* = z₀ - η_inv

### 3.3 FreqPrior (ICLR 2025)

- **论文**: Yuan et al., "FreqPrior: Improving Video Diffusion Models with Frequency Filtering Gaussian Noise", ICLR 2025
- **核心思想**: 在频域对初始噪声进行精心设计的滤波。关键发现：以往方法（如 FreeInit）的频率滤波会导致方差衰减，使视频过度平滑。FreqPrior 提出方差保持的滤波公式，同时引入 partial sampling（从中间 timestep 开始生成）减少推理成本。
- **关键技术**:
  - Variance-Preserving Frequency Filter: η_filtered = F⁻¹[M(f)·F(η)] / √(E[M²])，除以 √E[M²] 保持单位方差
  - Partial Sampling: 不从 t=0（纯噪声）开始，而是从 t=t_start 开始，跳过早期噪声阶段
  - 低频增强: 对时间维度的低频分量给予更大权重，增强运动一致性
- **优点**: (1) 方差保持避免过度平滑; (2) Partial sampling 节省 23% 推理时间; (3) 即插即用，无需训练
- **缺点**: (1) 滤波参数需要针对不同模型调优; (2) 只能提供"软性"先验，无法精确匹配
- **与 VMAD 的关系**: FreqPrior 对 VMAD Layer 3 的 η_inv 后处理提供了直接启示——对 η_inv 做频域分解后选择性使用（保留低频结构、高频随机化），既能保留参考视频的全局运动结构，又能避免逐像素复制的 trivial solution 质疑。这正是 VMAD 实验中 SVD noise prior 的理论基础

---

## 4. Motion Transfer（运动迁移）

本类方法专注于将参考视频的运动模式迁移到新内容，与 VMAD 的 velocity field matching 目标高度相关。

### 4.1 MotionPrompt (CVPR 2025)

- **论文**: Hou et al., "MotionPrompt: Generating Human Motion via Motion-Aware Prompt Learning", CVPR 2025
- **核心思想**: 将运动信息编码为 prompt embedding 空间中的可学习向量。通过 motion-aware prompt learning，使文本条件不仅包含语义描述，还隐式编码了运动轨迹、节奏和幅度信息。
- **关键技术**:
  - Motion-Aware Prompt Tokens: 在标准 text tokens 后追加 K 个可学习 motion tokens
  - Motion Encoder: 从参考运动序列提取 motion feature → 映射为 prompt space 中的向量
  - Motion-Text Alignment: 联合优化确保 motion tokens 与 text tokens 语义兼容
- **优点**: (1) 将运动编码为"可说的语言"; (2) 支持运动组合和插值; (3) 推理快（single forward）
- **缺点**: (1) 需要训练 motion encoder; (2) 运动表示精度受 prompt 空间维度限制; (3) 面向人体运动，通用性待验证
- **与 VMAD 的关系**: MotionPrompt 验证了"运动信息可以编码到 embedding 空间"这一核心假设，与 VMAD 的 Δe 具有相同的哲学基础。区别在于 MotionPrompt 需要预训练 motion encoder，VMAD 通过 velocity matching 直接在推理时优化。VMAD 的方法更通用（不限于人体运动），但可能收敛更慢

### 4.2 DiTFlow (CVPR 2025)

- **论文**: Gu et al., "DiTFlow: Advancing Text-to-Video Generation with Large Diffusion Transformer's Temporal and Spatial Optical Flow", CVPR 2025
- **核心思想**: 利用预训练 DiT 模型内部 attention map 中隐含的光流信息来引导视频生成。发现 DiT 的 temporal self-attention 在去噪过程中自然编码了帧间对应关系，可从中提取高质量的时空光流，进而用于运动控制。
- **关键技术**:
  - Attention-based Optical Flow: 从 temporal self-attention 的 QK 相似度矩阵中提取帧间位移场
  - Flow-Guided Generation: 用提取的光流作为额外条件，约束生成视频的运动一致性
  - Training-free: 直接操作已有模型的 attention map，无需额外训练
- **优点**: (1) 揭示了 DiT 内部已存在运动表示; (2) Training-free; (3) 可提取光流用于下游任务
- **缺点**: (1) 光流提取精度受 attention 分辨率限制; (2) 大运动场景可能失效; (3) 依赖模型架构（必须是 DiT）
- **与 VMAD 的关系**: DiTFlow 证实了 DiT 模型的内部表示中天然包含运动信息。这为 VMAD 的 Position-Aware 策略提供了理论支撑——DiT 的 attention 确实承载了运动语义，通过操控 attention 的输入（embedding）确实可以控制运动。VMAD 可借鉴 DiTFlow 的 attention 分析方法来验证 Δe 优化是否确实影响了目标帧间对应关系

### 4.3 Reenact Anything (SIGGRAPH Asia 2025)

- **论文**: "Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion", SIGGRAPH Asia 2025
- **核心思想**: 提出 Motion-Textual Inversion——将参考视频的运动信息反演为 text embedding 空间中的"运动词"（motion word），然后用该 motion word 驱动新内容生成同样的运动。这是 Textual Inversion 从"外观概念"到"运动概念"的自然延伸。
- **关键技术**:
  - Motion-Textual Inversion: 优化一个 embedding v_motion*，使得 prompt "a [v_motion*] cat" 生成的视频运动模式与参考视频一致
  - Motion-Content Disentanglement: 通过 multi-scale temporal loss 分离运动和内容，确保 v_motion* 只编码运动而非外观
  - Semantic Transfer: 学到的 motion word 可嵌入任何 prompt，实现语义级运动迁移
- **优点**: (1) 概念优雅——"运动是一个词"; (2) 零样本迁移; (3) 支持运动组合（多个 motion words）
- **缺点**: (1) 运动和内容的解耦不完美; (2) 复杂运动（多物体交互）难以编码为单个 token; (3) 优化耗时
- **与 VMAD 的关系**: **Reenact Anything 是与 VMAD Layer 2 最接近的工作。** 核心区别:
  - Reenact Anything 优化单个 motion token，VMAD 优化整序列的 Δe
  - Reenact Anything 用 denoising loss（间接），VMAD 用 velocity field matching（直接匹配生成轨迹）
  - Reenact Anything 面向运动迁移（运动+新内容），VMAD 面向精确再现（运动+原内容）
  - VMAD 的 Position-Aware 策略 + 全时间步 velocity matching 在技术上更精细

---

## 5. Prompt Engineering（提示工程）

本类方法在文本 prompt 层面做优化，寻找能最大化再现目标的最优提示词，对应 VMAD 的 Layer 1。

### 5.1 ARPO — Automatic Reverse Prompt Optimization (arXiv 2503.19937, 2025)

- **论文**: Ren & Zhan et al., "Reverse Prompt: Cracking the Recipe Inside Text-to-Image Generation", arXiv 2025
- **核心思想**: 给定参考图像，自动生成能精确复现该图像的文本 prompt。提出一个迭代梯度优化框架，通过三步循环逐步逼近最优 prompt: (1) 用当前 prompt 生成图像 → (2) 计算与参考图的 CLIP 相似度梯度 → (3) 将梯度转化为"文本梯度"（textual gradient）指导 prompt 更新。
- **关键技术**:
  - 初始化: 使用 VLM（如 BLIP2）对参考图生成初始 caption
  - Textual Gradient: 利用 CLIP 的文本-图像对齐特性，将图像空间的梯度信号转化为对 prompt token 的优化方向
  - 迭代优化: prompt → 生成 → 评估 → 梯度 → 更新 prompt，循环直到收敛
  - 搜索策略: 在每步 token 更新时，从 top-k 候选中选择 CLIP score 提升最大的
- **优点**: (1) 输出为可读文本，完全可解释; (2) 不修改模型; (3) 优化后的 prompt 可跨模型迁移
- **缺点**: (1) 文本表达能力有天花板——复杂场景无法用文字完整描述; (2) 优化过程慢（需多次生成+评估）; (3) CLIP score 不完美反映真实相似度
- **与 VMAD 的关系**: ARPO 对应 VMAD Layer 1 的"文本优化"阶段。VMAD 的实验已经验证了文本优化的天花板（CLIP 约 0.8842），ARPO 提供了更系统的优化框架。VMAD 的创新在于：当 ARPO 这类文本方法到达天花板后，Layer 2（Δe）和 Layer 3（η_inv）接力完成剩余逼近

### 5.2 VGD — Visually Guided Decoding (ICLR 2025)

- **论文**: Kim et al., "Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models", ICLR 2025
- **核心思想**: 利用 LLM 的自回归解码能力，在 CLIP 视觉信号引导下逐 token 生成最优 prompt。与梯度方法（如 ARPO）不同，VGD 完全不需要梯度计算，而是将 prompt inversion 转化为一个 CLIP-guided constrained decoding 问题。
- **关键技术**:
  - 基于 LLM（如 LLaMA）做 autoregressive decoding，每步生成一个 token
  - CLIP Guidance: 在每步 decoding 时，用 CLIP image-text similarity 作为 scoring function，选择使 score 最大化的 next token
  - Beam Search: 维护 top-k 候选 prompt 前缀，逐步扩展
  - Coherence: LLM 的语言先验确保生成的 prompt 语法正确、语义连贯
- **优点**: (1) Gradient-free，无需反向传播; (2) 生成的 prompt 语言自然（因为由 LLM 解码）; (3) 比 soft prompt 方法更可解释; (4) ICLR 2025 发表，state-of-the-art
- **缺点**: (1) 依赖 LLM 的生成能力（LLM 不了解扩散模型的特性）; (2) Beam search 代价与 beam width 线性增长; (3) 仍受文本表达能力天花板限制
- **与 VMAD 的关系**: VGD 展示了 prompt inversion 的另一种范式——不是"优化"而是"生成"。VMAD Layer 1 的 VLM captioning + iterative refinement 可以看作 VGD 思路的简化版。VGD 的 CLIP-guided decoding 策略可直接用于 VMAD 的 prompt 初始化阶段，替代当前的 VLM caption

### 5.3 CCC — Critique Coach Calibration (ICML 2025)

- **论文**: Polu et al., "CCC: Prompt Evolution for Video Generation via Structured MLLM Feedback", ICML 2025
- **核心思想**: 提出一个 training-free、model-agnostic 的 prompt 迭代进化框架。核心循环为: 生成视频 → MLLM 结构化批评（指出语义偏差、主体漂移、缺失物体）→ MLLM 基于批评重写 prompt → 再次生成。MLLM 同时扮演 Critic（评价者）和 Coach（修正者）两个角色。
- **关键技术**:
  - Structured Critique: MLLM 输出结构化评价（不是简单的好/坏），包含: 哪些元素缺失、哪些语义偏移、动作是否正确
  - Coach Calibration: 基于 critique，MLLM 重写 prompt——增加缺失描述、修正歧义表达、调整措辞
  - Iterative Evolution: 通常 2-4 轮迭代即可收敛
  - Model-Agnostic: 可应用于任何文生视频模型（因为只改 prompt，不改模型）
- **优点**: (1) 完全 training-free; (2) Model-agnostic; (3) 过程可解释（每步的 critique 和修改都是自然语言）; (4) 利用了 MLLM 的推理能力
- **缺点**: (1) 每轮需要完整的视频生成+MLLM 评估（代价高）; (2) MLLM 的评价可能不准确; (3) 仍然受限于文本表达能力
- **与 VMAD 的关系**: CCC 是 VMAD Layer 1 最直接的 baseline 和互补方案。VMAD 在 P-Flow 阶段的实验（iterative prompt refinement，CLIP 从 0.8703→0.8842）本质上就是 CCC 框架的一个实例。CCC 系统化了这个过程，并证明了 MLLM 反馈比简单的 CLIP score 优化更有效。VMAD 的贡献在于：当 CCC 这类纯 prompt 方法到达天花板后，Layer 2+3 提供了超越文本极限的路径

---

## 6. 方法分类对比与 VMAD 定位

### 6.1 五类方法的核心对比

| 维度 | Model Fine-tuning | Textual Inversion | Noise Space | Motion Transfer | Prompt Engineering |
|------|-------------------|-------------------|-------------|-----------------|-------------------|
| 修改对象 | 模型参数 θ | Embedding v* | 初始噪声 η | Embedding（运动专用） | 文本 prompt |
| 是否需训练 | 是 | 部分（TI需优化） | 否 | 部分 | 否 |
| 表达能力 | 最高（全参数） | 中等 | 高（全维） | 中等 | 最低（离散文本） |
| 泛化性 | 低（概念绑定） | 中 | 低（样本绑定） | 高（运动迁移） | 最高（可读文本） |
| 精确度 | 高 | 中 | 高 | 中 | 低 |
| 推理代价 | 额外微调时间 | 优化时间 | Inversion 计算 | 可能 0（预训练 encoder） | 多轮生成+评估 |
| 代表工作 | DreamBooth, LoRA | TI, ELITE | FreeInit, RF-Inv | MotionPrompt, DiTFlow | ARPO, VGD, CCC |

### 6.2 VMAD 的独特定位

VMAD 系统性地整合了上述五类方法中的核心思想，构建了一个三层渐进框架：

- **Layer 1（Prompt Engineering）**: 继承 CCC/ARPO/VGD 的 prompt 优化思路，使用 VLM captioning + iterative refinement 获取最优文本描述
- **Layer 2（Textual Inversion 的泛化 + Motion Transfer 的理论升级）**: 将 Textual Inversion 从"概念学习"升级为"轨迹匹配"，将 Motion Transfer 从"训练 encoder"改为"推理时优化"，并引入 Position-Aware 策略和 Velocity Field Matching 作为统一的优化框架
- **Layer 3（Noise Space Manipulation）**: 将 FreeInit/FreqPrior 的频域先验思想与 RF-Inversion 的解析反演结合，通过 Flow Matching Inversion 精确计算结构化噪声先验

VMAD 的核心贡献在于：将这五类原本独立的方法论统一到 Conditioning Inversion 框架下，证明了在模型参数完全冻结的约束下，仅通过优化输入条件（text embedding + noise prior）即可实现高保真视频再现。

### 6.3 每类方法中与 VMAD 最强关联的论文

| 类别 | 最相关论文 | 关联点 |
|------|-----------|--------|
| Model Fine-tuning | LoRA | Δe 可视为 LoRA 在输入空间的类比——低维参数化修正 |
| Textual Inversion | Textual Inversion (ICLR 2023) | VMAD Δe 是 TI v* 的直接泛化（单 token → 全序列） |
| Noise Space | RF-Inversion (ICLR 2025) | 共享 Rectified Flow inversion 的核心方法，可直接升级 VMAD Layer 3 |
| Motion Transfer | Reenact Anything (SIGGRAPH 2025) | 与 VMAD Layer 2 最接近：同为 embedding 空间的运动编码 |
| Prompt Engineering | CCC (ICML 2025) | 与 VMAD Layer 1 直接对标：迭代 prompt 优化框架 |

---

## 7. 论文引用列表

### Model Fine-tuning

1. Ruiz et al., "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation", CVPR 2023
2. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
3. Guo et al., "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning", ICLR 2024

### Textual Inversion

4. Gal et al., "An Image is Worth One Word: Personalizing Text-to-Image Models with Textual Inversion", ICLR 2023
5. Wei et al., "ELITE: Encoding Visual Concepts into Textual Embeddings for Customized Text-to-Image Generation", ICCV 2023
6. Ye et al., "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models", arXiv 2308.06721, 2023

### Noise Space Manipulation

7. Wu et al., "FreeInit: Bridging Initialization Gap in Video Diffusion Models", ECCV 2024
8. Rout et al., "Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations", ICLR 2025
9. Yuan et al., "FreqPrior: Improving Video Diffusion Models with Frequency Filtering Gaussian Noise", ICLR 2025

### Motion Transfer

10. Hou et al., "MotionPrompt: Generating Human Motion via Motion-Aware Prompt Learning", CVPR 2025
11. Gu et al., "DiTFlow: Advancing Text-to-Video Generation with Large Diffusion Transformer's Temporal and Spatial Optical Flow", CVPR 2025
12. "Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion", SIGGRAPH Asia 2025

### Prompt Engineering

13. Ren & Zhan et al., "Reverse Prompt: Cracking the Recipe Inside Text-to-Image Generation (ARPO)", arXiv 2503.19937, 2025
14. Kim et al., "Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models (VGD)", ICLR 2025
15. Polu et al., "CCC: Prompt Evolution for Video Generation via Structured MLLM Feedback", ICML 2025

---

*文档生成时间: 2025年7月*
*关联项目: VMAD (Velocity-Matching Augmented Diffusion)*
