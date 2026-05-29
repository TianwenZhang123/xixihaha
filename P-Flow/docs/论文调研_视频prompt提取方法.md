# 论文调研：从目标视频提取生成引导信号的学术方法

> 调研目的：寻找比单纯 VLM captioning 更学术化的方法，从目标视频中提取初始 prompt/embedding，用于指导 T2V 生成。
> 调研时间：2025.05.28

---

## 一、方法分类体系

按技术路线分为四大类：

---

## 第一类：噪声空间反演（Noise-Space Inversion）

核心思想：将目标视频通过 ODE/SDE 反演回噪声空间，得到的噪声本身就编码了视频的结构和运动信息。

### 1. RF-Solver / RF-Edit (ICML 2025)

- **论文**：*Taming Rectified Flow for Inversion and Editing*
- **作者**：Wang et al.
- **链接**：https://github.com/wangjiangshan0725/RF-Solver-Edit
- **方法**：针对 Rectified Flow 模型提出高阶 ODE solver，显著降低反演-重建误差。用 Taylor 展开的高阶项修正 Euler 积分误差，使得 video → noise → video 的重建几乎无损。
- **核心贡献**：
  - 提出 RF-Solver 作为通用 sampler，可直接应用于 FLUX、OpenSora 等模型
  - 不仅提升采样质量，还显著提升 Inversion-Reconstruction 准确度
  - 基于 RF-Solver 提出 RF-Edit，实现高质量图像和视频编辑
- **与我们的关系**：**直接适用**。我们的 pipeline 已经在做 flow matching inversion（VAE encode → Euler ODE t=1→t=0），RF-Solver 提供了理论上更精确的反演方案，可以作为方法的理论基础。我们当前的 Euler inversion 是一阶近似，RF-Solver 的高阶修正可以作为改进方向。

---

### 2. RF-Inversion (ICLR 2025)

- **论文**：*Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations*
- **作者**：Rout et al.
- **链接**：https://rf-inversion.github.io/ | https://github.com/LituRout/RF-Inversion
- **方法**：将 rectified flow 的反演问题建模为**线性二次调节器（LQR）最优控制问题**，推导出等价的随机微分方程。
- **核心贡献**：
  - 证明了在 stochastic 设定下，反演得到的噪声具有更好的语义编辑性质
  - 提出动态最优控制框架，使反演过程可控
  - 已集成入 diffusers 库
- **与我们的关系**：提供了一个非常优雅的数学框架——把"从视频提取噪声先验"包装成最优控制问题。数学上比简单的 Euler ODE 反演更有深度。

---

### 3. Edit-Friendly DDPM Noise Space (CVPR 2024)

- **论文**：*An Edit Friendly DDPM Noise Space: Inversion and Manipulations*
- **作者**：Huberman-Spiegelglas et al.
- **链接**：https://github.com/AHMEDHAMID123/edit_firendly_ddpm_inversion
- **方法**：提出替代性的噪声空间，使得反演得到的噪声图具有更高方差、更利于编辑。
- **核心贡献**：
  - 证明标准 DDIM inversion 得到的噪声缺乏结构
  - 提出的 edit-friendly noise 保留了图像的空间布局信息
  - 无需优化过程，可快速实现文本引导的编辑
- **与我们的关系**：理论支撑——说明不同的噪声空间具有不同的"信息编码"性质。我们的 SVD 滤波本质上就是在噪声空间中做信息选择，这篇论文为"噪声空间具有结构"提供了理论依据。

---

### 4. FreeInit (ECCV 2024)

- **论文**：*FreeInit: Bridging Initialization Gap in Video Diffusion Models*
- **作者**：Wu et al.
- **链接**：https://tianxingwu.github.io/pages/FreeInit/ | https://github.com/TianxingWu/FreeInit
- **方法**：发现视频扩散模型存在训练-推理的初始化 gap，低频噪声分量主要编码运动信息。通过迭代式噪声重初始化（保留低频、替换高频），显著提升时序一致性。
- **核心贡献**：
  - 发现 training 阶段的噪声低频分量包含前一步去噪结果的运动信息
  - 提出 iterative refinement：生成 → 反演 → 保留低频 + 替换高频 → 重新生成
  - 无需额外训练，无 learnable parameters，即插即用
- **与我们的关系**：**高度相关**。我们的 SVD 时域滤波（ρ_m=0.9 保留低频运动）与 FreeInit 的核心发现完全一致。FreeInit 证明了"低频噪声编码运动"这一假设的正确性，可以互相引用作为理论支撑。我们的方法可以看作 FreeInit 思想在 flow matching + 目标视频引导场景下的推广。

---

## 第二类：Embedding 空间优化（Prompt Embedding Optimization）

核心思想：不直接用文本，而是在 text embedding 空间中通过梯度优化找到最能"描述"目标视频的 embedding 向量。

### 5. MotionPrompt (CVPR 2025)

- **论文**：*Optical-Flow Guided Prompt Optimization for Coherent Video Generation*
- **作者**：Nam et al. (KAIST)
- **链接**：https://motionprompt.github.io/ | https://github.com/HyelinNAM/MotionPrompt
- **方法**：训练一个光流判别器区分真实/生成视频的运动模式，然后用判别器梯度反向优化 learnable token embeddings。
- **核心贡献**：
  - 在反向采样过程中，对 prompt embedding 做梯度下降
  - 使生成视频的光流分布逼近目标视频的光流分布
  - 不损害生成内容的保真度
- **优化目标**：min_e L_disc(OF(x_gen(e)), OF(x_real))，其中 OF 为光流提取，L_disc 为判别器 loss
- **与我们的关系**：**非常适合我们的场景**。可以用目标视频的光流作为监督信号，优化 prompt embedding 使其编码运动信息，而不是简单用 VLM 描述。这是把"运动信息"注入 prompt 的最直接方式。

---

### 6. Reenact Anything (SIGGRAPH 2025)

- **论文**：*Semantic Video Motion Transfer Using Motion-Textual Inversion*
- **作者**：Kansy et al. (ETH Zurich / Disney Research)
- **链接**：https://mkansy.github.io/reenact-anything/ | https://arxiv.org/abs/2408.00458
- **方法**：在 frozen I2V 模型上，优化 text/image embedding tokens 使其捕获参考视频的运动语义。
- **核心发现**：
  - I2V 模型中，latent image input 主要控制外观
  - Cross-attention 注入的 text/image embedding 主要控制运动
  - 因此可以通过优化 embedding 来"提取"视频的运动模式
- **优化目标**：min_e ||ε_θ(z_t, e) - ε||²，即让优化后的 embedding e 能重建参考视频的去噪方向
- **与我们的关系**：**最直接的学术化方案**。可以把"从目标视频提取 prompt"包装为 motion-textual inversion——优化一组 learnable tokens 使其在 T2V 模型的 cross-attention 中编码目标视频的运动模式。最终 prompt = VLM caption（内容）+ optimized motion tokens（运动）。

---

### 7. RichSpace (ICLR 2025)

- **论文**：*Enriching Text-to-Video Prompt Space via Text Embedding Interpolation*
- **作者**：Cao et al.
- **链接**：https://arxiv.org/abs/2501.09982
- **方法**：在 T2V 模型的 text embedding 空间中做插值，发现 embedding 空间比离散 text 空间更丰富。
- **核心贡献**：
  - 证明了 text embedding 空间的连续性和可操作性
  - 离散文本无法表达的细微语义差异，在 embedding 空间中可以通过插值实现
  - 提出两阶段方法：text → embedding space → interpolation → optimal embedding
- **与我们的关系**：理论支撑——说明为什么要在 embedding 空间而非 text 空间工作。离散文本的表达能力有限，embedding 空间更适合编码复杂的视频语义。

---

## 第三类：Prompt Inversion（图像/视频 → 离散文本）

核心思想：从图像/视频反推出可读的离散 text prompt。

### 8. EDITOR (arxiv 2025)

- **论文**：*Effective and Interpretable Prompt Inversion for Text-to-Image Diffusion Models*
- **作者**：Li et al.
- **链接**：https://arxiv.org/abs/2506.03067
- **方法**：三阶段 pipeline：
  1. **初始化**：BLIP captioning 生成初始 prompt p₀
  2. **优化**：在 latent embedding space 中做梯度优化 refinement
  3. **解码**：embedding-to-text decoder 将优化后的 embedding 转回离散文本
- **核心贡献**：
  - 解决了 PEZ/PH2P 等方法生成不可读 prompt 的问题
  - 同时保持高 image similarity
  - 消融实验表明 captioning 初始化对性能至关重要
- **与我们的关系**：如果想保留"可读 prompt"的输出形式，EDITOR 的三阶段框架是最好的参考——先 VLM 初始化，再在 latent space 优化，最后解码回文本。这比单纯 VLM captioning 多了"优化"这一步，学术性显著提升。

---

### 9. PH2P (CVPR 2024)

- **论文**：*Prompting Hard or Hardly Prompting: Prompt Inversion for Text-to-Image Diffusion Models*
- **作者**：Mahajan et al. (UBC)
- **链接**：https://github.com/ubc-vision/Prompting-Hard-Hardly-Prompting
- **方法**：利用 diffusion loss 在不同 timestep sub-range 的特性差异，对 prompt tokens 做梯度优化，每步投影回最近的词表 token。
- **核心贡献**：
  - 发现不同 timestep 关注不同层次的信息（高 t 关注结构，低 t 关注细节）
  - 据此设计分段优化策略，减少优化过程的高方差问题
  - 生成的 prompt 可直接用于 text-to-image 重建
- **与我们的关系**：可以借鉴其 timestep-aware 优化策略。在 flow matching 中，不同的 t 值同样对应不同层次的信息。

---

## 第四类：视频级语义引导（Video-Level Semantic Conditioning）

核心思想：直接用参考视频作为条件信号注入生成模型。

### 10. Video-As-Prompt / VAP (ByteDance, 2025)

- **论文**：*Unified Semantic Control for Video Generation*
- **作者**：ByteDance
- **链接**：https://github.com/bytedance/Video-As-Prompt | https://arxiv.org/abs/2510.20888
- **方法**：将参考视频作为 semantic prompt，通过 Mixture-of-Transformers (MoT) expert 注入 frozen Video DiT。
- **核心贡献**：
  - 不做任何 inversion，直接把参考视频的 latent features 通过额外的 transformer expert 注入生成过程
  - 支持多种语义控制（动作、风格、场景）
  - Plug-and-play，不修改原始 DiT 参数
- **与我们的关系**：代表了另一个极端——完全不提取 prompt，而是直接用视频特征做条件。需要训练额外的 MoT 模块，成本较高，但学术性强。

---

### 11. Video-P2P (CVPR 2024)

- **论文**：*Video Editing with Cross-attention Control*
- **作者**：Liu et al.
- **链接**：https://github.com/JIA-Lab-research/Video-P2P
- **方法**：优化一个 shared unconditional embedding 实现视频反演，然后通过 cross-attention 控制实现编辑。
- **核心贡献**：
  - 用单一的 unconditional embedding 替代逐帧的 null-text inversion
  - 大幅降低内存开销（从 O(N×T) 降到 O(1)）
  - 实现了第一个真实世界视频编辑框架
- **与我们的关系**：shared embedding 的思想可以借鉴——用一个统一的 embedding 编码整段视频的语义，而非逐帧处理。

---

### 12. STEM Inversion (CVPR 2024)

- **论文**：*A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing*
- **作者**：Li et al.
- **链接**：https://github.com/STEM-Inv/STEM-Inv
- **方法**：用低秩表示（256 个 basis）对视频 inversion 过程建模，通过 EM 算法迭代优化。
- **核心贡献**：
  - 将视频帧的 latent 分解为少量 spatial-temporal bases 的线性组合
  - 实现高效且时序一致的反演
  - 比 DDIM inversion 更好地保持时序一致性
- **与我们的关系**：低秩分解的思想与我们的 SVD 方法异曲同工。STEM 用 EM 做低秩分解，我们用 SVD 做频谱分解，都是在降维空间中提取视频的核心信息。可以互相引用。

---

## 二、学术化包装方案

### 方案 A：Noise-Space Motion Embedding（推荐，与现有代码最契合）

我们已经实现了 flow matching inversion + SVD 滤波，学术化包装为：

1. **理论贡献**：证明 rectified flow 反演得到的噪声中，低频时域分量编码了运动先验（引用 FreeInit + Edit-Friendly Noise Space）
2. **方法贡献**：提出 SVD-based spectral decomposition 在噪声空间中分离 content/motion 信息（引用 STEM Inversion 的低秩思想）
3. **与 RF-Solver 的关系**：我们的 Euler inversion 是一阶近似，可以讨论高阶 solver 的改进空间

**优点**：与现有代码完全一致，只需补充理论分析和消融实验
**缺点**：novelty 可能不够，需要更强的实验结果支撑

---

### 方案 B：Motion-Textual Inversion（学术性更强，需要额外实现）

参考 Reenact Anything + MotionPrompt：

1. 在 T2V 模型的 text embedding 空间中引入 learnable motion tokens [M₁, M₂, ..., Mₖ]
2. 优化目标：使这些 tokens 在 cross-attention 中能重建目标视频的运动模式
3. 最终 prompt = VLM caption（描述内容）+ optimized motion tokens（编码运动）
4. 这样就把"VLM captioning"从唯一手段变成了"内容初始化"，运动信息由 learned embedding 补充

**优点**：novelty 强，有明确的优化目标和数学形式
**缺点**：需要额外实现 embedding 优化模块，需要 T2V 模型支持梯度回传

---

### 方案 C：Hybrid Multi-Modal Motion Prior（最完整的故事）

结合 A 和 B：

1. **Flow matching inversion → noise prior**（编码全局运动结构）
2. **Motion-textual inversion → embedding tokens**（编码细粒度运动语义）
3. **VLM captioning → text prompt**（编码场景内容）
4. **三者融合**：noise prior 控制初始化，motion tokens 通过 cross-attention 引导，text prompt 提供语义约束

**优点**：故事完整，三个层次各有分工，消融实验丰富
**缺点**：工作量最大，需要同时实现噪声先验和 embedding 优化

---

## 三、发论文的 Positioning 建议

如果选方案 C，论文 story 可以是：

> "现有 T2V 方法依赖纯文本 prompt 描述目标视频，但文本无法精确表达复杂运动模式。我们提出 **Multi-Modal Motion Prior**，从目标视频中同时提取三个层次的引导信号：
> (1) 噪声空间的运动先验（通过 flow matching inversion + spectral decomposition）
> (2) embedding 空间的运动 tokens（通过 motion-textual inversion）
> (3) 文本空间的场景描述（通过 VLM）
> 实验表明，这种多层次的 prompt 表示显著优于单一 VLM captioning。"

### 可能的论文标题：

- *Beyond Captioning: Multi-Modal Motion Prior for Video-Guided Video Generation*
- *Motion-Aware Prompt Inversion via Flow Matching for Text-to-Video Generation*
- *From Noise to Semantics: Hierarchical Motion Extraction for Video Generation*

---

## 四、关键引用关系图

```
我们的方法
├── 噪声空间反演
│   ├── FreeInit (ECCV 2024) — 低频噪声编码运动的理论基础
│   ├── RF-Solver (ICML 2025) — 高阶 flow matching inversion
│   ├── RF-Inversion (ICLR 2025) — LQR 最优控制框架
│   └── Edit-Friendly Noise (CVPR 2024) — 噪声空间具有结构的证据
├── Embedding 优化
│   ├── Reenact Anything (SIGGRAPH 2025) — motion-textual inversion
│   ├── MotionPrompt (CVPR 2025) — 光流引导的 embedding 优化
│   └── RichSpace (ICLR 2025) — embedding 空间的连续性
├── Prompt Inversion
│   ├── EDITOR (2025) — 三阶段 prompt inversion pipeline
│   └── PH2P (CVPR 2024) — timestep-aware 优化
└── 视频级引导
    ├── Video-As-Prompt (2025) — 参考视频作为 semantic prompt
    ├── Video-P2P (CVPR 2024) — shared unconditional embedding
    └── STEM Inversion (CVPR 2024) — 低秩时空分解
```

---

## 五、下一步行动建议

1. **短期（1-2周）**：把现有 noise prior 实验的理论包装做好，写出方案 A 的 method section
2. **中期（2-4周）**：实现 motion-textual inversion（方案 B），验证 embedding 优化是否能提升 XCLIP
3. **长期（1-2月）**：整合为方案 C 的完整 pipeline，跑全量实验，写论文
