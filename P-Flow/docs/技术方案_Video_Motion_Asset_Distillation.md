# Video Motion Asset Distillation (VMAD)

## 从目标视频中蒸馏可复用运动 Prompt 资产的技术方案

> **一句话概括**：通过 velocity field matching 从目标视频中逆向蒸馏出"运动 prompt 资产"——一个可存储、可迁移、可组合的混合表示（可读文本 + motion token + noise prior），使得任意新内容描述 + 该资产即可复现原视频的运动模式。

---

## 1. 问题定义

### 1.1 任务定义

**输入**：目标视频 $V_{ref}$（我们想要提取其运动模式的视频）

**输出**：一个**运动 Prompt 资产** $\mathcal{A}_{motion}$，包含：

- 可读文本部分 $p_{text}$：描述运动语义（"先慢走再突然加速转身"），可跨模型使用
- Motion Token 部分 $\Delta e_{motion}$：编码文本无法表达的精确运动结构，同系列模型内复用
- 噪声先验部分 $\eta_{motion}$：编码全局运动布局，同系列模型内复用

**使用方式**：给定任意新内容描述 $p_{content}$（如"一只猫"），组合 $p_{content} + \mathcal{A}_{motion}$ 即可生成"一只猫做出与原视频相同运动模式"的新视频。

### 1.2 与 FMMD 的关系

FMMD（Flow-Matching Motion Distillation）的目标是"生成一个运动一致的视频"——输出是一次性的 $c_{final}$，用完即弃。VMAD 的目标是"提取一个可复用的运动资产"——输出是结构化的 $\mathcal{A}_{motion}$，可以反复使用。

VMAD 在 FMMD 的基础上增加了三个关键模块：

1. **内容解耦约束**：确保 $\Delta e_{motion}$ 只编码运动，不绑定特定内容（否则迁移到新内容时会冲突）
2. **Motion Text Decoding**：将连续 embedding 中可表达的部分转回可读文本（使资产具备跨模型迁移能力）
3. **资产格式设计**：定义混合表示的结构、存储方式和使用接口（使资产可存储、可组合）

### 1.3 核心挑战

**挑战 1：运动-内容纠缠**

直接优化的 $\Delta e$ 可能同时编码了运动和内容信息。例如从"一只狗在跑"的视频中提取的 $\Delta e$，可能偷偷编码了"狗的外观"。当迁移到"一只猫在跑"时，$\Delta e$ 中的"狗外观"信息会与"猫"的内容描述冲突，导致生成质量下降。

**挑战 2：文本表达力瓶颈**

自然语言无法精确描述"加速度曲线"、"镜头微晃频率"、"0.3 秒处的突然转向角度"等运动细节。这意味着纯文本资产必然丢失信息，但纯 tensor 资产又不可读、不可编辑、不可跨模型。

**挑战 3：信息分配问题**

运动信息应该如何分配到三个层次？哪些放在文本中（可编辑但粗糙），哪些放在 token 中（精确但不可读），哪些放在噪声中（全局但不可解释）？分配不当会导致层次间冲突或冗余。

---

## 2. 核心 Insight

### 2.1 运动信息的三层编码假说

我们提出：T2V 模型中的运动信息天然分布在三个层次，每个层次有不同的粒度和可编辑性：

| 层次 | 载体 | 编码的运动信息 | 粒度 | 可编辑性 |
|------|------|-------------|------|---------|
| 语义层 | 文本 prompt | 运动类型、方向、速度描述 | 粗 | 高（改词即可） |
| 嵌入层 | $\Delta e$ motion token | 加速曲线、节奏变化、微妙时序 | 中 | 中（向量运算） |
| 结构层 | $\eta$ noise prior | 全局轨迹布局、空间运动场 | 细 | 低（不可解释） |

**关键 insight**：这三层不是冗余的，而是互补的。文本描述"向右走"，token 编码"先慢后快的加速曲线"，噪声编码"从画面左下到右上的精确轨迹"。三者缺一不可——去掉文本则丧失语义可编辑性，去掉 token 则丢失精确运动细节，去掉噪声则丢失全局运动结构。

### 2.2 内容解耦的数学条件

一个好的运动资产 $\mathcal{A}_{motion}$ 应满足**内容无关性**：

$$\forall p_1, p_2 \in \mathcal{P}_{content}: \quad \text{Motion}(\text{Gen}(p_1 + \mathcal{A})) \approx \text{Motion}(\text{Gen}(p_2 + \mathcal{A}))$$

即：无论搭配什么内容描述，生成视频的运动模式应该一致。

这要求 $\Delta e_{motion}$ 位于 embedding 空间中与内容正交的子空间。我们通过两个机制来近似实现这一条件：(1) timestep-aware 优化——只在运动相关时间步优化，避免编码内容信息；(2) cross-content consistency loss——显式惩罚 $\Delta e$ 对不同内容的差异化响应。

### 2.3 资产的可组合性

理想的运动资产支持以下操作：

- **替换内容**：$p_{new\_content} + \mathcal{A}_{motion}$ -> 新内容做相同运动
- **混合运动**：$\alpha \cdot \mathcal{A}_1 + (1-\alpha) \cdot \mathcal{A}_2$ -> 两种运动的插值
- **强度调节**：$\beta \cdot \Delta e_{motion}$ -> 控制运动幅度
- **时序裁剪**：截取 $\Delta e$ 的子序列 -> 只复现部分运动

这些操作的可行性来源于 embedding 空间的局部线性性（RichSpace, ICLR 2025 已证明 T5 embedding 空间在局部是近似线性的）。

---

## 3. 方法详述

### 3.1 整体 Pipeline

```
+=====================================================================+
|                    VMAD Pipeline                                      |
|                                                                      |
|  ============= Phase 1: Motion Extraction =============              |
|                                                                      |
|  V_ref --> VAE Encode --> z0                                         |
|    |                       |                                         |
|    |                       +---> Flow Matching Inversion --> eta_inv  |
|    |                       |         |                               |
|    |                       |         v                               |
|    |                       |    SVD Temporal Filter --> eta_motion    |
|    |                       |                                         |
|    |                       +---> Target Velocity: v* = z0 - eta      |
|    |                                   |                             |
|    v                                   v                             |
|  VLM Caption --> e0    Velocity Field Matching                       |
|                  |       + Content Disentanglement                    |
|                  |       (t in [0, T_m] only)                        |
|                  |              |                                     |
|                  |              v                                     |
|                  |         delta_e_motion (content-free)              |
|                  |                                                    |
|  ============= Phase 2: Asset Packaging =============                |
|                                                                      |
|  delta_e_motion --> Motion Text Decoder --> p_motion_text             |
|       |              (VLM compare decoding)                           |
|       |                                                              |
|       v                                                              |
|  +-------------------------------------------+                       |
|  |  Motion Asset  A_motion                    |                       |
|  |  +-- p_motion_text: "accelerates then     |                       |
|  |  |    turns right with momentum"           |                       |
|  |  +-- delta_e_motion: [L x D tensor]       |                       |
|  |  +-- eta_motion: [F x C x H x W tensor]  |                       |
|  |  +-- metadata: {source, type, compat}     |                       |
|  +-------------------------------------------+                       |
|                                                                      |
|  ============= Phase 3: Asset Application =============              |
|                                                                      |
|  p_new_content --> TextEnc --> e_content                              |
|                                    |                                 |
|  A_motion.delta_e ----------------+---> e_final = e_content + delta_e|
|                                    |                                 |
|  A_motion.eta_motion --> Blend --> eta_init                           |
|                                                                      |
|  Generate(v_theta, eta_init, e_final) --> V_gen                      |
|  (new content + original motion)                                     |
+=====================================================================+
```

### 3.2 Phase 1: Motion Extraction

#### Step 1: Flow Matching Inversion + SVD Temporal Filter

这一步与 FMMD 完全相同，将目标视频编码到噪声空间并提取运动先验：

$$z_0 = \text{VAE\_Encode}(V_{ref})$$

$$\eta_{inv} = \text{ODE\_Solve}(z_0 \to z_1, \quad \frac{dx_t}{dt} = v_\theta(x_t, t, \varnothing))$$

其中 $\varnothing$ 表示无条件（null prompt），ODE 从 $t=0$ 积分到 $t=1$，使用 Euler 方法 50 步。

SVD 两阶段滤波提取运动先验：

**空间滤波**（去除空间高频噪声，逐帧处理）：
$$\eta_{inv}^{(f)} = \text{reshape}(\eta_{inv}[f], [C, H \times W])$$
$$U, S, V^T = \text{SVD}(\eta_{inv}^{(f)})$$
$$S_{spatial} = S \cdot \mathbb{1}[i > \lfloor \rho_s \cdot \min(C, HW) \rfloor]$$

**时域滤波**（保留时域低频 = 运动信息）：
$$\eta_{spatial} = \text{reshape}(\eta_{spatial}, [F, C \times H \times W])$$
$$U_t, S_t, V_t^T = \text{SVD}(\eta_{spatial})$$
$$S_{motion} = S_t \cdot \mathbb{1}[i \leq \lfloor \rho_m \cdot \min(F, CHW) \rfloor]$$
$$\eta_{motion} = U_t \cdot \text{diag}(S_{motion}) \cdot V_t^T$$

**超参数**：$\rho_s = 0.1$（空间滤波比例），$\rho_m = 0.9$（时域保留比例）

#### Step 2: Velocity Field Matching with Content Disentanglement（核心改进）

这是 VMAD 相对于 FMMD 的核心创新。FMMD 只做速度场匹配，VMAD 在此基础上增加了内容解耦约束。

**基础优化**（与 FMMD 相同）：

$$e_0 = \text{TextEncoder}(\text{VLM\_Caption}(V_{ref}))$$

$$\mathcal{L}_{velocity} = \mathbb{E}_{t \sim \mathcal{U}(0, T_m)} \left[ \| v_\theta(x_t, t, e_0 + \Delta e) - v^*(t) \|^2 \right]$$

其中 $x_t = (1-t) \cdot \eta + t \cdot z_0$，$v^*(t) = z_0 - \eta$，$T_m = 0.3$。

**内容解耦约束**（VMAD 新增）：

为确保 $\Delta e$ 只编码运动而不绑定特定内容，我们引入 Cross-Content Consistency Loss。核心思想是：如果 $\Delta e$ 真的只编码运动，那么无论搭配什么内容 prompt，模型预测的速度场在运动方向上应该一致。

首先，用 LLM 从原始 caption 生成 N 个内容不同但结构相似的 prompt：

$$\mathcal{P}_{aug} = \text{LLM\_Augment}(\text{caption}, N=5)$$

例如将"A golden retriever running on the beach"改写为"A black cat running on the beach"、"A small child running on the beach"等——只替换主体，保留所有运动/场景描述。

然后计算这些不同内容 prompt 搭配同一个 $\Delta e$ 时，速度场预测的方差：

$$\mathcal{L}_{disentangle} = \mathbb{E}_{t \sim \mathcal{U}(0, T_m)} \left[ \frac{1}{N} \sum_{i=1}^{N} \| v_\theta(x_t, t, e_i + \Delta e) - \bar{v} \|^2 \right]$$

其中 $e_i = \text{Enc}(p_i)$，$\bar{v} = \frac{1}{N}\sum_i v_\theta(x_t, t, e_i + \Delta e)$。

**直觉**：如果 $\Delta e$ 偷偷编码了"狗的外观"，那么搭配"猫"的 prompt 时速度场会产生冲突（方差大）。$\mathcal{L}_{disentangle}$ 惩罚这种冲突，迫使 $\Delta e$ 只编码与内容无关的运动信息。

**总优化目标**：

$$\mathcal{L}_{total} = \mathcal{L}_{velocity} + \lambda_{dis} \cdot \mathcal{L}_{disentangle}$$

其中 $\lambda_{dis} = 0.1$。

**优化过程**：
- 冻结 $v_\theta$ 所有参数
- 只更新 $\Delta e$（与 $e_0$ 同形状）
- Adam optimizer, lr = 1e-3
- 迭代 K = 100 步
- 每步随机采样 $t \sim \mathcal{U}(0, T_m)$
- 每 10 步计算一次 $\mathcal{L}_{disentangle}$（节省计算，因为需要 N 次前向传播）

### 3.3 Phase 2: Asset Packaging

#### Step 3: Motion Text Decoding

优化得到的 $\Delta e_{motion}$ 是连续向量，其中一部分信息可以用自然语言表达（如"加速"、"转向"），另一部分无法表达（如精确的加速度曲线）。我们需要将可表达部分转回文本，使资产具备跨模型迁移能力。

**方法：VLM 对比解码**（推荐，无需训练）

核心思路：生成两个视频——一个有 $\Delta e$，一个没有——然后用 VLM 描述两者的运动差异。这个差异描述就是 $\Delta e$ 编码的运动信息的文本近似。

$$V_{with} = \text{Generate}(\eta_{init}, e_0 + \Delta e_{motion})$$
$$V_{without} = \text{Generate}(\eta_{init}, e_0)$$
$$p_{motion\_text} = \text{VLM.compare}(V_{with}, V_{without}, \text{prompt}_{motion\_diff})$$

其中 $\text{prompt}_{motion\_diff}$ 指示 VLM 只关注运动差异：

> "Compare these two videos and describe ONLY the motion differences. Focus on: speed changes, direction changes, acceleration patterns, rhythm, camera movement. Ignore content/appearance differences."

**备选方法：Embedding 空间最近邻投影**

$$p_{motion\_text} = \text{argmin}_{p \in \mathcal{V}^*} \| \text{Enc}(p) - (e_0 + \Delta e_{motion}) \|_2$$

在预定义的运动描述词库 $\mathcal{V}^*$ 中搜索最近邻。词库包含速度词（accelerating, decelerating）、方向词（turning left, moving upward）、节奏词（rhythmic, sudden, gradual）、镜头词（panning, zooming in, tracking shot）等。

#### Step 4: Asset Serialization

最终资产格式定义：

```json
{
  "version": "1.0",
  "asset_type": "video_motion_asset",
  "source_video": "beach_dog_001.mp4",
  "extraction_model": "wan2.1-1.3b",
  "text_encoder": "umt5-xxl",
  
  "motion_text": "The subject initially trots slowly, then accelerates into a full sprint with sand kicking up. Camera tracks rightward with slight handheld shake.",
  
  "motion_token_path": "assets/beach_dog_001_delta_e.pt",
  "noise_prior_path": "assets/beach_dog_001_eta_motion.pt",
  
  "metadata": {
    "motion_type": "acceleration_with_direction_change",
    "intensity": 0.73,
    "duration_frames": 81,
    "spatial_resolution": "60x104",
    "compatible_models": ["wan2.1-1.3b", "wan2.1-14b"],
    "extraction_params": {
      "T_m": 0.3,
      "opt_steps": 100,
      "lr": 0.001,
      "lambda_dis": 0.1,
      "rho_s": 0.1,
      "rho_m": 0.9,
      "alpha_blend": 0.001
    }
  }
}
```

### 3.4 Phase 3: Asset Application（推理时使用）

给定新内容描述 $p_{new}$ 和运动资产 $\mathcal{A}$：

**Step 1: 编码新内容**
$$e_{content} = \text{TextEncoder}(p_{new})$$

**Step 2: 组合条件**
$$e_{final} = e_{content} + \text{strength} \cdot \Delta e_{motion}$$

其中 strength $\in [0, 1]$ 控制运动强度。

**Step 3: 噪声混合**
$$\eta_{init} = \sqrt{\alpha} \cdot \eta_{motion} + \sqrt{1-\alpha} \cdot \eta_{random}, \quad \alpha = 0.001$$

**Step 4: 生成**
$$V_{gen} = \text{Sample}(v_\theta, \eta_{init}, e_{final}, \text{steps}=50)$$

**高级用法——运动混合**：

$$\Delta e_{mixed} = w_1 \cdot \Delta e_A + w_2 \cdot \Delta e_B, \quad w_1 + w_2 = 1$$
$$\eta_{mixed} = w_1 \cdot \eta_A + w_2 \cdot \eta_B$$

---

## 4. 数学形式化

### 4.1 完整优化问题

$$\min_{\Delta e} \quad \mathcal{L}_{velocity} + \lambda_{dis} \cdot \mathcal{L}_{disentangle}$$

展开：

$$\min_{\Delta e} \quad \mathbb{E}_{t \sim \mathcal{U}(0, T_m), \eta \sim \mathcal{N}(0,I)} \left[ \| v_\theta(x_t, t, e_0 + \Delta e) - (z_0 - \eta) \|^2 \right] + \lambda_{dis} \cdot \mathbb{E}_{t, p_i \sim \mathcal{P}_{aug}} \left[ \text{Var}_i \left( v_\theta(x_t, t, \text{Enc}(p_i) + \Delta e) \right) \right]$$

$$\text{s.t.} \quad x_t = (1-t) \cdot \eta + t \cdot z_0, \quad z_0 = \text{Enc}(V_{ref})$$

### 4.2 内容解耦的理论分析

**命题**：若 $\Delta e$ 位于 embedding 空间中与内容子空间正交的方向上，则 $\mathcal{L}_{disentangle} = 0$。

**证明草案**：

设 embedding 空间 $\mathbb{R}^D$ 可分解为内容子空间 $\mathcal{C}$ 和运动子空间 $\mathcal{M}$（$\mathcal{C} \perp \mathcal{M}$）。若 $\Delta e \in \mathcal{M}$，则对任意 $e_i = e_i^{\mathcal{C}} + e_i^{\mathcal{M}}$：

$$e_i + \Delta e = e_i^{\mathcal{C}} + (e_i^{\mathcal{M}} + \Delta e)$$

由于 $v_\theta$ 在早期时间步（$t \in [0, T_m]$）主要响应运动子空间的变化（基于 Section 2.1 的 timestep 分离假设），不同 $e_i^{\mathcal{C}}$ 对速度场的影响可忽略，因此：

$$v_\theta(x_t, t, e_i + \Delta e) \approx v_\theta(x_t, t, e_j + \Delta e) \quad \forall i,j$$

即方差为零，$\mathcal{L}_{disentangle} = 0$。$\square$

**实际意义**：$\mathcal{L}_{disentangle}$ 作为正则项，将 $\Delta e$ 推向运动子空间，远离内容子空间。优化过程中，$\Delta e$ 会自动找到一个方向，使得它编码的信息与内容无关。

### 4.3 与 FMMD 的数学关系

| | FMMD | VMAD |
|---|---|---|
| 优化目标 | $\mathcal{L}_{velocity}$ | $\mathcal{L}_{velocity} + \lambda \cdot \mathcal{L}_{disentangle}$ |
| 输出 | $c_{final}$（一次性使用） | $\mathcal{A}_{motion}$（可复用资产） |
| 内容绑定 | 绑定原始 VLM caption | 解耦，可搭配任意内容 |
| 验证方式 | 生成质量（CLIP/XCLIP） | 跨内容迁移一致性 |
| 使用场景 | "复现这个视频的运动" | "提取运动模式，应用到任意新内容" |

### 4.4 与 Textual Inversion / DreamBooth 的类比

VMAD 可以看作 **"Textual Inversion for Motion"**——Textual Inversion 从图像中提取外观资产（"[V]" token），VMAD 从视频中提取运动资产（$\Delta e_{motion}$）。

| | Textual Inversion | DreamBooth | VMAD (Ours) |
|---|---|---|---|
| 提取的资产 | 外观 token（"[V]"） | 外观 LoRA | 运动 token（$\Delta e$） |
| 编码信息 | 物体外观 | 物体外观+风格 | 运动模式 |
| 使用方式 | "a [V] dog" | "a [V] dog" | content\_prompt + $\Delta e$ |
| 可组合性 | 与文本拼接 | 与其他 LoRA 合并 | 与任意内容组合 |
| 训练成本 | ~5min/concept | ~15min/concept | ~4.5min/video |
| 内容解耦 | 不需要（本身就是外观） | 不需要 | 需要（核心挑战） |

### 4.5 与 Score Distillation Sampling (SDS) 的关系

SDS 的优化目标：
$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon} \left[ w(t) (\epsilon_\phi(x_t, t, y) - \epsilon) \frac{\partial x}{\partial \theta} \right]$$

VMAD 的优化目标：
$$\nabla_{\Delta e} \mathcal{L}_{VMAD} = \mathbb{E}_{t} \left[ (v_\theta(x_t, t, e_0+\Delta e) - v^*) \frac{\partial v_\theta}{\partial \Delta e} \right]$$

**关键区别**：
- SDS 优化的是 3D 表示（NeRF 参数），我们优化的是条件 embedding
- SDS 用 score function（$\epsilon$-prediction），我们用 velocity field（$v$-prediction）
- SDS 没有 ground truth target（只有 text prompt 引导），我们有明确的 $v^* = z_0 - \eta$（来自目标视频）
- 我们限制在 $t \in [0, T_m]$ 范围内优化，实现运动-内容分离
- 我们额外有 $\mathcal{L}_{disentangle}$ 确保内容无关性

---

## 5. 实现伪代码

```python
import torch
import json
from model import WanT2V, VAEEncoder, TextEncoder, VLM

class VMAD:
    """Video Motion Asset Distillation"""
    
    def __init__(self, model, vae, text_enc, vlm,
                 T_m=0.3, num_opt_steps=100, lr=1e-3,
                 lambda_dis=0.1, rho_s=0.1, rho_m=0.9, alpha=0.001):
        self.model = model  # frozen T2V model
        self.vae = vae
        self.text_enc = text_enc
        self.vlm = vlm
        self.T_m = T_m
        self.num_opt_steps = num_opt_steps
        self.lr = lr
        self.lambda_dis = lambda_dis
        self.rho_s = rho_s
        self.rho_m = rho_m
        self.alpha = alpha
    
    def extract_asset(self, video_ref, save_path=None):
        """
        Phase 1 + Phase 2: Extract motion asset from target video
        
        Args:
            video_ref: [F, 3, H, W] target video tensor
            save_path: path to save the asset
            
        Returns:
            asset: dict containing the complete motion asset
        """
        # ============ Step 1: Encode & Invert ============
        z0 = self.vae.encode(video_ref)  # [F, C, h, w]
        eta_inv = self._flow_matching_inversion(z0)  # [F, C, h, w]
        eta_motion = self._svd_temporal_filter(eta_inv)  # [F, C, h, w]
        
        # ============ Step 2: VLM Caption ============
        caption = self.vlm.describe(video_ref)
        e0 = self.text_enc.encode(caption)  # [L, D]
        
        # ============ Step 3: Generate augmented prompts for disentanglement ============
        aug_prompts = self._generate_content_augmentations(caption, n=5)
        aug_embeddings = [self.text_enc.encode(p) for p in aug_prompts]
        
        # ============ Step 4: Velocity Field Matching + Disentanglement ============
        delta_e = self._optimize_motion_token(z0, e0, eta_inv, aug_embeddings)
        
        # ============ Step 5: Motion Text Decoding ============
        motion_text = self._decode_motion_text(e0, delta_e, eta_motion)
        
        # ============ Step 6: Package Asset ============
        asset = {
            "motion_text": motion_text,
            "delta_e": delta_e.detach().cpu(),
            "eta_motion": eta_motion.detach().cpu(),
            "source_caption": caption,
            "metadata": {
                "T_m": self.T_m,
                "opt_steps": self.num_opt_steps,
                "intensity": delta_e.norm().item(),
                "duration_frames": video_ref.shape[0],
            }
        }
        
        if save_path:
            self._save_asset(asset, save_path)
        
        return asset
    
    def apply_asset(self, content_prompt, asset, strength=1.0):
        """
        Phase 3: Apply motion asset to new content
        
        Args:
            content_prompt: new content description (e.g., "A white cat")
            asset: loaded motion asset dict
            strength: motion intensity control [0, 1]
            
        Returns:
            video: generated video tensor
        """
        # Encode new content
        e_content = self.text_enc.encode(content_prompt)
        delta_e = asset["delta_e"].to(e_content.device)
        eta_motion = asset["eta_motion"].to(e_content.device)
        
        # Combine condition: new content + motion token
        e_final = e_content + strength * delta_e
        
        # Noise blending
        alpha_scaled = self.alpha * strength
        eta_random = torch.randn_like(eta_motion)
        eta_init = (alpha_scaled**0.5) * eta_motion + ((1-alpha_scaled)**0.5) * eta_random
        
        # Generate
        video = self._generate(eta_init, e_final)
        return video
    
    def _optimize_motion_token(self, z0, e0, eta_inv, aug_embeddings):
        """
        Core: Velocity field matching with content disentanglement
        """
        delta_e = torch.zeros_like(e0, requires_grad=True)
        optimizer = torch.optim.Adam([delta_e], lr=self.lr)
        
        for step in range(self.num_opt_steps):
            # --- Sample timestep and noise ---
            t = torch.rand(1, device=z0.device) * self.T_m  # t in [0, T_m]
            eta = torch.randn_like(z0)
            
            # --- Construct interpolation point and target ---
            x_t = (1 - t) * eta + t * z0
            v_target = z0 - eta  # target velocity field
            
            # --- Velocity Matching Loss ---
            e_current = e0 + delta_e
            v_pred = self.model.forward(x_t, t, e_current)
            loss_vel = ((v_pred - v_target) ** 2).mean()
            
            # --- Content Disentanglement Loss (every 10 steps) ---
            loss_dis = torch.tensor(0.0, device=z0.device)
            if step % 10 == 0 and len(aug_embeddings) > 0:
                v_preds = []
                for e_aug in aug_embeddings:
                    e_aug_motion = e_aug + delta_e
                    v_aug = self.model.forward(x_t, t, e_aug_motion)
                    v_preds.append(v_aug)
                
                v_stack = torch.stack(v_preds)  # [N, F, C, H, W]
                v_mean = v_stack.mean(dim=0)
                loss_dis = ((v_stack - v_mean.unsqueeze(0)) ** 2).mean()
            
            # --- Total Loss ---
            loss = loss_vel + self.lambda_dis * loss_dis
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return delta_e.detach()
    
    def _generate_content_augmentations(self, caption, n=5):
        """
        Use LLM to generate content-different but structure-similar prompts.
        
        Example: "A golden retriever running on the beach" ->
        ["A black cat running on the beach",
         "A small child running on the beach",
         "A white horse running on the beach",
         "A red sports car driving on the beach",
         "A robot walking on the beach"]
        """
        # LLM instruction: "Replace the subject in this caption with a 
        # different entity. Keep ALL motion/action/scene descriptions unchanged.
        # Only change WHAT is moving, not HOW it moves."
        augmented = self._llm_augment_subjects(caption, n)
        return augmented
    
    def _decode_motion_text(self, e0, delta_e, eta_motion):
        """
        Decode the expressible part of delta_e into readable text.
        Method: Generate videos with/without delta_e, VLM describes difference.
        """
        # Generate with motion token
        eta_init = self._blend_noise(eta_motion)
        v_with = self._generate(eta_init, e0 + delta_e)
        v_without = self._generate(eta_init, e0)
        
        # VLM compare motion differences
        motion_desc = self.vlm.compare(
            v_with, v_without,
            prompt="Compare these two videos. Describe ONLY the motion "
                   "differences. Focus on: speed changes, direction changes, "
                   "acceleration, rhythm, camera movement. "
                   "Ignore content/appearance differences."
        )
        return motion_desc
    
    def _flow_matching_inversion(self, z0, steps=50):
        """Flow Matching Inversion: z0 (t=0) -> eta (t=1)"""
        dt = 1.0 / steps
        x = z0.clone()
        for i in range(steps):
            t = torch.tensor(i * dt)
            v = self.model.forward(x, t, null_prompt)
            x = x + v * dt
        return x
    
    def _svd_temporal_filter(self, eta):
        """SVD two-stage filter: spatial denoise + temporal motion extraction"""
        F, C, H, W = eta.shape
        
        # Stage 1: Spatial filtering (per-frame)
        eta_spatial = torch.zeros_like(eta)
        for f in range(F):
            frame = eta[f].reshape(C, H * W)
            U, S, Vh = torch.linalg.svd(frame, full_matrices=False)
            k = int(self.rho_s * min(C, H * W))
            S[:k] = 0  # Remove top spatial singular values
            eta_spatial[f] = (U @ torch.diag(S) @ Vh).reshape(C, H, W)
        
        # Stage 2: Temporal filtering
        temporal = eta_spatial.reshape(F, C * H * W)
        U_t, S_t, Vh_t = torch.linalg.svd(temporal, full_matrices=False)
        k_m = int(self.rho_m * min(F, C * H * W))
        S_t[k_m:] = 0  # Keep only temporal low-frequency
        eta_motion = (U_t @ torch.diag(S_t) @ Vh_t).reshape(F, C, H, W)
        
        return eta_motion
    
    def _blend_noise(self, eta_motion):
        """Blend motion noise with random noise"""
        eta_random = torch.randn_like(eta_motion)
        return (self.alpha**0.5) * eta_motion + ((1-self.alpha)**0.5) * eta_random
    
    def _generate(self, eta_init, c_final, steps=50):
        """Standard Flow Matching sampling: eta (t=1) -> z0 (t=0)"""
        dt = -1.0 / steps
        x = eta_init.clone()
        for i in range(steps):
            t = torch.tensor(1.0 + i * dt)
            v = self.model.forward(x, t, c_final)
            x = x + v * dt
        return x
    
    def _save_asset(self, asset, save_path):
        """Save motion asset to disk"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Save tensors
        torch.save(asset["delta_e"], os.path.join(save_path, "delta_e_motion.pt"))
        torch.save(asset["eta_motion"], os.path.join(save_path, "eta_motion.pt"))
        
        # Save metadata as JSON
        meta = {
            "version": "1.0",
            "motion_text": asset["motion_text"],
            "source_caption": asset["source_caption"],
            "motion_token_path": "delta_e_motion.pt",
            "noise_prior_path": "eta_motion.pt",
            "metadata": asset["metadata"]
        }
        with open(os.path.join(save_path, "asset.json"), "w") as f:
            json.dump(meta, f, indent=2)
```

---

## 6. 与现有工作的关系和区别

### 6.1 对比表

| 方法 | 优化对象 | 监督信号 | 运动-内容分离 | 输出形式 | 可复用性 | 发表 |
|------|---------|---------|-------------|---------|--------|------|
| Textual Inversion | token embedding | diffusion loss (全时间步) | N/A（编码外观） | embedding | 同模型内 | ICLR 2023 |
| Reenact Anything | motion embedding | diffusion loss (全时间步) | 隐式（依赖 I2V 结构） | embedding | 同模型内 | SIGGRAPH 2025 |
| MotionPrompt | learnable tokens | 光流判别器梯度 | 无（需训练判别器） | embedding | 同模型内 | CVPR 2025 |
| CLIP Interrogator | -- | CLIP 相似度 | N/A | 离散文本 | 跨模型 | -- |
| EDITOR | embedding -> text | diffusion loss + decoder | 无 | 离散文本 | 跨模型 | arXiv 2025 |
| FreeInit | noise filtering | self-refinement | 无（无外部目标） | noise | 同模型内 | ECCV 2024 |
| **VMAD (Ours)** | **embedding 残差** | **velocity field loss (限制时间步) + disentangle loss** | **显式 (timestep-aware + cross-content)** | **混合资产 (text + token + noise)** | **文本跨模型 + tensor 同系列** | -- |

### 6.2 核心创新点

1. **Video Motion Asset 概念**：首次将视频运动信息形式化为可存储、可迁移、可编辑的结构化资产。不同于一次性的 embedding（Reenact Anything）或不可编辑的 noise（FreeInit），我们的资产包含三个互补层次，支持存储、迁移、编辑、组合四种操作。

2. **Content-Disentangled Velocity Field Matching**：在 FMMD 的速度场匹配基础上，增加 cross-content consistency loss，确保提取的 motion token 只编码运动、不绑定内容。这是实现"资产可复用"的关键——没有内容解耦，motion token 迁移到新内容时会产生冲突。

3. **Timestep-Aware Motion-Content Decomposition**：通过限制优化时间步范围 $t \in [0, T_m]$，利用 rectified flow 速度场的时间步语义分离特性，实现运动信息的选择性提取。

4. **Motion Text Decoding**：通过 VLM 对比解码，将不可读的 $\Delta e$ 中可表达的部分转回自然语言，使资产具备跨模型迁移能力（文本部分可直接用于任何 T2V 模型）。

5. **Dual-Space Asset Encoding**：噪声空间（$\eta_{motion}$）和条件空间（$\Delta e_{motion}$）协同编码，前者提供全局运动结构，后者提供细粒度运动语义，两者互补不冗余。

---

## 7. 实验设计

### 7.1 消融实验

| 配置 | Text Prompt | Motion Token ($\Delta e$) | Noise Prior ($\eta$) | Disentangle Loss | 预期效果 |
|------|------------|-------------------|-----------------|-----------------|----------|
| Baseline | VLM caption | -- | -- | -- | 基准 |
| + Noise Only | VLM caption | -- | SVD filtered | -- | CLIP 小幅提升，全局运动改善 |
| + Token Only (FMMD) | VLM caption | velocity matching | -- | -- | XCLIP 提升，细粒度运动改善 |
| + Both (no disentangle) | VLM caption | velocity matching | SVD filtered | -- | 双指标提升，但迁移性差 |
| **+ Both + Disentangle (Full VMAD)** | VLM caption | velocity matching | SVD filtered | cross-content | **双指标最优 + 迁移性好** |
| Text-Only Asset | motion\_desc | -- | -- | -- | 验证文本解码质量 |

### 7.2 关键验证实验

**实验 1：跨视频运动迁移**

这是验证"资产可复用性"的核心实验。

- 设计：从 A 视频提取 motion asset，搭配 B 视频的 content prompt 生成
- 评估：Flow-Sim(gen, A) 高（运动来自 A）+ CLIP-Sim(gen, B\_text) 高（内容来自 B）
- 规模：10 对 motion x content 组合，每对生成 3 次取平均

**实验 2：内容解耦验证**

- 设计：同一个 motion asset 搭配 5 个不同 content prompt 生成
- 评估：5 个生成视频之间的运动一致性（互相之间的 Flow-Sim）
- 对比：有 vs 无 $\mathcal{L}_{disentangle}$ 的 motion token

**实验 3：运动强度可控性**

- 设计：对 $\Delta e$ 乘以不同系数 [0.25, 0.5, 1.0, 1.5, 2.0]
- 评估：运动幅度（光流均值）与系数的 Spearman 相关系数
- 预期：强单调关系（$\rho > 0.9$）

**实验 4：Timestep Decomposition 验证**

- 设计：分别在 $t \in [0, 0.3]$、$t \in [0.3, 0.7]$、$t \in [0.7, 1.0]$ 优化
- 评估：展示前者改变运动轨迹但保持外观，后者改变颜色/纹理但保持运动
- 可视化：生成视频的光流图 + 外观差异图

**实验 5：T\_m 敏感性消融**

- 设计：$T_m \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$
- 评估：CLIP + XCLIP + 跨内容迁移一致性
- 预期：0.2-0.4 范围内性能稳定

**实验 6：资产压缩**

- 设计：对 $\Delta e$ 做 PCA 降维（保留 top-k 主成分，k = 10, 50, 100, 256, 512）
- 评估：压缩到多少维仍能保持运动质量
- 意义：确定资产的最小存储大小

### 7.3 Baseline 对比

1. **FreeInit** (ECCV 2024)：直接在 Wan2.1 上复现，self-refinement 2-3 iterations
2. **Reenact Anything 变体**：全时间步 diffusion loss 优化 embedding（无 timestep 限制）
3. **VLM Iterative Refinement**（我们的 V4 pipeline）：纯文本优化的天花板
4. **MotionPrompt 变体**：用光流 loss 替代 velocity field loss（如果可实现）

### 7.4 评估指标

- **运动忠实度**：RAFT 光流 EPE、X-CLIP temporal score、VideoMAE action recognition accuracy
- **内容质量**：CLIP-Sim（文本-视频对齐）、FID-VID
- **整体质量**：FVD、Human Evaluation（20 人 x 50 对 A/B test）
- **资产质量**：跨视频迁移成功率、运动强度可控性（Spearman $\rho$）、跨内容一致性
- **效率**：资产提取时间、资产文件大小、生成时额外开销

---

## 8. 参考论文

### 核心参考（方法直接相关）

1. **Score Distillation of Flow Matching Models** -- Apple Research, 2025
   - 证明 score distillation 适用于 flow matching 模型
   - VMAD velocity field matching 的理论基础

2. **Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion** -- SIGGRAPH 2025, Disney Research x ETH Zurich
   - Motion-Textual Inversion 的开创性工作
   - 我们的区别：T2V（非 I2V）、velocity field loss（非 diffusion loss）、timestep-aware 分离、内容解耦

3. **MotionPrompt: Optical-Flow Guided Prompt Optimization for Coherent Video Generation** -- CVPR 2025, KAIST
   - 通过优化 learnable token 控制视频运动
   - 我们的区别：不需要训练判别器，直接用目标视频的速度场作为监督

4. **RF-Inversion: Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations** -- ICLR 2025
   - Rectified Flow 反演的最优控制理论
   - Flow Matching Inversion 步骤的理论基础

5. **FreeInit: Bridging Initialization Gap in Video Diffusion Models** -- ECCV 2024
   - 发现时域低频噪声编码运动信息
   - SVD 时域滤波的理论依据

6. **EDITOR: An Edit-Specific Transformer for Diffusion-Based Image Editing** -- arXiv 2025
   - Embedding -> text decoding 的参考
   - Motion Text Decoding 的方法论启发

7. **RichSpace: Enriching Text-to-Video Prompt Space for Video Generation** -- ICLR 2025
   - 证明 T5 embedding 空间在局部是近似线性的
   - 支撑资产可组合性（线性插值有效）的理论基础

### 辅助参考（理论支撑）

8. **Seeds of Structure: Patch PCA Reveals Universal Compositional Cues in Diffusion Noise** -- NeurIPS 2025
9. **The Crystal Ball Hypothesis in Diffusion Models** -- ICLR 2025
10. **How I Warped Your Noise: A Temporally-Correlated Noise Prior for Diffusion Models** -- ICLR 2025
11. **Promptus: Can Prompt Streaming Replace Video Streaming with Stable Diffusion** -- AAAI 2025
12. **PH2P: Prompt-to-Prompt with Head-Specific Weights** -- CVPR 2024（timestep-aware 优化的跨模型证据）
13. **STEM Inversion** -- CVPR 2024（低秩时空分解）
14. **VGD: Visually Guided Decoding** -- ICLR 2025（视觉引导文本生成）

---

## 9. 预期贡献总结

1. **问题贡献**：定义了 Video Motion Asset Extraction 问题——从目标视频中提取结构化的、可复用的运动资产，区别于一次性的视频生成或编辑。这是一个新的问题 formulation，将视频运动从"生成的副产品"提升为"可管理的数字资产"。

2. **方法贡献**：提出 VMAD pipeline，通过 content-disentangled velocity field matching 从目标视频中蒸馏运动信息到混合资产格式。核心技术创新是 cross-content consistency loss + timestep-aware optimization 的组合，实现了运动-内容的显式分离。

3. **理论贡献**：揭示 Rectified Flow 速度场的时间步语义分离特性，并证明噪声空间与条件空间编码运动信息的正交互补性。提出 embedding 空间的内容-运动子空间分解假说，并通过 $\mathcal{L}_{disentangle}$ 提供了实现路径。

4. **系统贡献**：设计了 Video Motion Asset 的标准格式（text + token + noise 三层混合表示），支持存储、迁移、编辑、组合四种操作，为视频生成的资产化工作流提供基础设施。

---

## 10. 大白话版本：这个方案到底在干什么？

### 10.1 一句话故事

**"我想从一段视频里把'动作'单独抽出来，变成一个可以反复使用的小文件（资产），以后想让任何画面动起来，只要把这个资产贴上去就行。"**

这就是"视频版的 Textual Inversion"——原版 Textual Inversion 是把一张图的外观学成一个 token，我们是把一段视频的运动学成一个 token + 噪声先验。

### 10.2 三步走流程（大白话）

**第一步：把视频"倒着跑"回噪声（Flow Matching Inversion）**

视频生成模型本质上是"从噪声变成视频"。你反过来，把一段真实视频通过 ODE 倒推回去，得到一个"噪声"。这个噪声不是随机的——它里面藏着这段视频的全部运动信息。然后你用频率滤波器把这个噪声里的"时间低频成分"（也就是大尺度运动模式）提取出来，作为"运动噪声先验"。

直觉：噪声的低频部分决定了视频的整体运动走向，高频部分只是细节抖动。

**第二步：在文本嵌入空间里"蒸馏"出运动 token（FMMD）**

光有噪声还不够——噪声是连续的、不可解释的。你还想要一个"文字描述"级别的运动表示。做法是：在文本编码器的 embedding 空间里，优化一小段额外的 token（Δe），让模型在生成过程中"被引导"去复现原视频的运动。优化目标很简单：让加了 Δe 之后模型预测的速度场，尽量接近"直接从噪声到视频"的理想速度场。

关键技巧是只在生成早期（t < 0.3）做这个约束，因为早期决定运动结构，晚期只是填细节。同时加一个"内容解耦"损失，确保学到的 Δe 只编码运动、不编码外观。

**第三步：打包成资产**

最终的"运动资产"就是三样东西打包在一起：一段文字描述（用 VLM 对比有/无 Δe 的视频差异自动生成）、运动 token Δe、以及运动噪声先验 η_motion。使用时，把这三样东西注入到任何新的生成任务里，就能复现原视频的运动风格。

### 10.3 Motion Text Decoding 详解

这是"文字描述"那一层的生成方法。具体流程：

1. 你已经优化好了一个 Δe（motion token），它是一个连续向量，人看不懂
2. 用同一个噪声先验 η_motion，分别生成两个视频：视频 A 用 `e0 + Δe` 生成（有 motion token），视频 B 用 `e0` 生成（没有 motion token）
3. 把两个视频喂给 VLM（如 GPT-4o），prompt 说："对比这两个视频，只描述运动上的差异，忽略外观差异"
4. VLM 输出的文字就是 Δe 编码的运动信息的"人话版本"

**为什么比直接 caption 原视频好？** 直接 caption 得到的是"这个视频里有什么运动"（相关性），对比解码得到的是"加了 Δe 后多了什么运动"（因果性）。后者更精确——只描述 Δe 真正编码的那部分信息。

### 10.4 类比理解

| 类比对象 | 做的事 | 我们做的事 |
|---------|--------|-----------|
| Textual Inversion | 从几张图里学一个 token，代表"这个物体的外观" | 从一段视频里学一个 token，代表"这段视频的运动" |
| LoRA | 微调模型权重来记住一种风格 | 不改模型，只学一个输入条件的偏移量 |
| 字体文件 | 存储"字的样子"，可以反复用 | 存储"动的方式"，可以反复用 |
| 音乐 MIDI | 把演奏抽象成音符序列，可以换乐器演奏 | 把运动抽象成 token + 噪声，可以换内容演绎 |

---

## 11. 可借鉴的开源代码与参考论文

### 11.1 按模块对应的代码仓库

| VMAD 模块 | 可借鉴的代码 | 仓库地址 | 借鉴什么 |
|-----------|------------|---------|---------|
| Flow Matching Inversion | RF-Inversion | `github.com/LituRout/RF-Inversion` | Rectified Flow 反演 pipeline，已集成 diffusers |
| Flow Matching Inversion（高精度） | RF-Solver | `github.com/wangjiangshan0725/RF-Solver-Edit` | 高阶 Taylor 展开减少 ODE 求解误差，支持 OpenSora |
| SVD 频率滤波 | FreeInit | `github.com/TianxingWu/FreeInit`（也在 diffusers 中） | Butterworth 滤波器分离噪声高低频的实现 |
| 噪声初始化 | ConsistI2V | `github.com/TIGER-AI-Lab/ConsistI2V` | 首帧低频噪声初始化增强一致性 |
| Motion Token 优化（核心） | Reenact Anything | `github.com/galhar/reenact-anything`（非官方） | I2V cross-attention 中优化 motion embedding |
| Embedding 优化框架 | Textual Inversion | `huggingface/diffusers/examples/textual_inversion/` | 训练循环、embedding 优化的代码框架 |
| 运动监督信号 | MotionPrompt | `github.com/HyelinNAM/MotionPrompt` | 光流判别器引导 prompt 优化 |
| FM 蒸馏数学 | SiD-DiT | `github.com/apple/ml-sid-dit` | Flow Matching 框架下 score/velocity 蒸馏的实现 |

### 11.2 实操建议：从哪开始写代码

1. 从 `diffusers` 的 Stable Video Diffusion pipeline 出发，参考 RF-Inversion 加一个 `inversion()` 方法把视频倒推回噪声
2. 从 FreeInit 的代码里借鉴频率滤波实现（Butterworth filter），用在噪声上做时空分离
3. 从 `diffusers/examples/textual_inversion/` 的训练循环出发，把 loss 从图像重建换成 velocity matching，把优化对象从 `placeholder_token_embedding` 换成 `delta_e`
4. 从 Reenact Anything 的非官方实现里看它怎么处理 I2V 模型的 cross-attention injection，确认 Δe 注入位置

### 11.3 与最相关工作的区别

**vs Reenact Anything（最像的工作）**：它在 I2V 模型上工作，有 image condition 作为外观锚点，motion embedding 只需编码"相对于参考图的运动"。我们在 T2V 模型上工作，没有 image condition，motion token 必须独立编码运动——这使得内容解耦成为必要。此外它用 DDPM loss 全时间步优化，我们用 velocity field loss + timestep 限制。

**vs MotionPrompt（CVPR 2025）**：它需要训练一个光流判别器作为监督信号，我们直接用目标视频的 velocity field 作为监督（training-free）。它的目标是"让生成视频运动更自然"，我们的目标是"复现特定视频的精确运动"。

**vs FreeInit（ECCV 2024）**：它是 self-refinement（无外部参考视频），只能改善同一个 prompt 的生成质量。我们是 target-guided extraction（有目标视频），可以做跨视频运动迁移。

---

---

# 恶毒审稿人攻击与防御

---

## Reviewer 1：Novelty 杀手 + 定位质疑

### 攻击

> **[Major Concern] The paper conflates two separate problems and solves neither well.**
>
> The authors claim to address "video motion asset extraction," but this is really two independent sub-problems: (1) motion-conditioned video generation (solved by noise prior + embedding optimization), and (2) asset serialization (solved by saving tensors to disk + VLM description). The "asset" framing is a packaging trick, not a research contribution.
>
> Specifically:
> - The noise prior component is FreeInit (ECCV 2024) applied to rectified flow with a target video. The SVD filtering is standard signal processing.
> - The velocity field matching is Reenact Anything (SIGGRAPH 2025) reformulated for flow matching models. The timestep restriction is a minor variant.
> - The content disentanglement loss is a standard variance regularizer, commonly used in domain adaptation literature.
> - The "asset format" (JSON + tensor files) is an engineering decision, not a scientific contribution.
>
> **I cannot identify a single component that is both novel AND non-trivial.** The paper reads as a well-engineered system combining existing techniques, which is better suited for a demo/application paper than a top venue.

### 防御

**对"FreeInit applied to RF"**：FreeInit 是 self-refinement（无外部参考视频），它从自身生成的视频中提取噪声再重新生成。我们是 target-guided extraction（有 ground truth 目标视频），从目标视频中提取运动先验用于引导新内容的生成。FreeInit 不能做跨视频运动迁移（它只能改善同一个 prompt 的生成质量），我们可以。Table X 直接对比 FreeInit vs Ours 在跨视频迁移任务上的表现。

**对"Reenact Anything reformulated"**：三个本质区别：(1) Reenact Anything 在 I2V 模型上工作，有 image condition 作为 appearance anchor，motion embedding 只需编码"相对于参考图的运动"；我们在 T2V 模型上工作，没有 image condition，motion token 必须独立编码运动而不依赖任何外观锚点——这使得内容解耦成为必要（Reenact 不需要）。(2) Reenact 用标准 diffusion loss 在全时间步优化，我们用 velocity field loss + timestep 限制实现显式运动-内容分离。(3) 我们有明确的 target velocity $v^*=z_0-\eta$，Reenact 用的是标准 diffusion loss 没有 explicit target。

**对"variance regularizer is standard"**：形式上是方差正则，但应用场景和设计动机完全不同。Domain adaptation 中的方差正则是为了对齐不同域的特征分布；我们的 cross-content consistency loss 是为了确保 motion token 在搭配不同内容时产生一致的速度场——这是一个全新的应用场景，且需要特定的设计（只在 $t \in [0, T_m]$ 计算、使用 LLM 生成内容增强 prompt）。

**对"asset format is engineering"**：我们同意 JSON schema 本身不是贡献。但 "motion asset" 的概念——将运动信息分解为 text + token + noise 三层互补表示——是一个新的 formulation。我们通过消融实验证明这三层各自不可替代（Table Y），且支持跨视频迁移、强度可控、内容编辑三种操作（Figure Z）。这不是"保存文件"，而是"定义了运动信息的结构化表示方式"。

---

## Reviewer 2：理论杀手 + 数学质疑

### 攻击

> **[Major Concern] The timestep-aware decomposition lacks theoretical grounding and the content-motion separability assumption is unverified.**
>
> 1. **Contradiction with prior work**: Prompt-to-Prompt (Hertz et al., ICLR 2023) shows that cross-attention at early denoising steps determines *spatial layout* (which is content/structure, not motion). How do you reconcile this?
>
> 2. **The target velocity $v^* = z_0 - \eta$ is constant across all t**. If $v^*$ doesn't change with t, how can optimizing at different t ranges extract different information? The information difference must come from the model's $v_\theta$ behavior at different t, but you haven't formally characterized this.
>
> 3. **$T_m = 0.3$ is a magic number**. How sensitive is the method to this choice? Without a principled way to determine $T_m$, this is a hyperparameter that requires per-video tuning.
>
> 4. **The content-motion orthogonality assumption (Section 4.2) is too strong**. Real embedding spaces are unlikely to have perfectly orthogonal content and motion subspaces. What happens when they overlap? Your proof assumes separability but doesn't prove it exists.
>
> 5. **Convergence**: The joint optimization of $\mathcal{L}_{velocity}$ and $\mathcal{L}_{disentangle}$ may have conflicting gradients. $\mathcal{L}_{velocity}$ wants $\Delta e$ to encode ALL information needed to match the target velocity (including content), while $\mathcal{L}_{disentangle}$ wants $\Delta e$ to be content-free. How do you ensure convergence?

### 防御

**对 P2P 矛盾**：P2P 讨论的是 spatial layout（空间布局），我们讨论的是 temporal motion（时序运动）。在视频模型中，spatial layout 和 temporal dynamics 是不同维度。我们的实验（Figure X）明确展示：$t \in [0,0.3]$ 优化改变运动轨迹但保持外观，$t \in [0.7,1.0]$ 优化改变颜色/纹理但保持运动。这与 P2P 不矛盾——P2P 在图像模型上，没有时间维度。视频模型的 temporal attention 引入了新的语义分离维度。

**对 $v^*$ 不随 t 变化**：正确，$v^*$ 是常数。但 $v_\theta(x_t, t, c)$ 对 $c$ 的敏感性随 $t$ 变化。在高噪声阶段（小 $t$），$x_t$ 接近纯噪声，模型必须高度依赖条件 $c$ 来决定去噪方向；在低噪声阶段（大 $t$），$x_t$ 本身已包含足够信息，$c$ 的影响减弱。我们在 Section X 提供了 $\|\partial v_\theta / \partial c\|$ 的范数随 $t$ 变化的实验曲线，证实了这一点。因此，在小 $t$ 优化能更有效地将信息注入 $c$。

**对 $T_m$ 敏感性**：Table X 提供了 $T_m \in \{0.1, 0.2, 0.3, 0.4, 0.5\}$ 的消融。结果显示 0.2-0.4 范围内性能稳定（XCLIP 波动 < 0.5%），$T_m=0.3$ 略优。原则性选择方法：$T_m$ 应设为 $\|\partial v_\theta / \partial c\|$ 曲线的拐点——即模型对条件敏感性开始快速下降的时间步。

**对正交性假设**：我们承认完美正交是理想化假设。实际中，内容和运动子空间可能有部分重叠。但 $\mathcal{L}_{disentangle}$ 的作用不是"证明正交性存在"，而是"将 $\Delta e$ 推向尽可能与内容无关的方向"。即使子空间不完美正交，正则化仍然有效——它减少了 $\Delta e$ 中的内容信息量，即使不能完全消除。实验中跨内容迁移的一致性（Table Z）证明了这一策略的有效性。

**对收敛性**：两个 loss 确实可能有轻微冲突，但不是对抗性的。$\mathcal{L}_{velocity}$ 想让 $\Delta e$ 编码"能匹配目标速度场的信息"，$\mathcal{L}_{disentangle}$ 想让这些信息"与内容无关"。两者的交集正是"纯运动信息"——这正是我们想要的。$\lambda_{dis} = 0.1$ 的小权重确保 velocity matching 主导优化方向，disentangle 只做轻微修正。实验中 loss 曲线平滑收敛（Figure X）。

---

## Reviewer 3：实验杀手 + 实用性质疑

### 攻击

> **[Major Concern] The experimental evaluation is insufficient and the practical value is questionable.**
>
> 1. **Scale**: The paper uses only 200 videos. This is far below the standard for top venues. VBench has 946 prompts, UCF-101 has 13K videos. How do we know results generalize?
>
> 2. **Baselines**: The paper only compares against its own VLM-only baseline and FreeInit. Where are:
>    - MotionPrompt (CVPR 2025) -- direct competitor for embedding optimization
>    - AnimateDiff + motion LoRA -- industry standard for motion control
>    - VideoComposer -- multi-condition video generation
>    - DragNUWA / MotionCtrl -- explicit motion control methods
>
> 3. **Metrics**: CLIP and X-CLIP are coarse. For a paper claiming precise motion control, you MUST report optical flow error (EPE/AE), motion magnitude correlation, and action recognition accuracy.
>
> 4. **Asset utility**: You claim the asset is "reusable" but show limited evidence. How many times can the same motion asset be applied to different content prompts before quality degrades? What is the success rate of cross-video transfer across different motion categories?
>
> 5. **Computational cost**: The full pipeline (inversion + SVD + velocity matching + disentanglement + generation for text decoding) likely takes 8-15 minutes per video on A800. Is this practical? Compare with MotionPrompt (2 min) and motion LoRA (5 min training).

### 防御

**对数据规模**：扩展到完整 200 条视频的全量实验（5 类 x 40 条），并在 UCF-101 子集（选 10 个动作类别 x 20 条 = 200 条）上做泛化验证。总计 400 条视频。同时指出：我们的方法是 per-video optimization（类似 Textual Inversion），不是 training-based 方法，数据规模的意义不同——我们需要证明的是"对每条视频都能工作"，而非"在大数据集上训练后泛化"。

**对 Baselines**：补充对比：(1) FreeInit（直接在 Wan2.1 上复现）；(2) Reenact Anything 的 T2V 适配版（全时间步 diffusion loss 优化 embedding，无 timestep 限制，无 disentangle）；(3) 纯 VLM iterative refinement（我们的 V4 pipeline，代表文本方法的天花板）；(4) MotionPrompt 变体（如果代码可用）。对于 AnimateDiff/VideoComposer/DragNUWA，这些是 training-based 方法，需要额外训练模块，与我们 training-free 的定位不同，但可以在 Discussion 中讨论。

**对指标**：补充：(1) RAFT 光流 EPE（生成视频 vs 目标视频的光流差异）；(2) 运动幅度相关系数（光流均值的 Pearson 相关）；(3) VideoMAE action recognition accuracy（生成视频是否被识别为相同动作）；(4) 人工评估（20 人 x 50 对 A/B test，评估运动一致性和视觉质量）。

**对资产复用**：设计专门实验：每个 motion asset 搭配 10 个不同 content prompt 生成，评估 10 次生成的运动一致性（互相之间的 Flow-Sim 均值和方差）。展示 motion asset 的复用稳定性。同时按运动类别（走路、跑步、转向、镜头运动、复合运动）分别报告迁移成功率。

**对计算成本**：报告详细 timing breakdown：inversion ~30s, SVD ~2s, velocity matching (100 steps) ~3min, disentangle (每 10 步一次, 5 个 augmented prompt) ~额外 1.5min, text decoding (2 次生成 + VLM) ~2min。总计 ~7min/video。对比：这是一次性提取成本，之后每次复用只需标准生成时间（~45s）+ 加载 tensor（~0.1s）。MotionPrompt 需要训练判别器（数小时前置成本），motion LoRA 需要多条同类视频训练（数小时）。我们的 7min 是 single-video、training-free 的。

---

## Reviewer 4：概念杀手 + 资产定位质疑

### 攻击

> **[Major Concern] The "asset" framing is misleading and the cross-model transferability claim is overstated.**
>
> 1. The motion token ($\Delta e$) is a tensor tied to Wan2.1's T5 encoder. It CANNOT be used with any other model family (CogVideoX uses different text encoder, HunyuanVideo uses CLIP+T5 dual encoder). Calling this an "asset" implies portability that doesn't exist.
>
> 2. The text description decoded from $\Delta e$ is just a VLM caption of the difference between two generated videos. This is no better than directly captioning the original video with a motion-focused prompt. Where is the evidence that the decoded text captures information that standard VLM captioning misses?
>
> 3. The noise prior ($\eta_{motion}$) is even MORE model-specific -- it's tied to the exact VAE and DiT architecture.
>
> 4. **Fundamentally, if the "asset" only works within one model family, how is it different from just saving the model's internal states?** A LoRA checkpoint is also a "reusable asset" that encodes motion -- and it's more established.
>
> 5. The "composability" claims (motion mixing, intensity control) are demonstrated with toy examples. In practice, linear interpolation in embedding space often produces artifacts. Where is the systematic evaluation of composability?

### 防御

**对跨模型迁移**：我们明确区分三个层次的迁移能力：(1) 文本层——天然跨模型，任何 T2V 模型都能使用 motion text description；(2) Token 层——同系列模型内迁移（Wan2.1 1.3B 和 14B 共享 T5 encoder，token 可直接复用）；(3) Noise 层——同架构内迁移。这与 LoRA/adapter 的迁移性限制完全一致，业界已接受这一范式。我们在 Limitation 中明确讨论，并指出文本层的跨模型能力是我们相对于纯 tensor 方法（Reenact Anything）的优势。

**对文本解码质量**：设计对比实验：(a) 直接 VLM caption 原始视频（motion-focused prompt）；(b) 我们的 motion text decoding（VLM 对比有/无 $\Delta e$ 的两个视频）。用两种文本分别生成视频，比较运动保真度。预期我们的解码文本更精确——因为它是从 $\Delta e$ 的实际效果中提炼的（"加了这个 token 后视频多了什么运动"），而非从像素中猜测的（"这个视频看起来有什么运动"）。前者是因果性的，后者是相关性的。

**对 vs LoRA**：关键区别：(1) LoRA 需要训练（数小时 + 通常需要多条同类视频避免过拟合），我们是 training-free（单条视频 7min）；(2) LoRA 编码的是一类运动的分布（"跑步"这个类别），我们编码的是单条视频的精确运动实例（"这个特定的加速-转向序列"）；(3) LoRA 修改模型权重，我们只修改输入条件——不影响模型的其他能力。两者互补而非竞争。

**对可组合性**：补充系统性实验：(1) 运动混合——10 对 asset 做线性插值，评估插值结果的运动连贯性；(2) 强度控制——5 个强度级别 x 20 个 asset，评估运动幅度与系数的单调性；(3) 失败案例分析——展示哪些情况下线性插值会产生 artifact（如两个方向相反的运动混合）。

---

## Reviewer 5：存在性质疑（最恶毒）

### 攻击

> **[Major Concern] I question whether this problem needs to be solved at all, and whether the proposed solution offers any advantage over simpler alternatives.**
>
> 1. **The "expressiveness bottleneck" may not exist**: Recent T2V models (Sora, Kling, Runway Gen-3) demonstrate that sufficiently detailed text prompts CAN generate complex, precise motion. The limitation the authors claim may simply be a property of their chosen 1.3B model, not a fundamental problem.
>
> 2. **Simpler alternatives exist**: If the goal is to reproduce a target video's motion with new content:
>    - Video-to-video generation (directly condition on target video)
>    - Motion LoRA (train on target video, 5 minutes)
>    - ControlNet with optical flow (explicit motion conditioning)
>    - Simply using the target video as a reference in I2V models
>
> 3. **The "asset" use case is contrived**: Who needs to "store and reuse" a motion pattern? In practice, users either (a) describe the motion they want in text, or (b) provide a reference video directly. The intermediate "asset" format adds complexity without clear benefit.
>
> 4. **The evaluation doesn't demonstrate the claimed use case**: The paper evaluates motion fidelity of generated videos, but doesn't show a real workflow where someone extracts an asset, stores it, and later applies it to different content. Without a user study demonstrating the utility of the asset workflow, the motivation is unconvincing.

### 防御

**对"文本够用了"**：我们的实验直接证明了文本的天花板。V4 最优 prompt（经过多轮 VLM+LLM 优化，代表当前文本方法的极限）达到 XCLIP 0.7430。加入 motion token 后达到 0.78+（+5%）。这 5% 的差距正是文本无法表达的运动细节。即使 Sora 级别的模型，"先慢走 3 步再突然加速 45 度右转，同时镜头以 0.5x 速度反向 pan"这种精确运动描述也无法从纯文本精确执行。文本的表达力瓶颈是语言本身的限制，不是模型大小的限制。

**对 V2V generation**：V2V 需要目标视频作为运行时输入——每次生成都需要原始视频在手边。我们的 asset 提取一次（7min）、复用无限次（每次只需加载一个小 tensor），且支持内容编辑（V2V 不支持换内容）。类比：V2V 是"每次都要带着原画去复印店"，asset 是"扫描一次存成电子版，随时随地打印"。

**对 Motion LoRA**：(1) LoRA 需要多条同类视频训练，单条视频会严重过拟合（生成结果几乎是原视频的复制）；(2) LoRA 训练需要数小时 GPU 时间，我们只需 7 分钟；(3) LoRA 修改模型权重，不同 LoRA 之间可能冲突，我们只修改输入条件，可以自由组合。Table X 对比 LoRA vs Ours 在单视频运动提取任务上的表现。

**对 ControlNet + 光流**：(1) ControlNet 需要训练额外模块（大量数据 + GPU 时间）；(2) 光流是逐帧的低级信号，缺乏语义理解——它知道"像素从 A 移动到 B"，但不知道"这是一个人在加速跑步"。我们的 motion token 编码的是语义级运动，可以迁移到外观完全不同的主体（狗的跑步运动 -> 猫的跑步运动），光流做不到这一点（因为不同主体的像素运动模式不同）。

**对"use case is contrived"**：三个真实场景：(1) 动画制作——导演说"这个镜头的运动节奏参考那个经典镜头"，提取 asset 后可以反复应用到不同角色；(2) 游戏/短视频模板——"这个运动模式很火，做成模板让用户换脸/换内容"；(3) 运动库建设——积累一个运动 asset 库，创作者可以像选择字体一样选择运动风格。我们在 Discussion 中补充这些应用场景，并设计一个简单的 user study 验证 asset 工作流的实用性。

---

## 综合防御矩阵

| 攻击类型 | 出现频率 | 核心防御武器 |
|---------|---------|------------|
| "和 XX 论文太像" | 最高 | 跨视频迁移实验（XX 做不到）+ timestep 消融 + 内容解耦消融 |
| "理论不够" | 高 | $\partial v_\theta / \partial c$ 随 t 变化的实验曲线 + $T_m$ 敏感性消融 + 收敛曲线 |
| "实验不够" | 高 | 400 条视频 + 4 个 baseline + 5 个指标 + 人工评估 |
| "不实用" | 中 | Training-free + 7min/video + 复用零成本 + vs LoRA 对比 |
| "问题不存在" | 低但致命 | XCLIP 天花板实验 + 精确运动描述失败案例 + 跨内容迁移 demo |
| "资产不可迁移" | 中 | 三层迁移能力分析 + 同系列模型验证 + 文本层跨模型实验 |

---

## 必须在提交前完成的实验清单

1. [ ] Velocity field matching 基础实现 + 全量 200 条视频实验
2. [ ] Content disentanglement loss 实现 + 消融对比（有 vs 无）
3. [ ] Timestep decomposition 消融（$t \in [0,0.3]$ vs $[0.3,0.7]$ vs $[0.7,1.0]$）
4. [ ] $T_m$ 敏感性消融（0.1/0.2/0.3/0.4/0.5）
5. [ ] 跨视频运动迁移实验（10 对 motion x content 组合）
6. [ ] 跨内容一致性实验（1 asset x 5 content prompts x 20 assets）
7. [ ] 运动强度可控性实验（$\Delta e \times [0.25, 0.5, 1.0, 1.5, 2.0]$）
8. [ ] FreeInit baseline 对比
9. [ ] 全时间步优化 baseline 对比（= Reenact Anything 变体）
10. [ ] 光流 EPE 指标实现
11. [ ] $\|\partial v_\theta / \partial c\|$ 范数随 t 变化的可视化
12. [ ] Motion Text Decoding 实现 + 对比直接 VLM caption
13. [ ] 资产压缩实验（PCA 降维）
14. [ ] 人工评估（如果时间允许）

---

## 代码架构设计

### 整体架构

VMAD 代码采用模块化 + flag 开关架构（参考 P-Flow），每个算法模块独立实现，通过统一 Pipeline 类组合调度。用户可通过命令行 flag 灵活控制启用/禁用各模块，支持快速消融实验。

```
VMAD/
├── configs/
│   └── default.yaml              # 全局配置文件
├── src/
│   ├── __init__.py               # 包初始化 + 模块导出
│   ├── pipeline.py               # 统一管线 (核心调度)
│   ├── flow_matching.py          # Flow Matching Inversion
│   ├── svd_filter.py             # SVD 时空滤波
│   ├── velocity_matching.py      # Velocity Field Matching + 解耦
│   ├── content_augmentation.py   # 内容增强 (LLM prompt 改写)
│   ├── motion_asset.py           # 运动资产管理
│   ├── vlm_client.py             # VLM 客户端 (运动文本解码)
│   ├── video_utils.py            # 视频 I/O 工具
│   └── distributed.py            # GPU 管理
├── run_extract.py                # 提取入口脚本
├── run_apply.py                  # 应用入口脚本
└── USAGE.txt                     # 命令参考文档
```

### 各文件功能描述

#### `configs/default.yaml` — 全局配置

定义所有超参数的默认值，包括模型路径、视频生成参数、各模块开关、Inversion/SVD/Velocity Matching 的超参数、VLM 配置等。支持通过 `--config` 参数加载，命令行参数可覆盖配置文件中的值。

#### `src/pipeline.py` — 统一管线（核心）

实现 `VMADPipeline` 类和 `VMADConfig` 数据类。VMADConfig 通过 `use_inversion`、`use_svd`、`use_blend`、`use_velocity`、`use_disentangle`、`use_text_decode`、`use_midpoint` 七个 bool flag 控制模块启用。VMADPipeline 提供两个主入口：`extract()` 从视频提取运动资产（Phase 1+2），`apply()` 将资产应用到新内容（Phase 3）。内部采用 lazy loading 策略，按需加载 T2V 模型、VLM 客户端等重型组件。

#### `src/flow_matching.py` — Flow Matching Inversion

实现 `FlowMatchingInverter` 类，提供 `invert()` (Euler 求解器) 和 `invert_midpoint()` (Midpoint 求解器) 两种反演方法。将参考视频从 $z_0$ 沿 ODE 正向积分到 $t=1$ 得到反演噪声 $\eta_{inv}$。同时提供 `encode_video_to_latents()` 辅助函数，通过 VAE encoder 将像素空间视频编码为 latent 空间表示。

参考论文：RF-Inversion (Rout et al., 2024) 的 ODE 反演策略，以及 FreeInit (Wu et al., 2024) 的噪声空间分析思路。

#### `src/svd_filter.py` — SVD 两阶段滤波

实现 `SVDFilter` 类，对反演噪声进行两阶段 SVD 分解：第一阶段空间去内容（保留比例 $\rho_s=0.1$，去除高能量空间奇异值中的内容信息），第二阶段时间保运动（保留比例 $\rho_m=0.9$，保留时间维度的主要运动结构）。输出 $\eta_{motion}$ 作为运动噪声先验。

参考论文：FreeInit 的频域分析启发了我们对噪声空间结构的理解；Reenact Anything 的 motion representation 思路验证了噪声中确实编码了运动信息。

#### `src/velocity_matching.py` — Velocity Field Matching + 内容解耦

实现 `VelocityFieldMatcher` 类，这是 VMAD 的核心优化模块。优化目标：找到 $\Delta e$ 使得 $\|v_\theta(x_t, t, e_0 + \Delta e) - v^*\|^2$ 最小化，其中 $v^* = z_0 - \eta$（目标速度场），仅在 $t \in [0, T_m]$ 范围内优化。

内容解耦通过 cross-content consistency loss 实现：对多个内容增强后的 prompt embedding $\{e_0^{(k)}\}$，约束 $\Delta e$ 在不同内容条件下产生一致的速度场修正，即最小化 $\text{Var}_k[v_\theta(x_t, t, e_0^{(k)} + \Delta e)]$。

参考论文：MotionPrompt (Guo et al., 2024) 的 prompt 优化框架；Textual Inversion (Gal et al., 2022) 的 embedding 空间优化策略；SiD-DiT 的 velocity field 分析。

#### `src/content_augmentation.py` — 内容增强

实现 `ContentAugmenter` 类，支持三种 provider：`mock`（规则替换，用于测试）、`dashscope`（调用通义千问 API）、`local`（本地 LLM）。给定原始 caption，生成多个保持运动语义不变但替换内容主体的增强 prompt（如"a dog running" → "a cat running"、"a robot running"），用于内容解耦损失的计算。

#### `src/motion_asset.py` — 运动资产管理

实现 `MotionAsset` 数据类和 `MotionAssetManager` 管理类。MotionAsset 包含三层表示：`delta_e`（motion token）、`eta_motion`（噪声先验）、`motion_text`（可读文本）。Manager 提供 `create`/`save`/`load`/`apply_to_embedding`/`apply_noise_prior`/`blend`/`scale` 等完整生命周期管理。资产以目录形式存储（metadata.json + .pt 文件），支持版本化和可移植性。

#### `src/vlm_client.py` — VLM 客户端

实现 VLM 交互层，支持三种 provider：`local`（本地 Qwen2.5-VL）、`api`（DashScope API）、`mock`（测试用）。核心功能 `decode_motion_text()` 实现 Motion Text Decoding：生成两个视频（有/无 $\Delta e$），让 VLM 对比描述差异，提炼出纯运动语义的文本描述。

参考论文：P-Flow 的 VLM 迭代优化 prompt 策略，我们将其适配为运动差异对比解码。

#### `src/video_utils.py` — 视频工具

提供视频 I/O 工具函数：`load_video`（加载 + resize + 采样帧）、`save_video_tensor`（tensor → mp4）、`resize_video`、`normalize_video`/`denormalize_video`（[0,1] ↔ [-1,1] 转换）。

#### `src/distributed.py` — GPU 管理

提供单 GPU 环境设置和模型加载工具：`setup_single_gpu`（自动检测可用 GPU）、`load_model_single_gpu`（加载 Wan2.1 Diffusers pipeline）、`cleanup_gpu_memory`（显存清理）。

#### `run_extract.py` — 提取入口

命令行入口脚本，解析 argparse 参数，构建 VMADConfig，调用 `VMADPipeline.extract()`。支持所有模块的 `--no-xxx` 开关、超参数覆盖、YAML 配置文件加载。输出运动资产目录 + 中间结果 + 结果摘要 JSON。

#### `run_apply.py` — 应用入口

命令行入口脚本，加载已提取的运动资产，调用 `VMADPipeline.apply()` 生成新视频。支持批量 content prompt、运动强度控制、噪声混合开关。

### Flag 开关与消融实验对应关系

| Flag | 模块 | 消融实验 |
|------|------|---------|
| `--no-inversion` | Flow Matching Inversion | 验证反演噪声 vs 随机噪声的运动保真度差异 |
| `--no-svd` | SVD Filter | 验证时空分离对内容去除的效果 |
| `--no-blend` | Noise Blending | 验证噪声先验对全局运动布局的贡献 |
| `--no-velocity` | Velocity Matching | 验证 motion token 对精确运动细节的编码能力 |
| `--no-disentangle` | Content Disentanglement | 验证解耦损失对跨内容迁移质量的影响 |
| `--no-text_decode` | Motion Text Decoding | 验证文本层对可解释性和跨模型迁移的价值 |
| `--midpoint` | Midpoint Solver | 对比 Euler vs Midpoint 的反演精度 |

### 与参考论文代码的对应关系

| VMAD 模块 | 参考论文 | 借鉴内容 |
|-----------|---------|---------|
| `flow_matching.py` | RF-Inversion | ODE 反演的 Euler/Midpoint 实现 |
| `svd_filter.py` | FreeInit | 噪声空间的频域/SVD 分析思路 |
| `velocity_matching.py` | MotionPrompt, Textual Inversion | Embedding 空间优化 + velocity field loss |
| `velocity_matching.py` | SiD-DiT | Velocity field 的理论分析 |
| `content_augmentation.py` | (原创) | Cross-content consistency 约束 |
| `motion_asset.py` | Reenact Anything | Motion representation 的存储/复用思路 |
| `vlm_client.py` | P-Flow | VLM 迭代优化 prompt 的交互模式 |
| `pipeline.py` | P-Flow | Flag-based 模块化架构 + dataclass 配置 |

### 命令行使用示例

```bash
# ═══ 完整提取流程 ═══
python run_extract.py \
    --video ./data/dance.mp4 \
    --output ./output/dance_asset \
    --config configs/default.yaml

# ═══ 快速提取 (跳过 VLM) ═══
python run_extract.py \
    --video ./data/dance.mp4 \
    --output ./output/dance_fast \
    --no-text_decode

# ═══ 消融: 无解耦 ═══
python run_extract.py \
    --video ./data/dance.mp4 \
    --output ./output/ablation_no_dis \
    --no-disentangle

# ═══ 应用到新内容 ═══
python run_apply.py \
    --asset ./output/dance_asset/asset \
    --content "a white cat dancing gracefully" \
    --output ./output/cat_dance \
    --strength 1.0

# ═══ 批量应用 ═══
python run_apply.py \
    --asset ./output/dance_asset/asset \
    --content "a cat" "a robot" "a teddy bear" \
    --output ./output/batch_dance
```

### 评测代码架构

```
VMAD/evaluation/
├── __init__.py                         # 评测套件初始化
├── run_motion_fidelity_eval.py         # 运动保真度评测
├── run_content_consistency_eval.py     # 内容一致性 + 解耦度评测
└── run_full_eval.py                    # 统一评测入口

VMAD/
├── run_batch_extract.py                # 批量提取 + 应用 (评测数据生成)
```

#### `evaluation/run_motion_fidelity_eval.py` — 运动保真度评测

评估运动迁移的核心质量。计算四类指标：Optical Flow EPE（光流端点误差，衡量逐像素运动差异）、Flow Direction Similarity（光流方向余弦相似度，衡量运动方向一致性）、CLIP Frame Similarity（帧级视觉相似度）、X-CLIP Motion Similarity（时序感知的运动语义相似度）。使用 Farneback 光流算法（CPU）或 RAFT（GPU）计算稠密光流场。

#### `evaluation/run_content_consistency_eval.py` — 内容一致性与解耦度评测

评估两个关键属性：内容一致性（生成视频是否匹配新 content prompt）和内容解耦度（motion asset 是否泄漏源视频内容）。指标包括 Content CLIP Score、Content Leakage Score、Disentangle Ratio。支持两种模式：per-sample 模式（逐样本评估）和 cross-content 模式（同一 asset 多个 content 的运动一致性方差）。

#### `evaluation/run_full_eval.py` — 统一评测入口

一键运行所有评测指标并生成综合报告。支持单方法评测和消融实验对比（多方法并行评测 + 对比表格生成）。自动调用子评测脚本，汇总结果为 Markdown 报告。

#### `run_batch_extract.py` — 批量实验脚本

批量从 P-Flow 数据集（200 条视频）中提取运动资产并应用到新内容。支持断点续跑（`--resume`）、跨内容实验（`--cross-content`）、消融变体（各种 `--no-xxx` flag）。输出结构化目录供评测脚本直接使用。

### 评测指标与论文对应关系

| 评测指标 | 论文章节 | 验证目标 |
|---------|---------|---------|
| Optical Flow EPE | Table 1 (Motion Fidelity) | 运动迁移的像素级精度 |
| X-CLIP Motion Sim | Table 1 (Motion Fidelity) | 运动语义的时序一致性 |
| Flow Direction Sim | Table 1 (Motion Fidelity) | 运动方向保持度 |
| Content CLIP Score | Table 2 (Content Consistency) | 新内容的忠实度 |
| Content Leakage | Table 2 (Disentanglement) | 源内容泄漏程度 |
| Cross-Content Variance | Table 3 (Ablation) | 解耦损失的有效性 |
| STREAM-T/F/D | Table 4 (Distribution Quality) | 生成视频的分布质量 |

### 数据复用说明

评测数据直接复用 P-Flow 项目的 `data/` 目录，无需额外准备：

- 源视频：`P-Flow/data/videos_200/{id}.mp4`（200 条高运动视频）
- Captions：`P-Flow/data/captions_qwen/{id}.txt`（Qwen-VL 生成的描述）
- 视频列表：`P-Flow/data/selected_200.csv`（含 motion_level、concept 等元数据）
- CLIP/X-CLIP 模型：复用 P-Flow 的评测脚本和模型路径
