# Flow-Matching Motion Distillation (FMMD)

## 从噪声空间中逆向提取视频运动 Prompt 的技术方案

> **一句话概括**：不让 VLM "看"视频写描述，而是通过速度场匹配，直接从噪声空间中"逆向蒸馏"出最优的运动 prompt embedding。

---

## 1. 问题定义

**输入**：目标视频 $V_{ref}$（我们想要复现其运动模式的视频）

**输出**：一个条件表示 $c_{final}$（prompt embedding）+ 一个初始噪声 $\eta_{init}$，使得 T2V 模型从 $\eta_{init}$ 出发、以 $c_{final}$ 为条件生成的视频，在运动模式上忠实于 $V_{ref}$

**核心挑战**：文本 prompt 无法精确描述复杂运动（如"先慢走再突然加速转身"），VLM captioning 只能捕获语义内容，丢失了运动的精确结构。

---

## 2. 核心 Insight

### 2.1 Rectified Flow 的速度场编码了"条件→视频"的完整映射

在 Rectified Flow 框架中，模型学习的是一个速度场 $v_\theta(x_t, t, c)$，它描述了"在条件 $c$ 下，从噪声到数据的最优传输路径"。

**关键观察**：如果我们知道目标视频对应的"理想速度场"$v^*$，那么通过最小化 $v_\theta(x_t, t, c)$ 与 $v^*$ 的差异，就能反向求解出"什么样的条件 $c$ 能让模型走向目标视频"。

这就是**从噪声空间中逆向提取 prompt** 的数学本质。

### 2.2 速度场在不同时间步的语义分离

- **早期时间步**（$t \in [0, 0.3]$，高噪声阶段）：速度场主要编码**运动结构**（全局动态、轨迹方向）
- **晚期时间步**（$t \in [0.7, 1.0]$，低噪声阶段）：速度场主要编码**内容细节**（纹理、外观、颜色）

这意味着我们可以**只在早期时间步做优化**，从而只提取运动信息，不干扰内容。

### 2.3 噪声空间与条件空间的互补性

| 空间 | 编码的信息 | 优势 | 劣势 |
|------|-----------|------|------|
| 噪声空间（$\eta$） | 全局运动结构、空间布局 | 精确、无损 | 缺乏语义可控性 |
| 条件空间（$c$） | 语义内容、细粒度运动语义 | 可解释、可编辑 | 文本表达力有限 |

两者协同使用，可以实现对视频生成的精确运动控制。

---

## 3. 方法详述

### 3.1 整体 Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    FMMD Pipeline                                  │
│                                                                   │
│  V_ref ──→ VAE Encode ──→ z₀                                    │
│    │                        │                                     │
│    │                        ├──→ Flow Matching Inversion ──→ η_inv│
│    │                        │         │                           │
│    │                        │         ▼                           │
│    │                        │    SVD Temporal Filter ──→ η_motion │
│    │                        │         │                           │
│    │                        │         ▼                           │
│    │                        │    Noise Blending ──→ η_init        │
│    │                        │                                     │
│    │                        └──→ Target Velocity Field: v* = z₀-η │
│    │                                   │                          │
│    ▼                                   ▼                          │
│  VLM Caption ──→ e₀    Velocity Field Matching                   │
│                  │       min ||v_θ(x_t,t,e₀+Δe) - v*||²         │
│                  │              │                                  │
│                  │              ▼                                  │
│                  │         Δe_motion                               │
│                  │              │                                  │
│                  └──────┬───────┘                                  │
│                         ▼                                         │
│                  c_final = e₀ + Δe_motion                         │
│                         │                                         │
│                         ▼                                         │
│              Generate(v_θ, η_init, c_final) ──→ V_gen             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Step 1: Flow Matching Inversion（噪声反演）

将目标视频编码到噪声空间：

$$z_0 = \text{VAE\_Encode}(V_{ref})$$

$$\eta_{inv} = \text{ODE\_Solve}(z_0 \to z_1, \quad \frac{dx_t}{dt} = v_\theta(x_t, t, \varnothing))$$

其中 $\varnothing$ 表示无条件（null prompt），ODE 从 $t=0$ 积分到 $t=1$。

**实现细节**：使用 Euler 方法，50 步积分。

**参考论文**：
- RF-Inversion (ICLR 2025) — Rectified Flow 反演的最优控制理论
- RF-Solver (2024) — 高阶 Rectified Flow 求解器

### 3.3 Step 2: SVD Temporal Filter（运动先验提取）

对反演噪声做时域低频滤波，提取运动结构：

$$\eta_{inv} \in \mathbb{R}^{F \times C \times H \times W}$$

**空间滤波**（去除空间高频噪声）：

$$\eta_{inv}^{(f)} = \text{reshape}(\eta_{inv}[f], [C, H \times W])$$
$$U, S, V^T = \text{SVD}(\eta_{inv}^{(f)})$$
$$S_{spatial} = S \cdot \mathbb{1}[i > \lfloor \rho_s \cdot \min(C, HW) \rfloor]$$
$$\eta_{spatial}^{(f)} = U \cdot \text{diag}(S_{spatial}) \cdot V^T$$

**时域滤波**（保留时域低频 = 运动信息）：

$$\eta_{spatial} = \text{reshape}(\eta_{spatial}, [F, C \times H \times W])$$
$$U_t, S_t, V_t^T = \text{SVD}(\eta_{spatial})$$
$$S_{motion} = S_t \cdot \mathbb{1}[i \leq \lfloor \rho_m \cdot \min(F, CHW) \rfloor]$$
$$\eta_{motion} = U_t \cdot \text{diag}(S_{motion}) \cdot V_t^T$$

**超参数**：$\rho_s = 0.1$（空间滤波比例），$\rho_m = 0.9$（时域保留比例）

**噪声混合**：

$$\eta_{init} = \sqrt{\alpha} \cdot \eta_{motion} + \sqrt{1-\alpha} \cdot \eta_{random}, \quad \alpha = 0.001$$

**参考论文**：
- FreeInit (ECCV 2024) — 时域低频噪声编码运动信息
- Seeds of Structure (NeurIPS 2025) — 噪声低频分量编码全局构图

### 3.4 Step 3: Velocity Field Matching（核心——从速度场中逆向提取 prompt）

这是整个方法的核心创新。我们通过匹配速度场来"逆向"出最优的条件 embedding。

**目标速度场构造**：

在 Rectified Flow 中，数据 $z_0$ 和噪声 $\eta$ 之间的线性插值路径为：

$$x_t = (1-t) \cdot \eta + t \cdot z_0$$

对应的目标速度场为：

$$v^*(t) = z_0 - \eta$$

注意：这里 $\eta$ 可以选择 $\eta_{inv}$（反演噪声）或 $\eta_{random}$（随机噪声）。使用 $\eta_{inv}$ 时目标速度场更精确。

**条件 Embedding 优化**：

初始化：
$$e_0 = \text{TextEncoder}(\text{VLM\_Caption}(V_{ref}))$$

引入可学习残差：
$$e = e_0 + \Delta e$$

优化目标（**只在运动相关的时间步范围内优化**）：

$$\mathcal{L}_{motion} = \mathbb{E}_{t \sim \mathcal{U}(0, T_m)} \left[ \| v_\theta(x_t, t, e_0 + \Delta e) - v^*(t) \|^2 \right]$$

其中 $T_m = 0.3$ 为运动相关的时间步上界。

**优化过程**：
- 冻结模型 $v_\theta$ 的所有参数
- 只更新 $\Delta e$
- 使用 Adam 优化器，lr = 1e-3
- 迭代 K = 50~100 步

**最终条件**：

$$c_{final} = e_0 + \Delta e_{motion}$$

**参考论文**：
- Score Distillation of Flow Matching Models (Apple Research, 2025) — 证明 score distillation 适用于 flow matching，速度场匹配的理论基础
- Reenact Anything (SIGGRAPH 2025) — Motion-Textual Inversion，在 I2V 模型中优化 motion embedding
- MotionPrompt (CVPR 2025) — 通过光流判别器梯度优化 learnable token embedding

### 3.5 Step 4: Generation（生成）

$$V_{gen} = \text{Sample}(v_\theta, \eta_{init}, c_{final}, \text{steps}=50)$$

使用标准的 Euler 采样，从 $t=1$（噪声）到 $t=0$（数据）。

---

## 4. 数学形式化总结

### 4.1 完整优化问题

$$\min_{\Delta e} \quad \mathbb{E}_{t \sim \mathcal{U}(0, T_m), \eta \sim \mathcal{N}(0,I)} \left[ \| v_\theta(x_t, t, e_0 + \Delta e) - (z_0 - \eta) \|^2 \right]$$

$$\text{s.t.} \quad x_t = (1-t) \cdot \eta + t \cdot z_0, \quad z_0 = \text{Enc}(V_{ref})$$

### 4.2 与 Score Distillation Sampling (SDS) 的关系

SDS 的优化目标：
$$\nabla_\theta \mathcal{L}_{SDS} = \mathbb{E}_{t,\epsilon} \left[ w(t) (\epsilon_\phi(x_t, t, y) - \epsilon) \frac{\partial x}{\partial \theta} \right]$$

我们的优化目标（Flow Matching 版本）：
$$\nabla_{\Delta e} \mathcal{L}_{FMMD} = \mathbb{E}_{t} \left[ (v_\theta(x_t, t, e_0+\Delta e) - v^*) \frac{\partial v_\theta}{\partial \Delta e} \right]$$

**关键区别**：
- SDS 优化的是 3D 表示（NeRF 参数），我们优化的是条件 embedding
- SDS 用 score function（$\epsilon$-prediction），我们用 velocity field（$v$-prediction）
- SDS 没有 ground truth target，我们有明确的 $v^* = z_0 - \eta$（来自目标视频）
- 我们限制在 $t \in [0, T_m]$ 范围内优化，实现运动-内容分离

### 4.3 与 Textual Inversion 的关系

Textual Inversion 的优化目标：
$$\min_{v^*} \mathbb{E}_{t,\epsilon} \left[ \| \epsilon_\theta(x_t, t, c_{v^*}) - \epsilon \|^2 \right]$$

我们的优化目标：
$$\min_{\Delta e} \mathbb{E}_{t} \left[ \| v_\theta(x_t, t, e_0 + \Delta e) - v^* \|^2 \right]$$

**关键区别**：
- Textual Inversion 优化的是完整的 token embedding（编码外观+内容），我们只优化残差 $\Delta e$（只编码运动）
- Textual Inversion 在所有时间步优化，我们只在早期时间步优化（运动-内容分离）
- Textual Inversion 用 diffusion loss，我们用 velocity field loss（适配 Rectified Flow 架构）

---

## 5. 实现伪代码

```python
import torch
from model import WanT2V, VAEEncoder, TextEncoder, VLM

def fmmd_pipeline(video_ref, model, vae, text_enc, vlm, 
                  num_opt_steps=100, lr=1e-3, T_m=0.3, alpha=0.001,
                  rho_s=0.1, rho_m=0.9):
    """
    Flow-Matching Motion Distillation Pipeline
    
    Args:
        video_ref: 目标视频 [F, 3, H, W]
        model: 预训练 T2V 模型 (Wan2.1)
        T_m: 运动相关时间步上界
        alpha: 噪声混合权重
    
    Returns:
        c_final: 最终条件 embedding
        eta_init: 最终初始噪声
    """
    
    # ============ Step 1: Encode & Invert ============
    # VAE 编码
    z0 = vae.encode(video_ref)  # [F, C, h, w]
    
    # Flow Matching Inversion (Euler ODE, t: 0→1)
    eta_inv = flow_matching_inversion(model, z0, steps=50)  # [F, C, h, w]
    
    # ============ Step 2: SVD Temporal Filter ============
    eta_motion = svd_temporal_filter(eta_inv, rho_s=rho_s, rho_m=rho_m)
    
    # Noise Blending
    eta_random = torch.randn_like(eta_motion)
    eta_init = (alpha**0.5) * eta_motion + ((1-alpha)**0.5) * eta_random
    
    # ============ Step 3: VLM Caption → Initial Embedding ============
    caption = vlm.describe(video_ref)  # "A person walking in the park..."
    e0 = text_enc.encode(caption)  # [L, D]
    
    # ============ Step 4: Velocity Field Matching ============
    # 初始化可学习残差
    delta_e = torch.zeros_like(e0, requires_grad=True)
    optimizer = torch.optim.Adam([delta_e], lr=lr)
    
    for step in range(num_opt_steps):
        # 随机采样时间步（只在运动相关范围内）
        t = torch.rand(1) * T_m  # t ∈ [0, T_m]
        
        # 随机采样噪声（或使用 eta_inv）
        eta = torch.randn_like(z0)
        
        # 构造插值点
        x_t = (1 - t) * eta + t * z0  # [F, C, h, w]
        
        # 目标速度场
        v_target = z0 - eta  # [F, C, h, w]
        
        # 模型预测速度场（使用当前条件）
        e_current = e0 + delta_e
        v_pred = model.forward(x_t, t, e_current)  # [F, C, h, w]
        
        # 速度场匹配损失
        loss = ((v_pred - v_target) ** 2).mean()
        
        # 反向传播，只更新 delta_e
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # ============ Step 5: 组装最终条件 ============
    c_final = e0 + delta_e.detach()
    
    return c_final, eta_init


def flow_matching_inversion(model, z0, steps=50):
    """
    Flow Matching Inversion: z0 (t=0) → η (t=1)
    使用 Euler 方法沿 ODE 正向积分
    """
    dt = 1.0 / steps
    x = z0.clone()
    
    for i in range(steps):
        t = torch.tensor(i * dt)
        # 无条件速度场
        v = model.forward(x, t, null_prompt)
        x = x + v * dt
    
    return x  # η_inv at t=1


def svd_temporal_filter(eta, rho_s=0.1, rho_m=0.9):
    """
    SVD 两阶段滤波：空间去噪 + 时域保留运动
    
    Args:
        eta: [F, C, H, W] 反演噪声
        rho_s: 空间滤波比例（去除前 rho_s 的奇异值）
        rho_m: 时域保留比例（保留前 rho_m 的奇异值）
    """
    F, C, H, W = eta.shape
    
    # Stage 1: 空间滤波（逐帧）
    eta_spatial = torch.zeros_like(eta)
    for f in range(F):
        frame = eta[f].reshape(C, H * W)  # [C, HW]
        U, S, Vh = torch.linalg.svd(frame, full_matrices=False)
        # 去除前 rho_s 比例的奇异值（去除空间高频内容信息）
        k = int(rho_s * min(C, H * W))
        S[:k] = 0
        eta_spatial[f] = (U @ torch.diag(S) @ Vh).reshape(C, H, W)
    
    # Stage 2: 时域滤波
    temporal = eta_spatial.reshape(F, C * H * W)  # [F, CHW]
    U_t, S_t, Vh_t = torch.linalg.svd(temporal, full_matrices=False)
    # 保留前 rho_m 比例的奇异值（保留时域低频 = 运动）
    k_m = int(rho_m * min(F, C * H * W))
    S_t[k_m:] = 0
    eta_motion = (U_t @ torch.diag(S_t) @ Vh_t).reshape(F, C, H, W)
    
    return eta_motion


def generate(model, eta_init, c_final, steps=50):
    """
    标准 Flow Matching 采样: η (t=1) → z0 (t=0)
    """
    dt = -1.0 / steps
    x = eta_init.clone()
    
    for i in range(steps):
        t = torch.tensor(1.0 + i * dt)
        v = model.forward(x, t, c_final)
        x = x + v * dt
    
    return x  # z0_gen at t=0
```

---

## 6. 与现有工作的关系和区别

### 6.1 对比表

| 方法 | 优化对象 | 监督信号 | 运动-内容分离 | 模型类型 | 发表 |
|------|---------|---------|-------------|---------|------|
| Textual Inversion | token embedding | diffusion loss (全时间步) | ✗ | T2I | ICLR 2023 |
| Reenact Anything | motion embedding | diffusion loss (全时间步) | 隐式（依赖 I2V 结构） | I2V | SIGGRAPH 2025 |
| MotionPrompt | learnable tokens | 光流判别器梯度 | ✗（需训练判别器） | T2V | CVPR 2025 |
| Reverse Stable Diffusion | — (训练预测网络) | 回归 loss | ✗ | T2I | IJCV 2024 |
| Promptus | prompt embedding | 重建 loss | ✗ | T2I (逐帧) | AAAI 2025 |
| **FMMD (Ours)** | **embedding 残差 Δe** | **velocity field loss (限制时间步)** | **✓ (timestep-aware)** | **T2V** | — |

### 6.2 核心创新点

1. **Velocity Field Matching for Condition Inversion**：首次将 flow matching 的速度场匹配用于条件 embedding 的逆向优化。不同于 SDS（无 ground truth target）和 Textual Inversion（diffusion loss），我们有明确的目标速度场 $v^* = z_0 - \eta$。

2. **Timestep-Aware Motion-Content Decomposition**：通过限制优化的时间步范围 $t \in [0, T_m]$，实现了运动信息和内容信息的显式分离。这是基于"早期时间步编码运动、晚期时间步编码内容"的实证发现。

3. **Dual-Space Encoding**：噪声空间（SVD 滤波后的 $\eta_{motion}$）编码全局运动结构，条件空间（$\Delta e_{motion}$）编码细粒度运动语义，两者互补。

---

## 7. 实验设计

### 7.1 消融实验

| 配置 | 噪声先验 | Δe 优化 | 预期效果 |
|------|---------|---------|---------|
| Baseline | ✗ | ✗ | 纯 VLM caption 生成 |
| + Noise Prior | ✓ | ✗ | 全局运动结构改善 |
| + FMMD | ✗ | ✓ | 细粒度运动语义改善 |
| + Both (Full) | ✓ | ✓ | 最佳运动忠实度 |

### 7.2 关键验证实验

1. **Timestep Decomposition 验证**：
   - 固定 $\Delta e$，分别在 $t \in [0, 0.3]$ 和 $t \in [0.7, 1.0]$ 优化
   - 展示前者改变运动、后者改变外观

2. **跨视频迁移**：
   - 用 A 视频的 $\Delta e_{motion}$ + B 视频的 VLM caption
   - 生成视频应有 A 的运动 + B 的内容

3. **噪声频谱分析可视化**：
   - 画出 SVD 奇异值分布
   - 标注运动/内容的分界线

### 7.3 评估指标

- **运动忠实度**：光流相似度 (Flow-Sim)、X-CLIP temporal score
- **内容质量**：CLIP-Sim、FID-VID
- **整体质量**：FVD、Human Evaluation
- **效率**：优化时间、GPU 显存

---

## 8. 参考论文完整列表

### 核心参考（方法直接相关）

1. **Score Distillation of Flow Matching Models** — Apple Research, 2025
   - 证明 score distillation 适用于 flow matching 模型
   - 我们方法的理论基础：velocity field matching ≈ score distillation for RF

2. **Reenact Anything: Semantic Video Motion Transfer Using Motion-Textual Inversion** — SIGGRAPH 2025, Disney Research × ETH Zurich
   - Motion-Textual Inversion 的开创性工作
   - 我们的区别：T2V（非 I2V）、velocity field loss（非 diffusion loss）、timestep-aware 分离

3. **MotionPrompt: Optical-Flow Guided Prompt Optimization for Coherent Video Generation** — CVPR 2025, KAIST
   - 通过优化 learnable token 控制视频运动
   - 我们的区别：不需要训练判别器，直接用目标视频的速度场作为监督

4. **RF-Inversion: Semantic Image Inversion and Editing using Rectified Stochastic Differential Equations** — ICLR 2025
   - Rectified Flow 反演的最优控制理论
   - 我们 Flow Matching Inversion 步骤的理论基础

5. **FreeInit: Bridging Initialization Gap in Video Diffusion Models** — ECCV 2024
   - 发现时域低频噪声编码运动信息
   - 我们 SVD 时域滤波的理论依据

### 辅助参考（理论支撑）

6. **Reverse Stable Diffusion: What prompt was used to generate this image?** — IJCV 2024
   - 从生成图像中预测 prompt embedding 的开创性工作
   - 证明了"逆向提取 prompt"的可行性

7. **Promptus: Can Prompt Streaming Replace Video Streaming with Stable Diffusion** — AAAI 2025
   - 将视频帧反演为 prompt 表示
   - 证明了视频级别的 prompt inversion 是可行的

8. **Uncovering the Text Embedding in Text-to-Image Diffusion Models** — arXiv 2024
   - 系统分析了 text embedding 空间的结构
   - 证明 embedding 空间存在可解释的语义方向

9. **Seeds of Structure: Patch PCA Reveals Universal Compositional Cues in Diffusion Noise** — NeurIPS 2025
   - 噪声低频分量编码通用构图蓝图
   - 支撑我们"噪声空间包含结构信息"的论点

10. **The Crystal Ball Hypothesis in Diffusion Models** — ICLR 2025
    - 噪声中的 trigger patches 决定物体位置
    - 支撑我们"噪声空间包含布局信息"的论点

11. **Beyond Randomness: Understand the Order of the Noise in Diffusion** — arXiv 2025
    - 噪声天然编码语义模式
    - 支撑我们"噪声不是纯随机"的核心论点

12. **How I Warped Your Noise: A Temporally-Correlated Noise Prior for Diffusion Models** — ICLR 2025
    - 时间相关噪声保持帧间一致性
    - 支撑我们噪声先验的时域相关性设计

---

## 9. 预期贡献总结

1. **方法贡献**：提出 Flow-Matching Motion Distillation (FMMD)，首次通过速度场匹配从目标视频中逆向蒸馏出运动 prompt embedding，实现了"从噪声空间中选出 prompt"的效果。

2. **理论贡献**：揭示了 Rectified Flow 速度场在不同时间步的语义分离特性（早期=运动，晚期=内容），并基于此提出 timestep-aware 的运动-内容解耦优化策略。

3. **系统贡献**：提出 Dual-Space Encoding 框架，在噪声空间和条件空间中协同编码视频信息，两者互补地引导 T2V 生成。

4. **实验贡献**：通过系统的消融实验和跨视频迁移实验，验证了噪声空间与条件空间编码信息的正交互补性。
