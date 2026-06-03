# P-Flow 优化方向 TODO

> 创建时间: 2025-06  
> 最后更新: 2025-06-05  
> 目标: 在不修改 Wan2.1 模型和推理流程的前提下，通过输入侧优化（噪声、embedding）提升视频复现质量

---

## 核心约束

**P-Flow 的设计哲学是 "不动模型"**——所有优化只发生在输入侧（噪声先验、text embedding），Wan2.1 的模型参数和生成阶段的 ODE solver（Euler）不可修改。

这意味着：
- ✅ 可以改 inversion 阶段的 solver（因为 η_inv 只是一个输入）
- ✅ 可以优化 Δe（embedding 残差），只要最终通过 hook 注入
- ✅ 可以调整噪声混合策略（α、SVD 参数）
- ❌ 不能改生成阶段的 ODE solver（Wan2.1 pipeline 内部是 Euler，必须保持）
- ❌ 不能 fine-tune 模型参数

---

## 已完成实验总结

### embed_strength Grid Search ✅

| 方案 | CLIP | XCLIP | 说明 |
|------|------|-------|------|
| noise prior only (α=0.004) | 0.8953 | 0.7667 | baseline |
| velocity es=0.005 | 0.8990 | 0.7639 | 信号太弱 |
| **velocity es=0.02** | **0.8981** | **0.7705** | ✅ 当前最优 |
| velocity es=0.05 | 0.8965 | 0.7685 | 略过强 |

**结论**: es=0.02 是最优注入强度，XCLIP 首次超过 baseline（+0.38%）。

### 60步 Velocity Matching 实验 ✅ (2025-06-03)

| 配置 | ||Δe|| avg | 注入量 (es×||Δe||) | CLIP | XCLIP |
|------|-----------|-------------------|------|-------|
| 30步, es=0.02 | ~8.5 | 0.17 | **0.8981** | **0.7705** |
| 60步, es=0.02 | ~12.5 | 0.25 | 0.8935 | 0.7628 |

**结论**: 更多步数使 ||Δe|| 增大但指标反而下降。原因是注入过强 (0.25 > 最优区间 ~0.17)。同时观察到部分样本 loss 振荡（sample 31 final_loss=0.51），说明单时间步采样方差大、后期优化不稳定。

**启示**: 问题不是 "步数不够" 而是 "每步的优化效率低 + 能量分散"。

### Attention Pattern 分析 ✅ (2025-06-05)

在 Wan2.1 DiT 上 dump 了 30 层 cross-attention weights（5 样本 × 5 时间步），结论：

```
Position 0 relative weight: 2.01x mean  (阈值需 >3x 才算 sink)
→ ❌ Position 0 is NOT an attention sink in Wan2.1 DiT
```

**真正的 high-attention positions 在中后段**（对应 padding tokens）：

| Position | 相对权重 |
|----------|---------|
| pos 263 | 5.02x |
| pos 250 | 4.96x |
| pos 197 | 4.43x |
| pos 154 | 4.37x |
| pos 124 | 4.14x |
| pos 41 | 2.76x |
| pos 31 | 2.45x |
| pos 0 | 2.01x |

**关键发现**: Wan2.1 DiT 的 attention pattern 与 VMAD 假设的 U-Net "pos 0 attention sink" 完全不同。high-attention 集中在序列中后段（padding 区域），而非开头。这直接否定了 position-aware gradient scaling 的设计前提。

### Position-Aware + RF-Solver 联合实验 ❌ 已废弃

| 方案 | CLIP | XCLIP | 说明 |
|------|------|-------|------|
| full_opt (position_aware + rfsolver) | 0.8944 | 0.7508 | 全面退化 |

**失败根因**:
1. **RF-Solver solver mismatch**: 反演用 2 阶、生成用 Euler → η_inv 不是有效起点
2. **Position-Aware 假设错误**: Wan2.1 DiT 中 pos 0 不是 attention sink（仅 2x mean，需 >3x）
3. **梯度方向错误**: 原设计放大 pos 0 梯度、压制中段梯度，但实际高影响力位置在中后段

**决策**: 两者均已从代码中删除（2025-06-05）。

---

## ✅ 已实现 (待验证): Velocity Matching v2 三项改进

基于60步实验暴露的问题和论文调研，已在代码中实现以下三项互补改进：

### 改进 1: 分层多时间步采样 (Stratified Multi-Timestep Sampling)

**问题**: v1 每步只采 1 个随机时间步 t ∈ [0, T_m]，30步内只看 30 个 t 点，梯度方差大，导致：
- loss 振荡不收敛（60步实验中 sample 31/47 final_loss > 0.5）
- Δe 方向噪声大，需要更多步数补偿

**方案**: 将 [0, T_m] 分成 K=4 个等宽 bin，每 bin 内均匀采一个 t，每步看 4 个时间步。

**理论支撑**:
- CVPR 2025 "Adaptive Non-Uniform Timestep Sampling": 证明扩散模型训练中梯度方差跨时间步差异巨大，非均匀/分层采样可显著加速收敛
- NVIDIA CARV (2025): stratified-inverse-CDF 构造可额外带来 ~25% 方差下降
- 等效于将 30 步的时间覆盖密度提升到 120 步，但不增加优化迭代数

**代码**: `VelocityMatcher._sample_stratified_timesteps()`, 参数 `num_timesteps_per_step=K`

**显存影响**: 几乎为零。K=4 时只需循环 4 次 forward+backward（不是 batch 4 份 x_t），峰值显存不变。

### 改进 2: Padding-Aware 梯度 Mask

**问题**: v1 对 Δe 的全部 512 个位置统一优化，但真实 caption 只占 120-310 token（其余都是 padding）。能量被分散到无语义位置。

**方案**: 在 optimizer.step() 之前，对 padding 位置的梯度置零。只允许有效 token 区间的 Δe 更新。

**理论支撑**:
- NAACL 2025 "Padding Tone": 首次系统分析 padding tokens 在 T2I 模型中的作用。发现三种情况——padding 可能在 text encoding 阶段影响输出、在 diffusion cross-attention 时影响输出、或被完全忽略。Wan2.1 的 attention dump 显示 padding 区域有异常高权重 (4-5x)，说明模型在"看"这些位置但没有有意义的语义。
- Reenact Anything (SIGGRAPH 2025): 使用 inflated motion-text embedding，N tokens per frame，证明 token 数量和分配策略直接影响运动捕获质量

**代码**: `_compute_delta_e()` 调用 `_get_token_length()` 获取有效长度 → 传入 `matcher.optimize(token_length=...)` → 在每步 backward 后 `delta_e.grad.mul_(padding_mask)`

**预期效果**: Δe 能量集中在有语义的 token 上，等效的 per-token Δe 幅度更大，语义引导更强。

### 改进 3: 运动区域加权 Loss (Motion-Aware Weighting via LTD)

**问题**: v1 的 MSE loss 对所有像素、所有帧同等权重。但视频复现的核心是运动——静态背景即使 v_pred 差一点也不影响感知质量 (XCLIP 主要衡量运动一致性)。

**方案**: 计算 v* = z₀ - η_inv 的帧间方差 (temporal variance)，作为 per-pixel motion weight。高运动区域 (方差大) 获得更高 loss 权重。

**公式**:
```
σ²(x,y) = Var_f(v*[:,:,f,x,y])           # 帧间方差
w(x,y) = 1 + λ · log(1 + σ²(x,y) / μ)    # log 缩放防极端值
w_normalized = w / mean(w)                  # 归一化保持 loss 量级
L = mean(w · (v_pred - v*)²)               # 加权 MSE
```

**理论支撑**:
- "Latent Temporal Discrepancy as Motion Prior" (2025): 提出 LTD 度量帧间 latent 差异作为运动先验，指导 loss 加权。在不改架构的前提下，对高运动区域施加更大惩罚，对稳定区域常规优化。可应用于任何 diffusion video 框架。
- Motion Inversion (SIGGRAPH 2025): 通过 frame-to-frame debiasing 确保 embedding 主要编码运动动态，而非静态外观信息

**代码**: `VelocityMatcher._compute_motion_weight()`, 参数 `motion_weight_strength`

**预期效果**: Δe 优先拟合运动区域的 velocity field → XCLIP (运动一致性指标) 提升更明显。

---

## TODO 清单

### 1. [待做] 验证 Velocity Matching v2

**目的**: 确认三项改进是否带来指标提升

- [ ] 跑 30步 v2 实验 (K=4, motion_weight=1.0, padding_mask=ON, es=0.02)
- [ ] 对比 v1 baseline (30步, K=1, 无mask, 无加权, es=0.02): CLIP=0.8981, XCLIP=0.7705
- [ ] 逐项消融: (a) 只开 K=4 (b) 只开 mask (c) 只开 motion_weight (d) 全开
- [ ] 观察 ||Δe|| 变化 — mask 后 ||Δe|| 应该更小但 per-token 幅度更大

### 2. [待做] es 联合调参

**目的**: v2 改变了 Δe 的 norm 分布，最优 es 可能需要重新搜索

- [ ] 如果 v2 的 ||Δe|| 显著降低 (因为 mask 集中了能量)，es 可能需要上调
- [ ] 网格搜索: es ∈ {0.02, 0.03, 0.04} × v2 配置

### 3. [待做] 扩大样本量统计检验

**目的**: 当前 10 样本统计意义有限

- [ ] 扩到 30-50 样本
- [ ] 计算置信区间，确认改进的显著性（p < 0.05）

### 4. [探索] Adaptive Injection Strength

**背景**: 60步实验暴露的核心问题是 es × ||Δe|| 超出最优区间。

**方案**: 不用固定 es，而是在注入时自适应调整：
```python
target_injection_norm = 0.17  # 最优区间
adaptive_es = target_injection_norm / delta_e.norm().item()
injection = adaptive_es * delta_e  # ||injection|| ≈ 0.17 (恒定)
```

**优势**: 无论优化多少步、Δe 多大，最终注入量始终在最优区间。这也是一个很好的 paper 贡献点——"norm-aware adaptive injection"。

### 5. [探索] Δe Temporal Decomposition

**背景**: Reenact Anything 使用 inflated embedding (N tokens × F frames)，证明 per-frame token 对运动捕获至关重要。

**思路**: 当前 Δe 是 (1, L, D)——一个固定的 embedding 对所有时间步注入。可以探索 per-timestep Δe：
- 早期时间步 (t≈0, noise) 注入全局结构
- 晚期时间步 (t≈1, clean) 注入细节运动

**约束兼容性**: ✅ 只改 hook 逻辑，不改模型

---

## 长期方向（P2 优先级）

| 方向 | 思路 | 约束兼容性 | 参考 |
|------|------|-----------|------|
| η_inv 频域 mask | FFT low-pass 滤波，去除高频噪声保留运动结构 | ✅ 只改输入 | - |
| Δe 频率分解注入 | 早期步注入低频、晚期步注入高频 | ✅ 只改 hook 逻辑 | - |
| FlowEdit 式 inversion-free | 完全去除对 η_inv 精度的依赖 | ✅ 不改模型 | FlowEdit 2024 |
| Norm-Aware Adaptive Injection | es × ||Δe|| 恒定在最优区间 | ✅ 只改 hook | 本项目 60步实验 |
| Per-Timestep Δe | 不同 ODE 步用不同 Δe | ✅ 只改 hook | Reenact Anything 2025 |

---

## 已废弃方向

| 方向 | 废弃原因 | 日期 |
|------|----------|------|
| RF-Solver (2nd-order Taylor inversion) | 违反 "不改生成侧" 原则——反演用 2 阶但生成用 Euler，solver mismatch 导致 η_inv 不是有效起点 | 2025-06-05 |
| Position-Aware Gradient Scaling | Attention 分析证伪：Wan2.1 DiT 中 pos 0 仅 2.01x mean（需 >3x），不是 attention sink；实际 high-attention 在中后段 padding positions | 2025-06-05 |
| Token Decoding (Δe → text) | VMAD 已验证有害，直接舍弃 | 2025-06 |
| Content Disentanglement | P-Flow 的条件反演已天然产生纯净 η_inv，无需额外正则化 | 2025-06 |
| 60步 + es=0.02 | 注入过强 (0.25 > 最优 0.17)，指标全面下降。问题不在步数而在注入策略 | 2025-06-03 |

---

## 实现原则

1. **不动模型**: Wan2.1 的参数和生成 ODE solver 不可修改
2. **Additive Only**: 新功能通过 flag 控制，默认关闭
3. **Zero Regression**: 不修改已有方法的签名和行为
4. **Evidence-Based**: 所有新方向必须有理论支撑或实验验证后才实施
5. **Independent Testing**: 每个改动可独立消融验证

---

## 参考论文

| 论文 | 会议 | 与 P-Flow 的关系 |
|------|------|-----------------|
| Adaptive Non-Uniform Timestep Sampling | CVPR 2025 | 分层采样降低梯度方差 → 改进 1 |
| Latent Temporal Discrepancy as Motion Prior | 2025 | 运动加权 loss → 改进 3 |
| Padding Tone | NAACL 2025 | Padding tokens 在 T2I cross-attn 中的作用 → 改进 2 |
| Reenact Anything | SIGGRAPH 2025 | Motion-textual inversion, inflated embedding |
| Motion Inversion | SIGGRAPH 2025 | Frame-to-frame debiasing, motion embedding |
| SiD-DiT (Score Distillation of Flow Matching) | Apple 2025 | Velocity distillation in flow matching |
| CARV (Variance Reduction for Diffusion Teachers) | NVIDIA 2025 | Stratified-inverse-CDF, importance sampling |

---

## 参考资源

- Attention 分析脚本: `scripts/dump_cross_attention.py`
- Attention 分析结果: `/root/autodl-tmp/outputs/attention_analysis/attention_analysis.json`
- P-Flow 技术文档: `docs/P-Flow故事与技术演进.md`
- 实验记录: `docs/6-4周会.md`
- 60步实验日志: `/root/autodl-tmp/outputs/velocity_steps60/`
