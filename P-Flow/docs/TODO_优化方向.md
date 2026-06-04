# P-Flow 优化方向 TODO

> 创建时间: 2025-06  
> 最后更新: 2025-06-06  
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

## 已完成 TODO

### ✅ Velocity Matching v2 验证（2025-06-05）

- [x] 跑 30步 v2 实验 (K=4, motion_weight=1.0, padding_mask=ON, es=0.02)
- [x] Quick 验证（4 困难样本）：CLIP +1.5%, XCLIP +4.9%
- [x] 全量 10 样本验证：CLIP +0.2%, XCLIP +0.4%（提升有限）
- [x] 消融实验（motion_weight=0 on 3 退化样本）：证明 mw 非退化主因

**结论**: v2 对高运动样本有显著提升（#31 XCLIP +5.4%），但对低运动样本存在退化风险。整体均值提升边际，投入产出比不高。

---

## 当前 TODO 清单（优先级排序）

### 1. [优先] Per-sample es 自适应注入

**预期收益**: XCLIP +1-2%  
**成本**: 极低（改一行注入代码）  
**不需要重新跑 velocity matching 优化**

**背景**: 消融实验发现 L3 对不同样本效果相反。全局统一 es=0.02 对低运动样本注入了过多噪声（||Δe|| 小 = 信号弱，强制注入等于放大噪声）。

**方案**: 根据优化后 ||Δe|| 的大小自适应调整 es：
```python
# 方案 A: 阈值截断
es_adaptive = es_base * min(delta_e.norm().item() / threshold, 1.0)

# 方案 B: Norm-Aware（恒定注入量）
target_injection_norm = 0.17  # 最优区间
es_adaptive = target_injection_norm / delta_e.norm().item()
injection = es_adaptive * delta_e  # ||injection|| ≈ 0.17 (恒定)
```

**验证方式**: 用现有 v2 的 ||Δe|| 数据，只重新跑生成+评测（不需要重新优化 Δe），在全部 10 样本上验证。

### 2. [优先] L2 SVD 参数网格搜索（ρ_s × ρ_m）

**预期收益**: XCLIP +1-3%  
**成本**: 低（每组只跑生成，无额外 inversion）

**背景**: α=0.004 是 5 点网格搜索的最优值，但 SVD 的两个核心参数一直用默认值（ρ_s=0.1, ρ_m=0.9），从未搜索过。ρ_s 控制"去掉多少外观信息"，ρ_m 控制"保留多少运动成分"。

**搜索空间**: ρ_s ∈ {0.05, 0.1, 0.2} × ρ_m ∈ {0.7, 0.8, 0.9}（共 9 组）

**验证方式**: 固定 α=0.004，只跑 L1+L2（不加 L3），10 样本评测。找到最优 ρ_s/ρ_m 后再叠加 L3。

### 3. [中等] L1 迭代修复（matches 时跳过）

**预期收益**: CLIP +1-2%  
**成本**: 中等（需要重新跑生成 + VLM 推理）

**背景**: V4 iter1 是最佳 prompt，iter2/3 退化。原因是 VLM 报 4 维度全部 matches 时，LLM 仍强行修改引入噪声。

**方案**: 实现简单逻辑：
```python
if all(dim == "matches" for dim in vlm_feedback):
    skip_refine = True  # 保持当前 prompt 不变
```

只对仍有 mismatch 的样本（如 #21、#46）做 iter2 修复，已经 matches 的样本保持 iter1 prompt。

**预期**: 不伤害好的样本，只改善还有空间的样本。

### 4. [中等] 提升反演精度（Midpoint Solver）

**预期收益**: CLIP +0.5-1%  
**成本**: 低（Midpoint 已实现，2× forward 开销）

**背景**: 当前用 50 步 Euler inversion。RF-Solver 已被废弃（solver mismatch），但 Midpoint（2阶 Runge-Kutta）是安全的——它只是更精确地计算 η_inv，不改变 solver 类型。更精确的 η_inv 意味着 v* = z₀ - η_inv 更准，间接提升 L3 的 velocity matching 质量。

**注意**: Midpoint 使反演开销翻倍（50步 → 100 equivalent forwards），但只影响预处理阶段，不影响最终生成速度。

**验证方式**: 用 `--midpoint` flag 重新跑 inversion，然后正常跑 L3 + 生成 + 评测。

### 5. [必须做] 扩大样本量统计验证

**预期收益**: 不直接提升指标，但提供统计显著性  
**成本**: 高（30-50 样本 × 全 pipeline）

**背景**: 10 样本上 +0.4% 的提升不具统计意义。对于周会汇报和论文，需要在更大样本集上确认 pipeline 的稳定增益。

**方案**:
- [ ] 从 200 样本数据集中选取 30-50 个覆盖不同运动强度的样本
- [ ] 用当前最优配置（v1 更稳定，或 v2 + 自适应 es）跑全量
- [ ] 计算 95% 置信区间，paired t-test 检验 p < 0.05
- [ ] 按运动强度分组统计（高/中/低运动），分析 pipeline 的适用范围

---

## 长期探索方向

| 方向 | 思路 | 约束兼容性 | 优先级 | 参考 |
|------|------|-----------|--------|------|
| Per-Timestep Δe | 不同 ODE 步用不同 Δe，早期低频/晚期高频 | ✅ 只改 hook | P2 | Reenact Anything 2025 |
| η_inv 频域 mask | FFT low-pass 滤波，去除高频噪声保留运动结构 | ✅ 只改输入 | P2 | - |
| FlowEdit 式 inversion-free | 完全去除对 η_inv 精度的依赖 | ✅ 不改模型 | P3 | FlowEdit 2024 |
| ARPO 式 CLIP 引导迭代 | 多候选 prompt + CLIP 评分选最优 | ✅ 只改 prompt | P2 | ARPO 2025 |

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
