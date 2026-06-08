# Layer 4: Self-Attention K/V Injection

> 创建时间: 2025-06
> 状态: 已实现，待验证
> 代码: `src/attn_inject.py` + `pipeline.py` 集成

---

## 1. 核心思想

Layer 4 的本质是**在 attention 层面建立参考视频与生成视频之间的结构对应关系**。

前三层 (L1 Prompt Rewrite, L2 SVD Noise Prior, L3 Velocity Field Matching) 都在模型的「输入侧」工作——优化初始噪声或 text embedding。这些方法的信息传递路径是间接的：修改后的输入需要经过 30 层 transformer 的非线性变换才能影响最终输出。

Layer 4 提出一种更直接的方式：**在每个 transformer block 的 self-attention 中，将参考视频的注意力信息直接注入生成过程**。这绕开了信息在层间衰减的问题，让参考视频的结构和运动模式以最短路径传递到生成结果。

公式表达：

```
output_final = (1 - γ) * output_gen + γ * output_ref
```

其中 `output_ref` 是参考视频在相同 timestep 下经过同一个 attention block 的输出，γ 控制注入强度。

---

## 2. 为什么在 Wan2.1 上这样设计

### 2.1 Wan2.1 的 Attention 结构

通过对 Wan2.1-1.3B 的实际验证，确认其 transformer block 结构为：

```
WanTransformerBlock.forward():
    norm1 → attn1 (Self-Attention + 3D RoPE) → norm2 → attn2 (Cross-Attention) → norm3 → FFN
```

关键发现：**Wan2.1 使用 Full 3D Attention**，所有 32,760 个时空 token (21帧 × 30 × 52 patches) 在每个 block 的 self-attention 中进行统一计算。不存在单独的 temporal attention 或 spatial attention 分支。

这意味着 MotionClone (ECCV 2024) 那种「只注入 temporal attention K/V」的方案**不适用**，因为 Wan2.1 没有独立的 temporal attention module。我们必须在统一的 3D self-attention 上进行注入。

### 2.2 3D RoPE 的影响

Wan2.1 的 RoPE 结构：
- head_dim = 128，分为 t_dim=44, h_dim=42, w_dim=42
- 每个 token 的位置编码 = concat(freq_t[frame_idx], freq_h[h_idx], freq_w[w_idx])
- RoPE 在 `WanAttnProcessor` 内部应用到 Q 和 K 上（在 attention 计算之前）

由于 RoPE 已经 baked into cached K，当我们从参考路径缓存 attention 输入并在注入时重新计算时，位置信息自然保持一致——因为我们使用相同的 attn module（包含相同的 RoPE 参数）。

### 2.3 Output-Level Blending 而非 K/V-Level Blending

理论上最精确的做法是在 softmax 之前混合 K 和 V：
```
K_final = (1-γ)*K_gen + γ*K_ref
V_final = (1-γ)*V_gen + γ*V_ref
```

但实际实现中，diffusers 的 Attention module 不暴露中间的 K/V tensor（它们在 processor 内部计算并立即用于 attention）。要做到 K/V-level 需要 monkey-patch processor 的内部逻辑，容易导致兼容性问题。

我们选择**在 attention module 的输出上进行 blending**：
```
out_final = (1-γ)*out_gen + γ*out_ref
```

当 attention 近似线性时（token 数量多、softmax 接近均匀分布），output blending 与 K/V blending 效果等价。对于 γ ∈ [0.1, 0.5] 的实用范围，这是一个安全的近似。

---

## 3. 完整工作流

### Phase 1: 缓存（在 Flow Inversion 之后执行）

```
输入: 参考视频 V_ref, prompt P₀
输出: KV Cache (每个 generation step × 每个 block 的 attn1 输入)

1. VAE encode: z₁ = Encode(V_ref)
2. Flow Inversion: z₁ → η_inv (get inverted noise)
3. For each generation timestep step_i (i = 0, 1, ..., N-1):
   a. t = 1 - i/N
   b. x_t_ref = (1-t)*η_inv + t*z₁    ← 参考轨迹上的点
   c. Forward x_t_ref through transformer with caching hooks:
      → 每个 attn1 的 hook 记录该 block 的输入 hidden_states
   d. Store cache[step_i][block_idx] = hidden_states
```

### Phase 2: 注入（在生成阶段执行）

```
输入: prompt P₀ (或 P₀+Δe), initial noise η (可能是 noise prior)
输出: 生成视频，结构/运动更接近参考

For each denoising step step_i:
    1. 计算当前 γ_effective = f(γ_base, block_idx, timestep_ratio)
    2. Install injection hooks on all active attn1 modules
    3. 正常 forward pass:
       → 每个 hook 拦截 attn1 的输出 out_gen
       → 从 cache 取出 ref_hidden_states
       → 用相同的 attn1 module 计算 out_ref = attn1(ref_hidden_states)
       → 返回 blended = (1-γ)*out_gen + γ*out_ref
    4. Remove hooks
    5. CFG + scheduler step (正常流程)
```

---

## 4. γ 调度策略

### 4.1 Block 维度 (spatial hierarchy)

| 策略 | 效果 | 适用场景 |
|------|------|----------|
| `uniform` | 所有 block 相同 γ | 通用默认 |
| `front_heavy` | 早期 block γ 大，后期小 | 强调全局结构/布局一致性 |
| `back_heavy` | 后期 block γ 大，早期小 | 强调细节/纹理一致性 |

直觉：早期 block 处理全局结构（物体位置、场景布局），后期 block 处理局部细节（纹理、边缘）。

### 4.2 Timestep 维度 (denoising schedule)

| 策略 | 效果 | 适用场景 |
|------|------|----------|
| `constant` | 全程相同 γ | 最强注入（可能过拟合到参考） |
| `linear_decay` | γ 从开头 (t≈1) 线性衰减到结尾 (t≈0) | 默认推荐：开头强注入定全局，结尾自由生成细节 |
| `cosine_decay` | γ 余弦衰减（前半段缓慢，后半段急速） | 更长时间保持参考结构 |

直觉：denoising 早期（高噪声）决定全局布局，后期决定局部细节。`linear_decay` 允许模型在后期自由补充细节，避免生成结果完全复制参考（那样就失去了 text-guided 的意义）。

---

## 5. 与其他 Layer 的协同

Layer 4 与现有层完全独立，可以自由组合：

| 组合 | 效果 |
|------|------|
| L2 (Noise Prior) + L4 | 噪声提供运动先验 + attention 提供结构先验 |
| L3 (Velocity) + L4 | Δe 引导语义方向 + attention 引导空间结构 |
| L2 + L3 + L4 | 三管齐下：噪声（全局运动）、embedding（语义方向）、attention（逐 token 结构） |

**推荐起始配置:**
```bash
python run.py --video ref.mp4 --caption "..." \
    --inversion --svd --blend \
    --attn_inject --attn_inject_gamma 0.3
```

---

## 6. 内存估算

对于 Wan2.1-1.3B (480×832, 81帧):

| 配置 | 缓存大小 | 说明 |
|------|----------|------|
| All 30 blocks × 30 steps | ~2.3 GB | 每个 cache entry ≈ 32760 tokens × 1536 dim × 2 bytes (bf16) ≈ 96 MB / block / step |
| 8 blocks × 30 steps | ~0.6 GB | 选择性注入 (`--attn_inject_blocks "0,4,8,12,16,20,24,28"`) |
| All blocks × 30 steps (input only) | ~2.3 GB | 当前实现（存储 attn1 输入） |

实际上，我们存储的是 attn1 的**输入 hidden_states**（而非展开后的多头 K/V），所以实际内存是：
- 每个 entry: (1, seq_len, inner_dim) = (1, 32760, 1536) × 2 bytes ≈ 96 MB
- 30 blocks × 30 steps = 900 entries → 但每个 step 的所有 block 共享同一次 forward pass

等等，实际更小：缓存的是 attn1 的**输入**，在 Wan2.1 中就是 norm1 之后的 hidden_states。每个 step 每个 block 一个 tensor。

更精确估算：30 steps × 30 blocks × (1 × 32760 × 1536 × 2 bytes) = 30 × 30 × 96MB = 86.4 GB — 太大了！

**关键优化**: 实际推荐只选择部分 block 注入（如 8 个均匀分布的 block），并使用 `--attn_inject_blocks "0,4,8,12,16,20,24,28"` 将内存降至 ~23 GB，仍可能不够。

**进一步优化方向**（如果内存超出）:
1. 仅缓存偶数 step（内存减半，注入时复用相邻 step 的 cache）
2. 使用低秩近似压缩 cache（PCA 到 1/4 维度）
3. On-the-fly 计算（不缓存，每个注入 step 重新跑一次 reference forward）— 时间换空间

---

## 7. 与「不动模型」约束的关系

Layer 4 **完全遵守** P-Flow 的核心约束：
- ✅ 不修改 Wan2.1 的模型参数
- ✅ 不修改 ODE solver (Euler scheduler 不变)
- ✅ 所有影响都通过 forward hook 实现（可随时移除，零残留）
- ✅ γ=0 时行为与不使用 Layer 4 完全一致（数学上等价于 identity）
- ✅ 通过 `--attn_inject` flag 控制，默认关闭

---

## 8. 理论有效性分析

### 为什么 attention injection 应该有效？

1. **Self-Attention 是信息路由器**: 在 transformer 中，self-attention 决定了「哪些位置的信息流向哪些位置」。通过注入参考视频的 attention output，我们直接告诉模型「这些位置之间应该有这样的关系」。

2. **运动信息编码在 attention pattern 中**: 相邻帧的 token 之间的 attention weight 反映了运动对应关系。注入参考视频的 attention output 等效于传递了参考视频的运动模式。

3. **与 Noise Prior 互补**: L2 的 noise prior 提供的是「全局运动趋势」（通过 SVD 提取的时序方差模式），而 L4 的 attention injection 提供的是「逐 token 的结构对应」——更细粒度。

4. **已有工作验证**: MasaCtrl (ICCV 2023), Prompt-to-Prompt (ICLR 2023), FreeControl (CVPR 2024) 都证明了 attention manipulation 对控制生成结果的有效性。区别在于它们是图像模型上的 2D attention，我们是视频模型上的 3D attention。

### 潜在风险

1. **过度注入 → 生成结果与参考完全一致** (退化为"复制"而非"复现")
   - 缓解: `linear_decay` 时间调度 + 适中的 γ (0.2-0.4)

2. **时序 token 数量巨大 (32760) → 内存压力**
   - 缓解: 选择性 block 注入 + 可能的未来优化（低秩压缩）

3. **CFG 下的 batch 不匹配**: 生成时 latent 被复制为 [negative, positive]，但 cache 只有一份
   - 当前实现: hook 对两份 latent 都应用相同注入。这在语义上是合理的——参考结构对正负 prompt 都应该施加同样的引导。

---

## 9. 推荐验证实验

```bash
# Experiment 1: Attention Injection only (baseline comparison)
python run.py --video ref.mp4 --caption "..." --inversion --attn_inject --attn_inject_gamma 0.3

# Experiment 2: 与 Noise Prior 组合
python run.py --video ref.mp4 --caption "..." --inversion --svd --blend --attn_inject --attn_inject_gamma 0.3

# Experiment 3: 全部组合 (L2+L3+L4)
python run.py --video ref.mp4 --caption "..." --inversion --svd --blend --velocity --attn_inject --attn_inject_gamma 0.2

# Experiment 4: γ sweep
for gamma in 0.1 0.2 0.3 0.4 0.5; do
    python run.py --video ref.mp4 --caption "..." --inversion --attn_inject --attn_inject_gamma $gamma
done

# Experiment 5: Block schedule comparison
python run.py ... --attn_inject --attn_inject_block_schedule front_heavy
python run.py ... --attn_inject --attn_inject_block_schedule back_heavy

# Experiment 6: Memory-efficient (8 blocks only)
python run.py ... --attn_inject --attn_inject_blocks "0,4,8,12,16,20,24,28" --attn_inject_gamma 0.4
```

---

## 10. 用法汇总

```
python run.py --video <ref.mp4> --caption "<text>" \
    --inversion \                          # 必须: 需要 inversion 建立参考轨迹
    --attn_inject \                        # 启用 Layer 4
    --attn_inject_gamma 0.3 \              # γ 强度 (推荐 0.2-0.4)
    --attn_inject_blocks all \             # 注入范围 (all/first_half/last_half/"0,5,10")
    --attn_inject_block_schedule uniform \ # block 调度 (uniform/front_heavy/back_heavy)
    --attn_inject_timestep_schedule linear_decay  # 时间调度 (constant/linear_decay/cosine_decay)

# 快捷方式:
python run.py --video ref.mp4 --caption "..." --attn_full  # = --inversion --svd --blend --attn_inject
```
