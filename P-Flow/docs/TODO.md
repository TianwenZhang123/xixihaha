# P-Flow 后续实验计划

> **更新时间**：2026-06-16
>
> **当前最强配置**：L2(SVD v1 α=0.004) + L3(FI adaptive mid λ=0.05) → CLIP 0.9042, XCLIP 0.8138（vs Baseline +3.9%/+13.6%）

---

## 已完成的关键实验

| 实验 | CLIP | XCLIP | 结论 |
|------|------|-------|------|
| Baseline（裸 VLM caption，无优化） | 0.8703 | 0.7164 | — |
| Pure L2（SVD v1 α=0.004） | 0.8964 | 0.7874 | 黑盒最优，+2.4%/+5.1% |
| Pure L3（Anchor V3 gated） | 0.8845 | 0.7810 | 灰盒独立，+1.6%/+9.1% |
| 纯 FI（无 SVD） | 0.8819 | 0.7669 | FI 可独立工作但弱于 SVD+FI |
| **SVD+FI（当前最强）** | **0.9042** | **0.8138** | **+3.9%/+13.6%，L2+L3 正交互补确认** |
| v9+SVD | 0.8947 | 0.7973 | L1(v9)与L2兼容，XCLIP +1.3% |
| 代码清理验证（3样本） | 0.9296 | 0.8096 | 清理后逻辑等价，±0.02 波动正常 |

---

## 待做实验

### 方向A：扩大评测规模 ⭐⭐⭐

**目的**：所有实验只跑了 10 样本（验证甚至只有 3 样本），置信度低。需要可靠基准。

**操作**：
```bash
# 200 样本 L2+L3(FI) 基准评测
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/FI_200samples \
    --inversion --svd --blend --alpha 0.004 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose --resume
```

**预计耗时**：200 × ~260s ≈ 14.4 小时

---

### 方向B：验证 SVD 是否冗余 ✅ 已完成

**结论**：SVD 不冗余。SVD+FI 比纯 FI 在 CLIP 上高 2.5%，XCLIP 上高 5.8%。论文用方案A故事（三层正交互补架构）。

| 配置 | CLIP | XCLIP |
|------|------|-------|
| 纯 FI (no SVD) | 0.8819 | 0.7669 |
| SVD+FI | 0.9042 | 0.8138 |

**关键发现**：noSVD 模式下 6/10 样本 quality_scale=1.0（无门控），FI 注入缺乏样本级自适应，复杂样本（如 S21）XCLIP 崩塌到 0.61。

---

### 方向C：L1+L2+L3 三层全叠加 ⭐⭐⭐

**目的**：验证 v9(删减+VLM补充) + SVD + FI 的三层全叠加效果。v9 不碰运动描述 → 理论上与 FI 不冲突。

**操作**：
```bash
# 1. 先生成 v9 改写后的 caption（如尚未生成）
python scripts/rewrite_minimal.py \
    --input-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir data/captions_v9 \
    --video-dir data/videos \
    --backend dashscope --model qwen-plus \
    --vlm-provider dashscope --vlm-model qwen-vl-max \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --skip-existing

# 2. 用 v9 caption + SVD + FI 生成
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/v9_SVD_FI_10samples \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --inversion --svd --blend --alpha 0.004 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose
```

**预期**：
- v9 不碰运动 → 不与 FI 的特征空间运动引导冲突
- v9 补充视觉细节 → CLIP 可能提升（外观更准确）
- XCLIP 变化不确定：v9 的视觉细节补充可能让 FI 的特征对齐度更高（cos ↑ → adaptive_gate ↓），注入量自动减少

**判断标准**：
- 如果 CLIP > 0.9042 且 XCLIP ≥ 0.8138 → 三层成功叠加
- 如果 XCLIP < 0.8138 → v9 的视觉补充与 FI 的特征引导存在某种干扰

---

### 方向D：FI 精进实验 ⭐⭐

#### D1：EMA 跨步特征平滑（优先做，改动极小）

**问题**：每步独立取 ref_features[step_idx]，相邻步的参考特征可能不连续，导致注入信号跳变。

**方案**：在生成阶段对缓释的参考特征做时序 EMA 平滑

```python
# 当前：直接取当前步的参考特征
h_ref = ref_features[step_idx]

# 新方案：EMA 平滑（只在生成阶段做，不改缓存）
if step_idx == 0:
    h_ref_smooth = ref_features[0]
else:
    h_ref_smooth = ema_decay * h_ref_prev + (1 - ema_decay) * ref_features[step_idx]
h_ref_prev = h_ref_smooth.detach()  # 保存给下一步
```

**参数**：`ema_decay=0.7`（70% 来自上一步平滑值，30% 来自当前步原始值）

**改动位置**：`pipeline.py` 中 `_fi_hook_fn` 的 `h_ref = ref_features[step_idx]` 处

**预期收益**：XCLIP 微提升（时序更平滑），CLIP 几乎不变

**风险**：极低（EMA 是平滑操作，不会引入新问题）

---

#### D2：统一门控框架（第二做）

**问题**：当前 quality_scale（样本级）和 adaptive_gate（step×layer级）是两个独立门控，乘法关系缺乏统一语义。

**方案**：将 quality_scale 融入 adaptive_gate 的温度参数，统一为"置信度加权注入"

```python
# 当前：两个独立门控相乘
quality_scale = sigmoid_gate(mean_cos, threshold=0.05, k=20)   # 样本级 0~1
adaptive_gate = 1 - sigmoid(temp * (cos - 0.5))                # 特征级 0~1
lambda_eff = lambda_t * quality_scale * adaptive_gate

# 新方案：quality_scale 调节 adaptive_gate 的温度
temp_adaptive = base_temp * (1 + quality_scale)  # quality_scale 高→温度高→gate 更敏感
confidence = 1 - sigmoid(temp_adaptive * (cos(h_gen, h_ref) - 0.5))
lambda_eff = lambda_t * confidence
```

**直觉解释**：
- 低质量样本（quality_scale 低）→ 温度低 → gate 更平滑 → 注入更保守
- 高质量样本（quality_scale 高）→ 温度高 → gate 更敏感 → 区分度更好

**改动位置**：`pipeline.py` 中 `_fi_hook_fn` 内的 adaptive gate 计算逻辑

**预期收益**：低质量样本更稳定（如 S31），数学框架更干净

**风险**：低（温度参数微调，不改变注入方式）

---

#### D3：正交残差注入（第三做，风险中等）

**问题**：当前线性插值 `h = (1-λ)·h_gen + λ·h_ref` 在 h_gen 和 h_ref 方向差大时（cos < 0.3）可能产生"中间态"特征，不在 DiT 正常流形上。

**方案**：只注入 h_ref 相对于 h_gen 的正交分量，不改变 h_gen 的方向

```python
# 当前：线性插值
h_injected = (1 - lambda_eff) * h_gen + lambda_eff * h_ref

# 新方案：正交残差注入
delta = h_ref - h_gen
proj_coeff = (delta * h_gen).sum(dim=-1, keepdim=True) / (h_gen * h_gen).sum(dim=-1, keepdim=True).clamp(min=1e-8)
delta_perp = delta - proj_coeff * h_gen
h_injected = h_gen + lambda_eff * delta_perp
```

**直觉解释**：
- 线性插值：把 h_gen 往 h_ref 的方向"拉"
- 正交注入：只在 h_gen "缺少"的方向上补，不改变 h_gen 原有的方向
- cos 高时两者几乎等价；cos 低时正交注入更安全（不产生中间态）

**改动位置**：`pipeline.py` 中 `_fi_hook_fn` 的注入公式

**预期收益**：高风险样本（如 S31，当前唯一退步的）更安全

**风险**：中（`delta_perp` 的范数可能很大，需检查数值稳定性；且与 adaptive_gate 的交互需要实验验证）

---

## 实验优先级排序

| 优先级 | 实验 | 预计耗时 | 前置条件 |
|--------|------|---------|---------|
| 1 | 方向C：L1+L2+L3 三层全叠加 | ~1h（10样本） | 需先有 v9 caption |
| 2 | 方向D1：EMA 特征平滑 | ~1.5h（10样本） | 修改代码 ~5行 |
| 3 | 方向A：200 样本基准评测 | ~14h | 可与上面并行挂机 |
| 4 | 方向D2：统一门控 | ~1.5h（10样本） | 修改代码 ~15行 |
| 5 | 方向D3：正交残差注入 | ~1.5h（10样本） | 修改代码 ~20行，需数值验证 |

---

## 论文故事方向

**已确定**：方案A — 三层正交互补架构

- L1 在语义空间（"画什么样子"）
- L2 在噪声空间（"运动的节奏"）
- L3 在特征空间（"怎么动"）
- 三者空间正交、时序互补、信息互补
- SVD 不冗余（SVD+FI 比纯 FI 高 2.5%/5.8%），三层各有独立贡献

---

## A会发表现状评估

### 当前技术贡献盘点

| 贡献点 | 量级 | A会够不够 |
|--------|------|-----------|
| L1: v9 删减+VLM 补充 | Prompt 工程级别 | ❌ 不够，这是工程 trick |
| L2: SVD Noise Prior α=0.004 | 单一超参，2% 方向偏移 | ⚠️ 有了但太薄，8 个增强方向全部失败 |
| L3: Feature Injection + 自适应门控 | **核心贡献，XCLIP +13.6%** | ✅ 有新意，但需要更深的分析 |
| 三层正交互补性 | 实验上验证了 | ⚠️ 需要更强的理论支撑 |

### 🔴 A会的主要 Gap

#### Gap 1：缺少理论深度

当前论文叙事是"三层正交互补"——操作空间不同→正交→互补。但 A 会审稿人会问：

> "正交"只是你人为定义的（空间不同就叫正交？），有没有信息论/互信息/Fisher 信息量的形式化证明？

L2 的 α=0.004 天花板目前是"8 个方向消融实验证明"——这是经验结论，不是理论结论。审稿人会问：**为什么 0.004 是天花板？有没有理论解释？**

#### Gap 2：L2 贡献太薄

SVD+FI (0.9042/0.8138) vs 纯 FI (0.8819/0.7669)，L2 的独立贡献是 CLIP +2.5%, XCLIP +5.8%。但这个贡献的本质是什么？

- 如果只是"给了 FI 一个 quality_scale 门控信号" → 审稿人说"那直接从 η_inv 算门控就行了，SVD 滤波完全多余"
- 如果是"SVD 提供的微弱运动偏置被 FI 放大了" → 需要实验证据（比如对比 SVD blend z_T vs 纯随机 z_T 但保留 quality_scale）

#### Gap 3：评测规模和基线

- 所有核心实验只跑了 10 样本 → A 会审稿人会直接质疑统计显著性
- Baseline 只有一个模型（Wan2.1-1.3B）→ 缺少跨模型泛化性
- 缺少与 SOTA 方法的定量对比（如 Ctrl-Adapter、Video-ControlNet、AnyV2V 等）

#### Gap 4：缺少人评和用户研究

A 会普遍要求人工评估（MOS 分数），纯自动化指标（CLIP/XCLIP）说服力不足。

### 🟢 有 A 会潜力的点

1. **Feature Injection 本身有新意**：零训练、自适应门控、不修改 ODE 路径——在视频扩散模型中没有被充分探索
2. **三层架构的消融实验做得很扎实**：18+ 组消融，每组的失败原因都有分析
3. **Cross-attention 位置权重实验**：T5 encoder 的 U 型分布发现，对 prompt 设计有指导意义

---

## A会补充计划

### 必须补的（硬伤）

| 补充项 | 工作量 | 作用 |
|--------|--------|------|
| **200 样本评测** | ~14h GPU | 统计显著性 |
| **跨模型验证**（至少加一个，如 CogVideoX 或 Open-Sora） | ~1周 | 泛化性 |
| **与 SOTA 定量对比**（Ctrl-Adapter / AnyV2V 等） | ~1周 | 说服力 |
| **MOS 人工评估**（20人×50样本） | ~1-2周 | 审稿人必问 |
| **理论分析**：FI 为什么 work？为什么 α=0.004 是天花板？ | ~1-2周 | 论文深度 |

### 建议补的（加分项）

| 补充项 | 工作量 | 作用 |
|--------|--------|------|
| L2 贡献的消解实验：SVD blend z_T vs 纯随机 z_T 但保留 quality_scale | ~3h GPU | 隔离 L2 的真实贡献 |
| FI 注入层的分析：为什么 mid(10~19) 最优？不同层注入了什么语义？ | ~1周 | 论文 story 深度 |
| 失败案例深入分析（S31 为什么 XCLIP 崩塌） | ~1天 | 审稿人喜欢看 failure analysis |
| 可视化：FI 注入前后 DiT 注意力图的变化 | ~1周 | 直观展示 FI 的作用机制 |

### A会路线图

```
当前状态：有核心技术（FI +13.6%），有扎实消融，但缺深度和广度

Phase 1（1-2周）：补硬伤
  ├── 200 样本评测
  ├── SVD 贡献消解实验（隔离 quality_scale 的作用）
  └── 理论分析：FI 为什么 work + α 天花板的信息论解释

Phase 2（2-3周）：补广度
  ├── 跨模型验证（CogVideoX）
  ├── 与 SOTA 定量对比（2-3个 baseline 方法）
  └── MOS 人工评估

Phase 3（1-2周）：补深度
  ├── FI 层分析 + 可视化
  ├── 失败案例分析
  └── 论文写作

总预计：5-7周
```

### 投稿目标

| 目标会议 | 截稿时间 | 难度 | 现实性 |
|---------|---------|------|--------|
| CVPR 2027 | 2026-11 | A 会，最高 | 需要完成全部 3 个 Phase |
| ECCV 2026 | 2026-03 (已过) | A 会 | — |
| ACM MM 2026 | 2026-04 | B+ 会 | 需完成 Phase 1+2 |
| NeurIPS 2026 | 2026-05 | A 会 | 需完成全部 3 个 Phase |
| AAAI 2027 | 2026-08 | A 会 | 需完成全部 3 个 Phase |

**建议**：优先冲 NeurIPS 2026 或 CVPR 2027，同时以 ACM MM 2026 作为保底。
