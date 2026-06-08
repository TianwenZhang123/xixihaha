# L1 Prompt Rewrite 策略对比实验

> **实验目的**：验证不同 LLM 改写策略对 L1（Prompt Layer）生成质量的影响，找到超越 baseline 的最优改写方案。
>
> **实验时间**：2025-05-29
>
> **评测指标**：`orig_gen_clip` = 原始视频与生成视频的 CLIP 帧级相似度；`orig_gen_xclip` = 原始视频与生成视频的 X-CLIP 时序语义相似度。
>
> **样本规模**：20 samples（captions_qwen 前 20 个）
>
> **生成模型**：Wan2.1-T2V-1.3B（UMT5 text encoder）
>
> **生成参数**：steps=30, guidance=5.0, seed=42

---

## 实验总览

| # | 方案 | CLIP-Score (avg) | XCLIP-Score (avg) | vs Baseline |
|---|------|:---:|:---:|:---:|
| A | Baseline（原始 VLM caption） | 0.8864 | 0.7309 | — |
| B | 旧 LLM 改写 + Negative Prompt | 0.8580 | 0.6755 | ❌ CLIP -3.2%, XCLIP -7.6% |
| C | 旧 LLM 改写（去 Negative） | 0.8739 | 0.7185 | ❌ CLIP -1.4%, XCLIP -1.7% |
| D | **V4 LLM 改写** | **0.8873** | **0.7435** | ✅ CLIP +0.1%, XCLIP +1.7% |

---

## 实验 A：Baseline（原始 VLM Caption 直出）

**描述**：Qwen2.5-VL-7B 对原始视频生成 caption，不做任何后处理，直接送入 Wan2.1 生成视频。

**流程**：原始视频 → VLM caption → Wan2.1-T2V → 生成视频

**结果**：

- **CLIP-Score (avg)**：0.8864
- **XCLIP-Score (avg)**：0.7309

**定位**：作为所有改写策略的参照基准。VLM 原生输出虽然语义完整，但缺少针对 T2V 模型的优化（如 subject-first 结构、temporal chain 等）。

---

## 实验 B：旧 LLM 改写 + Negative Prompt

**描述**：使用旧版 `rewrite_hybrid.py` 的 System Prompt 对 VLM caption 进行 LLM 改写，同时启用 Negative Prompt 拼接策略。

**旧版 Prompt 问题**：
1. System Prompt 中包含 "If the original says 'dark brown hulls' or 'dappled sunlight', keep those exact phrases" — 被 LLM 误解为推荐词汇，导致无关场景（地铁、灯笼、水下）中出现 "dark brown hulls"
2. 输出模式高度模板化："initially...then...as the scene progresses...gradually..." 在全部 20 个样本中反复出现
3. 幻觉注入：改写后出现原文未提及的内容（sunken boats, kelp, barnacles 等）

**结果**：

- **CLIP-Score (avg)**：0.8580
- **XCLIP-Score (avg)**：0.6755
- **vs Baseline**：CLIP -0.0284（-3.2%），XCLIP -0.0554（-7.6%）

**结论**：旧改写策略 + Negative Prompt 双重劣化，是四组中表现最差的方案。Negative Prompt 可能干扰了 UMT5 对正向 prompt 的编码。

---

## 实验 C：旧 LLM 改写（去 Negative Prompt）

**描述**：使用同一旧版 System Prompt 进行 LLM 改写，但去掉 Negative Prompt，仅保留正向 caption。目的是隔离 Negative Prompt 的影响。

**结果**：

- **CLIP-Score (avg)**：0.8739
- **XCLIP-Score (avg)**：0.7185
- **vs Baseline**：CLIP -0.0125（-1.4%），XCLIP -0.0124（-1.7%）
- **vs 实验 B**：CLIP +0.0159，XCLIP +0.0430（去 Negative 后显著恢复）

**结论**：去掉 Negative Prompt 后指标大幅回升，但仍低于 baseline。说明旧版改写策略本身也存在问题——模板化输出和幻觉注入导致语义偏移。

---

## 实验 D：V4 LLM 改写（修复版）

**描述**：将 `rewrite_hybrid.py` 的 System Prompt 同步为 `run_hybrid_iter.py` 中经过验证的 V4 版本，核心原则为"复制原文 + 仅做 3 处手术式修改"。

**V4 策略要点**：
1. **OPENING**：Subject-first — 将主语提到句首，利用 UMT5 position-0 注意力权重优势
2. **ACTION**：Temporal chain — 仅对主体运动描述的 1-2 句添加时序连接词（initially → then → finally）
3. **ENDING**：Vivid keyword — 在末尾补充 1-2 个生动的视觉细节关键词

**约束条件**：
- 复制整个输入文本，仅做上述 3 处修改
- 不添加原文未提及的信息
- edit_ratio ≤ 50%（diff check），length ≥ 70%
- 默认 temperature=0.5（降低随机性）
- 4 个 few-shot 示例引导输出格式

**结果**：

- **CLIP-Score (avg)**：0.8873
- **XCLIP-Score (avg)**：0.7435
- **vs Baseline**：CLIP +0.0009（+0.1%），XCLIP +0.0126（+1.7%）

**观察**：
- 改写后文本平均长度比 ≈ 1.18（字数膨胀约 18%），在可接受范围内
- XCLIP 提升显著（+1.7%），说明 temporal chain 策略有效增强了生成视频的时序连贯性
- CLIP 小幅提升（+0.1%），说明 subject-first 策略帮助模型更准确地生成主体

---

## 关键发现与分析

### 1. Negative Prompt 对 Wan2.1-1.3B 有害

实验 B vs C 的对比表明，Negative Prompt 在当前模型上造成了额外 ~5.9% 的 XCLIP 损失。可能原因：Wan2.1 的 UMT5 encoder 并非为 negative prompt 设计，负向文本占用了 context 空间，干扰了正向语义编码。

### 2. 改写策略的关键在于"约束"而非"丰富"

旧版策略试图让 LLM "丰富" caption（加细节、加时序过渡），结果导致语义漂移。V4 策略反其道而行——要求 LLM "几乎不改"，仅做结构性微调，反而获得了正向增益。

### 3. Subject-first + Temporal chain 双策略有效

- Subject-first 利用了 UMT5 position-0 的注意力权重优势，确保生成视频的主体正确
- Temporal chain 仅对运动句添加时序词，增强了视频的动态连贯性，体现在 XCLIP 的 +1.7% 提升上

### 4. 版本不同步是 bug 根因

`rewrite_hybrid.py`（创建于 5/27 19:06）早于 V4 prompt 定稿（5/28 10:11 in `run_hybrid_iter.py`），两者从未同步。这是一个工程管理问题，已通过统一 prompt 源码修复。

---

## 后续方向

1. **叠加 L2/L3 层**：在 V4 caption 基础上启用 SVD Noise Prior（L2）和 Velocity Matching（L3），观察多层叠加的增益
2. **扩大验证规模**：从 20 samples 扩展到全量 200 samples，确认 V4 策略的稳健性
3. **收紧长度约束（可选）**：如需将 ratio 从 1.18 压到 ≤ 1.10，可在 prompt 中增加 "DO NOT add new sentences" 约束
4. **Negative Prompt 替代方案**：研究 CFG-scale 调节或 attention masking 等替代负向引导的方法

---

## 前 10 样本子集（用于 L2 叠加实验对照）

> 后续 L2（SVD Noise Prior）实验使用前 10 个样本，此处单独记录该子集的指标作为对照基线。

**样本集**：7, 17, 21, 31, 32, 33, 34, 43, 46, 47

### Baseline（原始 VLM caption 直出）

| 指标 | 均值 |
|------|------|
| orig_gen_clip | 0.8753 |
| orig_gen_xclip | 0.7491 |

### L1 V4 改写

| 指标 | 均值 | vs Baseline |
|------|------|------|
| orig_gen_clip | 0.8794 | +0.0041（+0.5%） |
| orig_gen_xclip | 0.7594 | +0.0103（+1.4%） |

### L1+L2（SVD Noise Prior, α=0.004, ρ_s=0.1, ρ_m=0.9）

| 指标 | 均值 | vs Baseline | vs L1 |
|------|------|------|------|
| orig_gen_clip | 0.8818 | +0.0065（+0.7%） | +0.0024（+0.3%） |
| orig_gen_xclip | 0.7863 | +0.0372（+5.0%） | +0.0269（+3.5%） |

> 实验日期：2025-06-08。L2 在 XCLIP 维度带来了 3.5% 的增量提升（运动一致性显著改善），但 CLIP 增量只有 0.3%（外观相似度提升饱和）。

---

## 文件引用

| 文件 | 说明 |
|------|------|
| `scripts/rewrite_hybrid.py` | L1 改写脚本（已修复为 V4 prompt） |
| `scripts/run_hybrid_iter.py` | V4 prompt 原始来源（lines 51-109） |
| `src/pipeline.py` | P-Flow 主管线（L1-L4 层控制） |
| `docs/复现指南_逐层验证.md` | 各层期望指标参考 |
