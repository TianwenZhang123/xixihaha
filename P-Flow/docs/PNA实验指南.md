# PNA (Prompt-Noise Alignment) 实验指南

> **版本**: v1.0 (2026-06-19)
>
> **状态**: 实验1 运行中

---

## 一、背景与动机

### 1.1 问题：当前门控方案不如基线

| 版本 | CLIP | X-CLIP | vs 基线 |
|------|------|--------|---------|
| Baseline（不加SVD+FI） | 0.883 | 0.730 | — |
| SVD+FI基线（α=0.004固定） | 0.899 | **0.771** | +0.041 |
| v3.5+（M_d×TSR, α_floor=0.002） | 0.896 | 0.754 | -0.017 ❌ |

**退化 case**：
- Case 21: X-CLIP 0.8064 → 0.6801 (-0.127)
- Case 43: X-CLIP 0.9156 → 0.8554 (-0.060)
- Case 72: X-CLIP 0.8869 → 0.7537 (-0.133)

### 1.2 根因分析

当前 L2 门控依赖 **LLM 离线打分的 M_d（运动明确度）× TSR（时序信号可靠性）**：

```
问题: M_d 是 LLM 根据 caption 判断的"运动强弱"，但:
1. M_d=1.0 不代表 SVD 提取的 η_temporal 方向与 prompt 一致
2. M_d=0.0 不代表 η_temporal 完全无用
3. LLM 无法感知具体样本的噪声特性
```

而 L3 FI 有 **Adaptive Gate**（cos(h_gen, h_ref)），在每步每层测量特征对齐度，方向对齐时少注入。L2 缺少这种在线对齐度判断。

### 1.3 核心思路

> "计算初始的 prompt 在 t2v 第一层编码后的方向和 SVD 初始噪声的方向" —— 用户提出

**PNA (Prompt-Noise Alignment)**: 用模型自身的一步前向探测 η_temporal 的方向是否有利。

---

## 二、门控体系全景

```
输入: η_temporal, caption, prompt_embeds
│
├── L2: 噪声空间门控 → 决定 α_eff
│   │
│   ├── [路径A] PNA 在线门控 (--pna_probe) ← 新!
│   │   一步模型前向 → PNA_score → α = α_min + PNA × (α_max - α_min)
│   │   不需要 LLM, 模型自己判断 η_temporal 方向是否有利
│   │
│   └── [路径B] M_d × TSR 融合门控 (--adaptive_alpha, 默认路径)
│       f(M_d, TSR) = M_d×TSR + (1-M_d)×0.1×TSR
│       α_eff = max(α_floor × max(M_d, 0.3), α_fusion)
│
├── L3: 特征空间门控 → 决定 λ_eff
│   │
│   ├── Quality Scale (样本级, η_temporal 帧间 cos)
│   ├── Adaptive Gate (步×层, cos(h_gen, h_ref))
│   ├── M_d × QS (默认关闭)
│   └── FI-α Coupling (--fi_alpha_coupling) ← 新!
│       λ_eff = λ × (α_eff / α_ref), α 低时 FI 注入同步降低
│
└── 最终: α_eff, λ_eff
```

---

## 三、PNA 探测原理

### 3.1 算法流程

```
Step 1: 构造两种噪声
  η_mixed = √α_test · η_temporal + √(1-α_test) · η_random    (α_test=0.004)
  η_random = 纯随机噪声

Step 2: 分别用模型做一步前向 (t=0.95, 接近纯噪声)
  feat_mixed = transformer(η_mixed, t=0.95, prompt_embeds)     → mid 层输出
  feat_random = transformer(η_random, t=0.95, prompt_embeds)   → mid 层输出

Step 3: 计算特征差异
  delta = feat_mixed - feat_random                          # η_temporal 带来的偏移

Step 4: 提取 PNA 指标
  relative_impact = ‖delta‖ / ‖feat_random‖              # 影响强度
  cos_consistency = cos(delta[:half], delta[half:])           # 方向一致性

Step 5: 映射到 [0,1]
  pna_raw = impact × (0.5 + 0.5 × consistency)
  PNA = sigmoid(-500 × (pna_raw - 0.003))                   # 归一化
```

### 3.2 物理意义

| 指标 | 含义 | 高值含义 | 低值含义 |
|------|------|----------|----------|
| `relative_impact` | η_temporal 对模型预测的影响强度 | 模型被显著影响 | 几乎无影响 |
| `cos_consistency` | 特征偏移的方向一致性 | 各处偏移一致（η_temporal 有序信号） | 偏移混乱（η_temporal 是噪声） |
| `PNA_score` | 综合评分 | η_temporal 有利 → α 可以大 | η_temporal 无利/有害 → α 应小 |

### 3.3 与 FI Adaptive Gate 的类比

| | L3 Adaptive Gate | L2 PNA |
|---|---|---|
| 测量什么 | cos(h_gen, h_ref) 特征对齐 | δ(feat_mixed, feat_random) 方向一致性 |
| 怎么测 | 每步每层 hook | 生成前一步探测 |
| 结果 | gate ∈ [0,1] → 调节 λ | PNA ∈ [0,1] → 调节 α |
| 核心思想 | 方向对齐时少注入 | 方向有利时多注入 |

---

## 四、实验设计

### 4.1 已有实验数据（10样本）

| Case | M_d | 类型 | SVD+FI基线 XCLIP | v3.5+ XCLIP | 期望 PNA 行为 |
|------|-----|------|------------------|-------------|-------------|
| 7 | 1.0 | physics-fluid | 0.768 | 0.809 ✅ | PNA高→α大→保持改善 |
| 21 | 1.0 | unusual-activity | 0.806 | 0.680 ⚠️ | PNA低→α小→修复退化 |
| 31 | 1.0 | unusual-activity | 0.556 | 0.665 ✅ | PNA中高→保持改善 |
| 32 | 1.0 | animal | 0.840 | 0.815 | PNA高→α大→恢复 |
| 33 | 1.0 | human-activity | 0.864 | 0.886 ✅ | PNA高→α大→保持 |
| 43 | 1.0 | animal | 0.916 | 0.855 ⚠️ | PNA极低→α极小 |
| 46 | 0.3 | unusual-subject | 0.765 | 0.731 | PNA看情况 |
| 72 | 0.0 | scene/camera | 0.887 | 0.754 ⚠️ | PNA低→α小→修复退化 |
| 73 | 0.3 | scene | 0.651 | 0.625 | PNA低→α小 |
| 80 | 0.0 | scene/camera | 0.662 | 0.719 ✅ | PNA看情况 |

### 4.2 实验矩阵

| 实验 | L2路径 | L3-FI耦合 | 目的 | 预计时间 |
|------|--------|-----------|------|----------|
| **实验1** | PNA (`--pna_probe`) | FI-α协同 (`--fi_alpha_coupling`) | **核心验证** | ~50min |
| **实验2** | PNA alone | 无协同 | 消融：FI-α贡献 | ~50min |
| **实验3** | M_d×TSR (`--adaptive_alpha`) | FI-α协同 | 对比：旧路径+新协同 | ~45min |

### 4.3 成功标准

- **整体**: mean X-CLIP ≥ 0.771（SVD+FI基线）
- **退化修复**: Case 21/43/72 不再显著退化
- **改善保持**: Case 7/31/33/80 保持或更好

---

## 五、运行命令

### 5.1 实验1: PNA + FI-α 协同（核心）

```bash
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/PNA_FIcoupling_10samples \
    --sample_ids 7 21 31 32 33 43 46 72 73 80 \
    --inversion --svd --blend \
    --adaptive_alpha --alpha_max 0.006 --alpha_min 0.0 \
    --pna_probe --pna_probe_step 0.95 \
    --pna_alpha_max 0.006 --pna_alpha_min 0.0005 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --fi_alpha_coupling --fi_alpha_ref 0.004 \
    --seed 42 --verbose
```

### 5.2 评测命令

```bash
cd /root/xixihaha/P-Flow && python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/PNA_FIcoupling_10samples \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir outputs/PNA_FIcoupling_10samples/eval_clip
```

### 5.3 实验2: PNA alone（消融）

```bash
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/PNA_only_10samples \
    --sample_ids 7 21 31 32 33 43 46 72 73 80 \
    --inversion --svd --blend \
    --adaptive_alpha --alpha_max 0.006 --alpha_min 0.0 \
    --pna_probe --pna_probe_step 0.95 \
    --pna_alpha_max 0.006 --pna_alpha_min 0.0005 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose
```

### 5.4 实验3: M_d×TSR + FI-α协同（对比）

```bash
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/MdTSR_FIcoupling_10samples \
    --sample_ids 7 21 31 32 33 43 46 72 73 80 \
    --inversion --svd --blend \
    --adaptive_alpha --alpha_max 0.006 --alpha_min 0.0 \
    --alpha_floor 0.002 --alpha_md_floor 0.3 \
    --md_file data/md_scores.csv \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --fi_alpha_coupling --fi_alpha_ref 0.004 \
    --seed 42 --verbose
```

---

## 六、日志解读指南

跑完实验后，重点查看以下日志行：

### 6.1 PNA 探测结果（每个样本一行）

```
[PNA] probe_t=0.95, α_test=0.004, layer=15,
      relative_impact=XXXXXX, cos_consistency=XXXX,
      pna_raw=XXXXXX, PNA_score=XXXX
```

**关注点**：
- `relative_impact`: 典型范围 0.001~0.01（很小是正常的，因为 α_test=0.004 本身就小）
- `cos_consistency`: 正值=方向一致，负值=方向混乱
- `PNA_score`: 最终 α 的决定因素

### 6.2 α 分配结果

```
[PNA-α] PNA_score=XXXX, α_min=0.0005, α_max=0.006 → α_eff=XXXXXX
```

### 6.3 FI-α 协同（如果启用）

```
[FI-α Coupling] α_eff=XXXXXX, α_ref=0.004, scale=XXXXX
      (λ_max: 0.05→XXXXX)
```

### 6.4 评测结果

```
[1/10] 7 CLIP(o-t=X, g-t=X, o-g=X) XCLIP(o-t=X, g-t=X, o-g=X)
[2/10] 21 CLIP(...) XCLIP(...)
...
Evaluation complete!
  orig_gen_clip mean:  X.XXXXXX
  orig_gen_xclip mean: X.XXXXXX
```

---

## 七、后续调整策略

### 7.1 根据 PNA 数据分布调参

| PNA 数据情况 | 需要做的 |
|---|---|
| PNA 区分度好（退化case低，改善case高） | ✅ 直接对比评测，跑 30 样本全量验证 |
| PNA 全都≈0 或 ≈1（无区分度） | 调整 sigmoid 参数：`(k, center)` 当前为 `(500, 0.003)` |
| PNA 和预期相反（退化的 case PNA 高） | 可能需要换指标：用帧间 PNA score 差异代替全局 cos_consistency |
| relative_impact 全 < 0.001 | 提高 α_test 到 0.01 或 0.02，增强探测信号 |

### 7.2 如果 PNA 无效的备选方案

**备选 A**: 用帧级 PNA — 对每帧分别计算 cos_consistency，再平均
**备选 B**: 用模型输出的 z0 预测差异（而非 mid 层特征）作为 PNA 信号
**备选 C**: 回到 M_d×TSR 但用更精细的 α 分段（如按场景类型查表）

---

## 八、代码修改清单

### pipeline.py 新增/修改

| 位置 | 改动 | 说明 |
|------|------|------|
| `PFlowConfig` | 新增 `pna_probe`, `pna_probe_step`, `pna_alpha_max`, `pna_alpha_min` | PNA 配置参数 |
| `_compute_pna_score()` | 新方法 (~90行) | PNA 探测核心逻辑 |
| `_get_latents()` | 新增 PNA 路径分支 | 当 `pna_probe=True` 时走 PNA 门控 |
| `run()` | 新增 `_current_prompt_embeds` 缓存 | 为 PNA 提供 prompt_embeds |
| `_generate_with_fi()` | 新增 FI-α Coupling | λ 随 α 同步缩放 |

### run.py 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pna_probe` | False | 启用 PNA 在线门控 |
| `--pna_probe_step` | 0.95 | PNA 探测的 t 值 |
| `--pna_alpha_max` | 0.006 | PNA 门控 α 上限 |
| `--pna_alpha_min` | 0.0005 | PNA 门控 α 下限 |
| `--fi_alpha_coupling` | False | FI-α 协同开关 |
| `--no_fi_alpha_coupling` | False | 显式关闭 FI-α 协同 |
| `--fi_alpha_ref` | 0.004 | FI-α 参考值 |

---

## 九、关键文件路径

```
P-Flow/
├── src/pipeline.py          # 主逻辑（PNA + 门控）
├── run.py                  # CLI 入口
├── docs/
│   ├── PNA实验指南.md        # 本文档
│   ├── 场景感知门控策略.md   # 旧门控体系文档
│   └── 方法总结_P-Flow三层架构.md
└── outputs/
    └── PNA_FIcoupling_10samples/  # 实验1 输出
        ├── run_log.txt          # 生成日志（含 PNA 数据）
        ├── eval_clip/            # 评测结果
        └── sample_*/            # 每个样本的视频
```
