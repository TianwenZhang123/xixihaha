# v5 smoothG 实验分析与 SVD-only 注入量扫描实验设计

> **日期**：2026-06-21
>
> **核心发现**：v5 smoothG (cap=0.6, pw=1.2) 的 X-CLIP=0.7716 超过 Baseline (SVD+FI α=0.004)，但方案 G 给出的 α 与之前消融实验的"经验值"偏差很大。这说明经验值可能不完全正确，需要重新验证。

---

## 一、实验结果对比

### 1.1 各方案评测结果

| 方案 | X-CLIP mean | CLIP mean | PNA α 偏差 | 说明 |
|------|------------|-----------|-----------|------|
| **Baseline (SVD+FI, α=0.004 固定)** | ~0.771 | 0.909 | N/A (固定α) | 论文默认配置 |
| **v5 smoothG (cap=0.6, pw=1.2)** | **0.7716** | **0.9155** | 大 (+0.0034~+0.0043) | PNA自适应G, 无Coupling |
| v6 schemeE (分段修正) | 0.7468 | 0.9142 | 最小 (≤0.0001) | PNA自适应E |
| v6 schemeG1 (cap=0.20, pw=1.0) | 0.7436 | 0.9139 | 小 (~0.0003) | PNA自适应G1 |

### 1.2 逐样本 v5 smoothG X-CLIP

| Sample | 类型 | temporal_frame_cos | 方案G α | 经验α | G偏差 | X-CLIP | Quality Scale |
|--------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 🐾 animal (金毛雪地) | 0.1643 | 0.005222 | 0.0015 | +0.0037 | 0.8271 | 0.1957 |
| 50 | 🌀 unusual (猫坐车) | 0.1161 | 0.005808 | 0.0015 | +0.0043 | 0.8100 | 0.5348 |
| 73 | 🎬 scene (仓库) | 0.4214 | 0.004195 | 0.0008 | +0.0034 | 0.6289 | 0.6748 |
| 80 | 🎬 scene+cam (蒲公英) | 0.7951 | 0.000500 | 0.0008 | -0.0003 | 0.7467 | 0.7407 |
| 111 | 🎬 scene+cam (航拍森林) | 0.3041 | 0.005142 | 0.0008 | +0.0043 | 0.8452 | 0.6260 |

---

## 二、关键矛盾：经验值 vs 实际效果

### 2.1 矛盾核心

之前 6-18 消融实验得出的"经验值"（不同场景的最佳α）是基于 **SVD+FI 联合实验** 的结果。但 v5 smoothG 的 α 偏离经验值很多，X-CLIP 反而更好。有两种可能：

**假设 A：经验值是错的**
- 6-18 消融实验的样本数少（每组5个case），经验值不具代表性
- FI 的 quality_scale 和 adaptive_gate 在联合实验中弥补了 SVD 过度注入的负面影响
- 实际上 α 可以比经验值更大

**假设 B：经验值对的，但 FI 弥补了 SVD 的问题**
- SVD 给了过高的 α（如 case 73 的 α=0.0042 vs 经验 0.0008），注入了错误的运动方向
- 但 FI 的 quality_scale（0.6748）已经自动压低了 λ，且 adaptive_gate 进一步降低了实际注入量
- 所以 SVD+FI 联合时 FI 补救了 SVD 的错误

### 2.2 判断方法

**必须隔离 FI 的影响，单独测试 SVD 的效果**。如果：
- SVD-only 时 α=0.0042 确实让 case 73 崩溃 → 假设 B 正确，经验值对的
- SVD-only 时 α=0.0042 效果还行 → 假设 A 正确，经验值需要修正

---

## 三、SVD 已有的内容分离能力

### 3.1 当前 SVD 两阶段滤波

代码位于 `src/svd_filter.py`：

- **Stage 1: Spatial Decontenting** (去外观/内容)
  - reshape 为 (C×F, H×W)，做 SVD，去除 top-k_s 空间主成分
  - 输出：`noise_after_spatial`（去外观后的噪声）+ `eta_spatial`（被移除的外观分量）
  - `eta_spatial` 已经在 `svd_stats` 中返回

- **Stage 2: Temporal Retention** (保运动)
  - reshape 为 (C×H×W, F)，做 SVD，保留 top-k_m 时间主成分
  - 输出：`eta_temporal`（运动先验噪声）

### 3.2 当前 blend 公式

```python
# 基础混合 (β=0):
η = √α · η_temporal + √(1-α) · η_random

# 三路混合 (β>0, 启用外观分量):
η = √α · η_temporal + √β · η_spatial + √(1-α-β) · η_random
```

- `η_temporal`: SVD Stage 2 提取的运动先验（去外观保运动）
- `η_spatial`: SVD Stage 1 分离的外观/内容分量（已实现 renorm 匹配量级）
- 当前默认 β=0.0，不使用外观分量

### 3.3 η_spatial 的潜在用途

`eta_spatial` 已经在 `svd_stats["eta_spatial"]` 中可用，且 `_get_latents` 已实现了 β>0 时的三路混合（含 renorm）。这意味着：

- **对场景型样本**：η_spatial 包含外观/内容信息，可能比 η_temporal（只有运动方向）更适合注入
- **时序一致性**：η_spatial 包含"场景是什么"的信息，而 η_temporal 包含"场景怎么动"的信息。场景型视频更需要"是什么"而非"怎么动"

---

## 四、实验设计

### 4.1 实验目标

1. **隔离 FI 影响**：纯 SVD 实验，确定 α 的真实最佳值
2. **重新验证经验值**：在不同 cos 的 case 上做 α 扫描
3. **探索 η_spatial 的作用**：对场景型样本，β>0 是否能改善效果

### 4.2 实验组设计

#### 实验 1：SVD-only α 扫描（5 case × 5 α 值）

**目标**：隔离 FI，确定不同场景下 SVD 单独的最佳 α

| 实验组 | Case | cos | 类型 | α 值 |
|--------|------|-----|------|------|
| 1a | 32 | 0.16 | 🐾 animal | 0.001, 0.002, 0.004, 0.006, 0.008 |
| 1b | 50 | 0.12 | 🌀 unusual | 0.001, 0.002, 0.004, 0.006, 0.008 |
| 1c | 73 | 0.42 | 🎬 scene | 0.0005, 0.001, 0.002, 0.004, 0.006 |
| 1d | 80 | 0.80 | 🎬 scene+cam | 0.0005, 0.001, 0.002, 0.004, 0.006 |
| 1e | 111 | 0.30 | 🎬 scene+cam | 0.0005, 0.001, 0.002, 0.004, 0.006 |

> 注意：场景型 (73, 80, 111) 的 α 扫描范围偏向更小值

#### 实验 2：SVD+默认配置对照（无 FI）

**目标**：验证固定 α=0.004 的 SVD-only 基线效果

5 个 case 用固定 α=0.004、不加 FI、不加 PNA。

#### 实验 3：SVD η_spatial 探索（β>0）

**目标**：对场景型 case，探索外观分量的注入效果

仅对 case 73 和 111，在最优 α 基础上增加 β 扫描（0.001, 0.002, 0.004）。

### 4.3 具体命令

#### 实验 1a：Case 32 α 扫描 (SVD-only, 无FI, 无PNA)

```bash
# α=0.001
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_32/alpha_001 \
    --sample_ids 32 \
    --inversion --svd --blend \
    --alpha 0.001 \
    --seed 42 --verbose

# α=0.002
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_32/alpha_002 \
    --sample_ids 32 \
    --inversion --svd --blend \
    --alpha 0.002 \
    --seed 42 --verbose

# α=0.004 (默认)
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_32/alpha_004 \
    --sample_ids 32 \
    --inversion --svd --blend \
    --alpha 0.004 \
    --seed 42 --verbose

# α=0.006
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_32/alpha_006 \
    --sample_ids 32 \
    --inversion --svd --blend \
    --alpha 0.006 \
    --seed 42 --verbose

# α=0.008
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_32/alpha_008 \
    --sample_ids 32 \
    --inversion --svd --blend \
    --alpha 0.008 \
    --seed 42 --verbose
```

> 注意：不用 `--feature_inject`，不用 `--adaptive_alpha`，不用 `--pna_probe`。
> 这样是纯 SVD 固定 α 注入，没有任何自适应门控和 FI 的干扰。

#### 实验 1c：Case 73 α 扫描 (SVD-only)

```bash
# α=0.0005
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_73/alpha_0005 \
    --sample_ids 73 \
    --inversion --svd --blend \
    --alpha 0.0005 \
    --seed 42 --verbose

# α=0.001
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_73/alpha_001 \
    --sample_ids 73 \
    --inversion --svd --blend \
    --alpha 0.001 \
    --seed 42 --verbose

# α=0.002
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_73/alpha_002 \
    --sample_ids 73 \
    --inversion --svd --blend \
    --alpha 0.002 \
    --seed 42 --verbose

# α=0.004
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_73/alpha_004 \
    --sample_ids 73 \
    --inversion --svd --blend \
    --alpha 0.004 \
    --seed 42 --verbose

# α=0.006
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_sweep/sample_73/alpha_006 \
    --sample_ids 73 \
    --inversion --svd --blend \
    --alpha 0.006 \
    --seed 42 --verbose
```

#### 实验 2：SVD+默认配置对照（5 case，固定 α=0.004）

```bash
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_default_5samples \
    --sample_ids 32 50 73 80 111 \
    --inversion --svd --blend \
    --alpha 0.004 \
    --seed 42 --verbose

# 评测
cd /root/xixihaha/P-Flow && OMP_NUM_THREADS=4 python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/SVD_only_default_5samples \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir outputs/SVD_only_default_5samples/eval_clip
```

#### 批量评测（扫描完成后一次性评测所有）

```bash
# 评测所有 α 扫描结果
for case in 32 50 73 80 111; do
    for alpha_dir in /root/xixihaha/P-Flow/outputs/SVD_only_sweep/sample_${case}/alpha_*; do
        alpha_name=$(basename $alpha_dir)
        echo "=== Evaluating sample_${case}/${alpha_name} ==="
        cd /root/xixihaha/P-Flow && OMP_NUM_THREADS=4 python evaluation/run_clip_xclip_eval.py \
            --orig-dir data/videos \
            --gen-dir $alpha_dir \
            --caption-dir /root/xixihaha/test-v200/test-v200/captions \
            --output-dir $alpha_dir/eval_clip
    done
done
```

### 4.4 预期结果分析

#### 如果经验值正确（假设 B）：
- Case 73 (cos=0.42)：α=0.001 时 X-CLIP 最高，α≥0.004 时崩溃
- Case 32 (cos=0.16)：α=0.004 时 X-CLIP 最高，α=0.008 时开始退化
- → **PNA 方案 E 是对的**，需要修正方案 G 的 cap/power

#### 如果经验值有误（假设 A）：
- Case 73：α=0.004 时 X-CLIP 仍然不错（或好于 α=0.001）
- Case 32：α=0.006~0.008 时 X-CLIP 仍然不错
- → **可以给更大的 α**，需要修正经验值和场景分类阈值

### 4.5 后续方向

根据实验 1 的结果，决定：

1. **如果假设 A 正确**（经验值偏低）：
   - 修正 PNA 方案 G 的 cap/power 参数
   - 更新场景分类的 α 建议值
   - 做 β>0 实验，探索 η_spatial 对场景型的改善

2. **如果假设 B 正确**（经验值对的，FI 弥补了 SVD 的问题）：
   - 保持 PNA 方案 E 或修正方案 G 使其对齐经验值
   - FI 的场景自适应 λ_max 优化（降低场景型的 FI 注入量）
   - β>0 实验仍然有价值（η_spatial 可能更适合场景型）

---

## 五、批量执行脚本

为简化实验，建议写一个批量脚本一次跑完所有扫描：

```bash
#!/bin/bash
# SVD-only α 扫描批量脚本
# 用法: bash scripts/svd_alpha_sweep.sh

set -e
cd /root/xixihaha/P-Flow

# ── Case 32 (animal, cos=0.16): α ∈ {0.001, 0.002, 0.004, 0.006, 0.008}
for alpha in 0.001 0.002 0.004 0.006 0.008; do
    alpha_str=$(echo $alpha | sed 's/0\.//')
    echo "=== Case 32, α=$alpha ==="
    python run.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/SVD_only_sweep/sample_32/alpha_${alpha_str} \
        --sample_ids 32 \
        --inversion --svd --blend \
        --alpha $alpha \
        --seed 42 --verbose
done

# ── Case 50 (unusual, cos=0.12): α ∈ {0.001, 0.002, 0.004, 0.006, 0.008}
for alpha in 0.001 0.002 0.004 0.006 0.008; do
    alpha_str=$(echo $alpha | sed 's/0\.//')
    echo "=== Case 50, α=$alpha ==="
    python run.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/SVD_only_sweep/sample_50/alpha_${alpha_str} \
        --sample_ids 50 \
        --inversion --svd --blend \
        --alpha $alpha \
        --seed 42 --verbose
done

# ── Case 73 (scene, cos=0.42): α ∈ {0.0005, 0.001, 0.002, 0.004, 0.006}
for alpha in 0.0005 0.001 0.002 0.004 0.006; do
    alpha_str=$(echo $alpha | sed 's/0\.//' | sed 's/^0*//')
    echo "=== Case 73, α=$alpha ==="
    python run.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/SVD_only_sweep/sample_73/alpha_${alpha_str} \
        --sample_ids 73 \
        --inversion --svd --blend \
        --alpha $alpha \
        --seed 42 --verbose
done

# ── Case 80 (scene+cam, cos=0.80): α ∈ {0.0005, 0.001, 0.002, 0.004, 0.006}
for alpha in 0.0005 0.001 0.002 0.004 0.006; do
    alpha_str=$(echo $alpha | sed 's/0\.//' | sed 's/^0*//')
    echo "=== Case 80, α=$alpha ==="
    python run.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/SVD_only_sweep/sample_80/alpha_${alpha_str} \
        --sample_ids 80 \
        --inversion --svd --blend \
        --alpha $alpha \
        --seed 42 --verbose
done

# ── Case 111 (scene+cam, cos=0.30): α ∈ {0.0005, 0.001, 0.002, 0.004, 0.006}
for alpha in 0.0005 0.001 0.002 0.004 0.006; do
    alpha_str=$(echo $alpha | sed 's/0\.//' | sed 's/^0*//')
    echo "=== Case 111, α=$alpha ==="
    python run.py \
        --data_dir data/videos \
        --caption_dir /root/xixihaha/test-v200/test-v200/captions \
        --output_dir outputs/SVD_only_sweep/sample_111/alpha_${alpha_str} \
        --sample_ids 111 \
        --inversion --svd --blend \
        --alpha $alpha \
        --seed 42 --verbose
done

# ── 实验2: SVD+默认配置 (5 case, α=0.004 固定) ──
echo "=== SVD-only default (α=0.004, 5 samples) ==="
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/SVD_only_default_5samples \
    --sample_ids 32 50 73 80 111 \
    --inversion --svd --blend \
    --alpha 0.004 \
    --seed 42 --verbose

# ── 批量评测 ──
echo "=== Evaluating all sweep results ==="
for case in 32 50 73 80 111; do
    for alpha_dir in /root/xixihaha/P-Flow/outputs/SVD_only_sweep/sample_${case}/alpha_*; do
        if [ -d "$alpha_dir" ]; then
            echo "--- Evaluating $alpha_dir ---"
            OMP_NUM_THREADS=4 python evaluation/run_clip_xclip_eval.py \
                --orig-dir data/videos \
                --gen-dir $alpha_dir \
                --caption-dir /root/xixihaha/test-v200/test-v200/captions \
                --output-dir $alpha_dir/eval_clip
        fi
    done
done

# 评测实验2
echo "=== Evaluating SVD-only default ==="
OMP_NUM_THREADS=4 python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/SVD_only_default_5samples \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir outputs/SVD_only_default_5samples/eval_clip

echo "=== All experiments complete! ==="
```

---

## 六、时间估算

- 每个 case 每次生成约 4 分钟（250s）
- 5 case × 5 α = 25 次生成 → 约 100 分钟
- 实验 2（5 case × 1 次）→ 约 20 分钟
- 评测 → 约 10 分钟
- **总计约 130 分钟（2小时10分钟）**

---

## 七、预期结果与决策树

```
SVD-only α 扫描结果
    │
    ├── 场景型(73,80,111): α>0.002 时 X-CLIP 大幅下降
    │   → 假设B正确: 经验值对的，FI弥补了SVD问题
    │   → PNA应选方案E(低α), FI的λ_max需要场景自适应
    │   → β>0实验: η_spatial可能更适合场景型
    │
    ├── 场景型(73,80,111): α=0.004 时 X-CLIP 仍然OK
    │   → 假设A正确: 经验值偏低
    │   → PNA方案G的参数可以继续用
    │   → 需要修正经验值, 场景分类阈值需要调整
    │
    └── 运动型(32,50): α=0.006~0.008 时仍然OK
        → α可以更大, 运动型场景容忍度更高
        → PNA方案G的运动型分支是正确的
```
