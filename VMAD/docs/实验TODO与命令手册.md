# VMAD 实验 TODO 与命令手册

> **最后更新**：2025-06-01
> **环境**：AutoDL A800 80GB, Wan2.1-T2V-1.3B-Diffusers
> **数据**：10 个样本 (7, 17, 21, 31, 32, 33, 34, 43, 46, 47)

---

## 一、实验 TODO（做完打 ✅）

### 准备工作

- [x] 汇总 baseline VLM caption 到 `/root/autodl-tmp/outputs/baseline_captions/`
- [x] Velocity Field Matching v1 单样本验证（sample 43, 33帧100步, Loss 10→2.8）

### 主线实验

| # | 实验 | 状态 | 预计耗时 | 依赖 |
|---|------|------|---------|------|
| 0 | Bug 修复 + Pipeline 等价性验证 | ✅ 已完成 | 2h | 无 |
| 1 | Phase 1: α 扫描 (0.0/0.001/0.01/0.1/1.0) | ⬜ 待跑 | 1-2h | 无 |
| 2 | Phase 3a: 完整三层 baseline caption | ✅ 已完成 | 4h | Phase 1 确认 α |
| 2b | Layer 2 Strength 扫描 (alpha=0.001/0.01/0.05/0.1) | ✅ 已完成 | 1h | Phase 3a |
| 3 | Caption 消融: V4 caption + Layer 2 单样本验证 | ✅ 已完成 | 0.5h | Phase 3a |
| **3b** | **V4 caption + Layer 2 多样本验证（10样本）** | **⬜ 待跑** | **4.5h** | **#3 确认有效** |
| 4 | Phase 3b: 步数扫描 50/100/200/500 | ⬜ 待跑 | 6-8h | Phase 3a |
| 5 | Phase 3c: Position-Aware vs Uniform | ⬜ 待跑 | 3-4h | Phase 3a |
| 6 | Phase 3d: T_m 扫描 0.3/0.5/0.7/1.0 | ⬜ 待跑 | 6-8h | Phase 3a |
| 7 | Phase 4: Motion Transfer 跨内容迁移 | ⬜ 待跑 | 4-5h | Phase 3d |
| 8 | Phase 5: VLM 反馈闭环迭代 | ⬜ 待跑 | 5-6h | Phase 3a |
| 9 | Phase 2: RF-Solver 高阶反演 | ⬜ 可选 | 2-3h | Phase 1 |
| 10 | Phase 6: 高级优化方向 | ⬜ 可选 | TBD | Phase 3-5 |

### 判断门控

| 完成后 | 看什么 | 下一步决策 |
|--------|--------|-----------|
| Phase 1 跑完 | α=1.0 的 CLIP 是否 >0.95 | 是→η_inv 质量OK；否→排查 inversion |
| Phase 1 跑完 | α=0.001 vs α=0.0 的 CLIP 差值 | >0.01→有实用价值；<0.005→需做 Phase 2 RF-Solver |
| Phase 3a 跑完 | CLIP 是否超过 0.884 (P-Flow V4上限) | 是→Layer 2 验证成功，进消融；否→需调参 |
| Caption 消融跑完 | baseline vs V4 的 CLIP 差值 | baseline更好→确认方向正确；V4更好→后续改用 V4 |
| Caption 消融单样本 ✅ | D(XCLIP) > C(XCLIP) | 是(+0.0226)→Δe 在时序维度有独特价值，扩 10 样本验证 |

---

## 二、命令详解

### 通用参数说明

```
--video-dir         原始参考视频目录，里面是 {id}.mp4
                    用于：VAE 编码得到 z₀、Flow Matching Inversion 得到 η_inv

--caption-dir       caption 文本目录，里面是 {id}.txt
                    用于：T5 编码得到 e₀（Velocity Field Matching 的优化起点）
                    ⚠️ Inversion 本身用的是空字符串""，与此参数无关

--output-dir        输出总目录（资产/视频/日志全部在此下）

--sample-ids        只处理这些 ID 的视频

--alpha             η_inv 混合比例：eta = sqrt(α)·η_inv + sqrt(1-α)·η_random
                    0.0 = 纯随机噪声（等同无 inversion）
                    1.0 = 纯反演噪声（理论重建上限）
                    0.001~0.01 = 实用区间

--num_opt_steps     Velocity Field Matching 的优化步数（默认100）

--no-velocity       关闭 Layer 2 (Δe 优化)
--no-svd            关闭 SVD 频域滤波
--no-disentangle    关闭内容解耦正则化
--no-position_aware 关闭 position-aware 梯度缩放（改用 uniform）

--content SELF      Apply 阶段用视频自己的 caption 生成（复现模式）
--content "a cat"   Apply 阶段用指定内容生成（迁移模式）

--T_m               时间步上界：1.0=全步匹配（最高保真），<1=只匹配前部分（运动迁移用）

--resume            断点续跑，跳过已有结果的样本
-v                  详细日志模式
```

---

## 三、实验命令（可直接复制执行）

### Phase 1: α 扫描

**目的**：验证 Flow Matching Inversion 的质量。α=0.0 是 baseline（纯随机噪声），α=1.0 是理论上限（完美重建）。中间的 α 值找到"少量 η_inv 就能提升保真度"的甜蜜点。

**为什么关闭 velocity/svd/disentangle**：隔离变量，只测 Layer 3 (η_inv) 本身的贡献。

```bash
cd /root/autodl-tmp/videofake/VMAD

for alpha in 0.0 0.001 0.01 0.1 1.0; do
    outdir=/root/autodl-tmp/outputs/vmad_phase1_alpha${alpha}
    mkdir -p ${outdir}
    echo "========== Phase 1: alpha=${alpha} =========="
    python run_batch_extract.py \
        --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption-dir /root/autodl-tmp/outputs/baseline_captions \
        --output-dir ${outdir} \
        --sample-ids 7 17 21 31 32 33 34 43 46 47 \
        --alpha ${alpha} \
        --no-velocity \
        --no-svd \
        --no-disentangle \
        --content SELF \
        --resume \
        -v \
        2>&1 | tee ${outdir}/run.log
done
```

**评测**：
```bash
cd /root/autodl-tmp/videofake/VMAD

for alpha in 0.0 0.001 0.01 0.1 1.0; do
    dir=/root/autodl-tmp/outputs/vmad_phase1_alpha${alpha}
    mkdir -p ${dir}/flat
    # 找到生成的视频并 link 到 flat 目录
    find ${dir} -name "*.mp4" -path "*/generated/*" -exec ln -sf {} ${dir}/flat/ \;
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --gen-dir ${dir}/flat \
        --output-dir ${dir}/eval \
        2>&1 | tee ${dir}/eval.log
done
```

---

### Phase 3a: 完整三层验证（核心实验）

**目的**：在 Phase 1 确定的最优 α 上，叠加 Layer 2 (Velocity Field Matching)。这是 VMAD 相对于 P-Flow 的核心增量——如果 CLIP 超过 0.884，说明 Δe 优化有效。

**为什么用 baseline caption**：让 Δe 承担"文本无法表达的残差"编码，Layer 1 和 Layer 2 职责分离。

```bash
cd /root/autodl-tmp/videofake/VMAD

outdir=/root/autodl-tmp/outputs/vmad_phase3a_full
mkdir -p ${outdir}
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir} \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.001 \
    --num_opt_steps 200 \
    --no-svd \
    --no-disentangle \
    --content SELF \
    --resume \
    -v \
    2>&1 | tee ${outdir}/run.log
```

---

### Caption 消融: V4 caption + Layer 2（✅ 单样本已完成，待扩 10 样本）

**目的**：定量回答"如果 e₀ 已经被 V4 优化过，Δe 还能做多少增量？"

**单样本结论（sample #7）**：V4 caption 主攻 CLIP（+0.0247），Δe 主攻 XCLIP（+0.0226）。两者互补，但贡献维度不同。

**10 样本验证命令**（待跑，预计 4.5h）：

```bash
cd /root/autodl-tmp/videofake/VMAD

# 实验组：V4 caption + Layer 2 α=0.01（10 样本）
outdir=/root/autodl-tmp/outputs/vmad_v4caption_l2_10samples
mkdir -p ${outdir}
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir} \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.01 \
    --num_opt_steps 200 \
    --no-svd --no-disentangle --no-blend --no-token_decode \
    --content SELF \
    --resume -v \
    2>&1 | tee ${outdir}/run.log

# 对照组：V4 caption + 三层全关（10 样本）
outdir_ctrl=/root/autodl-tmp/outputs/vmad_v4caption_pure_10samples
mkdir -p ${outdir_ctrl}
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir_ctrl} \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.01 \
    --num_opt_steps 200 \
    --no-svd --no-disentangle --no-blend --no-token_decode --no-velocity \
    --content SELF \
    --resume -v \
    2>&1 | tee ${outdir_ctrl}/run.log

# 评测
python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir}/generated \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir}/eval

python evaluation/run_reproduction_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${outdir_ctrl}/generated \
    --caption-dir /root/autodl-tmp/outputs/hybrid_iter_v4/captions_iter0 \
    --output-dir ${outdir_ctrl}/eval
```

---

### Phase 3b: 优化步数扫描

**目的**：找到"步数-效果"的最优平衡点。v1 实验显示 50-60 步后基本收敛，但那是 33 帧的情况。81 帧可能需要更多步。

```bash
cd /root/autodl-tmp/videofake/VMAD

for steps in 50 100 200 500; do
    outdir=/root/autodl-tmp/outputs/vmad_phase3b_steps${steps}
    mkdir -p ${outdir}
    echo "========== Phase 3b: steps=${steps} =========="
    python run_batch_extract.py \
        --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption-dir /root/autodl-tmp/outputs/baseline_captions \
        --output-dir ${outdir} \
        --sample-ids 7 17 21 31 32 33 34 43 46 47 \
        --alpha 0.001 \
        --num_opt_steps ${steps} \
        --no-svd \
        --no-disentangle \
        --content SELF \
        --resume \
        -v \
        2>&1 | tee ${outdir}/run.log
done
```

---

### Phase 3c: Position-Aware vs Uniform 消融

**目的**：验证 P-Flow 位置实验的结论（position 0 权重 10-15×）在 Layer 2 中是否也有效果。如果 position-aware 明显优于 uniform，说明这个设计有价值。

```bash
cd /root/autodl-tmp/videofake/VMAD

# Position-aware（默认，已在 Phase 3a 跑过）
# 这里只需跑 Uniform 对照组：

outdir=/root/autodl-tmp/outputs/vmad_phase3c_uniform
mkdir -p ${outdir}
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir} \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.001 \
    --num_opt_steps 200 \
    --no-svd \
    --no-disentangle \
    --no-position_aware \
    --content SELF \
    --resume \
    -v \
    2>&1 | tee ${outdir}/run.log
```

---

### Phase 3d: T_m 时间步上界扫描

**目的**：T_m 控制 Δe 对哪些去噪步骤负责。T_m=1.0 是全步匹配（最高保真），T_m<1 只匹配前部分步骤（只捕获运动结构，用于后续跨内容迁移）。

```bash
cd /root/autodl-tmp/videofake/VMAD

for tm in 0.3 0.5 0.7 1.0; do
    outdir=/root/autodl-tmp/outputs/vmad_phase3d_tm${tm}
    mkdir -p ${outdir}
    echo "========== Phase 3d: T_m=${tm} =========="
    python run_batch_extract.py \
        --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
        --caption-dir /root/autodl-tmp/outputs/baseline_captions \
        --output-dir ${outdir} \
        --sample-ids 7 17 21 31 32 33 34 43 46 47 \
        --alpha 0.001 \
        --num_opt_steps 200 \
        --T_m ${tm} \
        --no-svd \
        --no-disentangle \
        --content SELF \
        --resume \
        -v \
        2>&1 | tee ${outdir}/run.log
done
```

---

### Phase 4: Motion Transfer 跨内容迁移

**目的**：验证 VMAD 的独特能力——用 T_m=0.5 提取的 Δe 只编码运动，应用到新主体上。

```bash
cd /root/autodl-tmp/videofake/VMAD

outdir=/root/autodl-tmp/outputs/vmad_phase4_transfer
mkdir -p ${outdir}
python run_batch_extract.py \
    --video-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --caption-dir /root/autodl-tmp/outputs/baseline_captions \
    --output-dir ${outdir} \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --alpha 0.0 \
    --num_opt_steps 200 \
    --T_m 0.5 \
    --no-svd \
    --no-disentangle \
    --content "a white cat" "a robot" "a teddy bear" "an astronaut" "a goldfish" \
    --cross-content \
    --resume \
    -v \
    2>&1 | tee ${outdir}/run.log
```

注意：这里 `--alpha 0.0` 因为迁移时不能用原视频的 η_inv（那是原内容的空间结构）。

---

## 四、通用评测命令

```bash
# 对任意输出目录做评测
dir=/root/autodl-tmp/outputs/<实验目录>
mkdir -p ${dir}/flat
find ${dir} -name "*.mp4" -path "*/generated/*" -exec ln -sf {} ${dir}/flat/ \;
python evaluation/run_clip_xclip_eval.py \
    --orig-dir /root/autodl-tmp/data/video-200/water_mark_out \
    --gen-dir ${dir}/flat \
    --output-dir ${dir}/eval \
    2>&1 | tee ${dir}/eval.log
```

---

## 五、实验结果记录区

### Phase 1: α 扫描结果

| α | CLIP | Δ CLIP (vs α=0.0) | XCLIP | Δ XCLIP | 备注 |
|---|------|-------------------|-------|---------|------|
| 0.0 | | | | | baseline（纯随机） |
| 0.001 | | | | | |
| 0.01 | | | | | |
| 0.1 | | | | | |
| 1.0 | | | | | 理论重建上限 |

### Bug 修复 + Pipeline 等价性验证（sample 7）

| 实验配置 | orig_gen_clip | orig_gen_xclip | 说明 |
|----------|--------------|----------------|------|
| P-Flow baseline | 0.9159 | 0.7108 | 参考锚点 |
| P-Flow reseed (同 caption/seed) | 0.9159 | 0.7108 | 确认可复现 |
| VMAD 原版（修复前） | 0.8007 | — | Bug 1+2 |
| VMAD --no-blend 修复后 | 0.8986 | 0.6384 | Bug 2 残留 |
| **VMAD 纯净基线（三层全关）** | **0.9150** | **0.7077** | **= P-Flow ✅** |

### Layer 2 (Δe) Strength 扫描结果（sample 7）

| Alpha | Effective Strength | orig_gen_clip | orig_gen_xclip | Δ CLIP vs 基线 |
|-------|-------------------|--------------|----------------|---------------|
| 0 (基线) | 0 | **0.9150** | **0.7077** | — |
| 0.001 | 0.001 | 0.9086 | 0.6857 | -0.0064 |
| **0.01** | **0.01** | **0.9192** | **0.6987** | **+0.0042** |
| 0.05 | 0.05 | 0.8999 | 0.5925 | -0.0151 |
| 0.1 | 0.1 | 0.9016 | 0.5670 | -0.0134 |

**结论**：alpha=0.01 是当前最优注入强度，CLIP 超过 P-Flow baseline。曲线呈倒 U 形。

### Phase 3a: 完整三层结果

| 配置 | CLIP | XCLIP | 备注 |
|------|------|-------|------|
| P-Flow V4 iter1 (Layer 1上限) | 0.8842 | 0.7430 | 锚点 |
| Phase 1 最优 α (仅 L1+L3) | — | — | 待跑 |
| Phase 3a (L1+L2+L3, baseline caption) | — | — | 需以 alpha=0.01 重跑 |
| Phase 3a (L1+L2+L3, V4 caption) | — | — | 待跑 |

### Phase 3b/3c/3d 消融结果

（待填）

---

## 六、关键路径总览

```
/root/autodl-tmp/
├── data/video-200/water_mark_out/          # 原始视频 {id}.mp4
├── outputs/
│   ├── baseline/sample_{id}/vlm_caption.txt  # baseline VLM caption (分散)
│   ├── baseline_captions/{id}.txt            # baseline VLM caption (已汇总✅)
│   ├── hybrid_iter_v4/captions_iter0/{id}.txt # V4 LLM 改写 caption
│   ├── vmad_phase1_alpha{X}/                 # Phase 1 输出
│   ├── vmad_phase3a_full/                    # Phase 3a 输出
│   ├── vmad_phase3a_v4caption/               # Caption 消融输出
│   ├── vmad_phase3b_steps{N}/                # Phase 3b 输出
│   ├── vmad_phase3c_uniform/                 # Phase 3c 输出
│   ├── vmad_phase3d_tm{X}/                   # Phase 3d 输出
│   └── vmad_phase4_transfer/                 # Phase 4 输出
└── videofake/VMAD/                           # 代码仓库
```
