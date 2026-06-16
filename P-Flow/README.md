# P-Flow: Pipeline as Flags — 渐进式视频复现框架

通过逐层叠加信息（文字 → 噪声 → 轨迹引导），让 T2V 模型仅凭文字和噪声重新生成与参考视频高度一致的版本。所有改动点以 CLI flag 形式组合，无需维护多个代码分支。

---

## 核心思想

| 层 | 技术 | 信息维度 | 实际执行的事 |
|----|------|---------|-------------|
| Layer 1 | v9 Subtract+Supplement | 语义："什么在动" | Step1: LLM 纯删减（去 preamble/hedging/summary，不碰运动描述）→ Step2: VLM 看视频补充真实视觉细节（颜色/材质/光照/空间关系） |
| Layer 2 | SVD Noise Prior | 结构："从哪里开始动" | 参考视频 → VAE 编码 → Flow Inversion 反演噪声 → SVD 去内容保运动 → 与随机噪声按 α=0.004 混合 |
| Layer 3 | Feature Injection (FI) | 特征："往哪个方向去噪" | 反演时 inline hook 缓存 DiT 中间层特征，生成时以残差方式注入：h = (1-λ)·h_gen + λ·h_ref |
---

## 目录结构

```
P-Flow/
├── models/                            # ← 需要准备 (见下方说明)
│   ├── Wan2.1-T2V-1.3B-Diffusers/    #    T2V 生成模型
│   ├── Qwen2.5-VL-7B-Instruct/      #    VLM 模型
│   ├── clip-vit-base-patch32/        #    CLIP 评测模型
│   └── xclip-base-patch32/           #    X-CLIP 评测模型
├── data/
│   ├── videos/                        # ← 需要准备: 参考视频 ({id}.mp4)
│   ├── captions_qwen/                #    VLM 原始 caption ({id}.txt)
│   └── captions_v9/                 #    Layer 1 产出: v9 改写 (脚本自动生成)
├── outputs/                           #    实验输出 (自动创建)
├── src/
│   ├── pipeline.py                    #    统一管线 (PFlowConfig + PFlowPipeline)
│   ├── flow_matching.py               #    Flow Matching Inverter
│   ├── svd_filter.py                  #    SVD 两阶段滤波
│   ├── vlm_client.py                  #    VLM 客户端
│   ├── video_utils.py                 #    视频 I/O
│   └── distributed.py                 #    GPU 推理工具
├── scripts/
│   ├── rewrite_minimal.py             #    Layer 1: v9 Subtract+Supplement 改写
│   ├── reproduce.sh                   #    一键复现脚本
│   └── ...
├── evaluation/
│   ├── run_clip_xclip_eval.py         #    CLIP/X-CLIP 评估
│   └── run_stream_eval.py             #    流式评估
├── docs/
│   ├── P-Flow故事与技术演进.md         #    技术文档 (架构/原理/演进)
│   └── 复现指南_逐层验证.md            #    逐层复现详细指南
├── run.py                             #    CLI 入口
└── requirements.txt
```

---

## Quick Start

### 1. 安装依赖

```bash
cd P-Flow
pip install -r requirements.txt
```

### 2. 准备模型

```bash
mkdir -p models

# 软链接到实际模型位置 (根据你的环境修改路径)
ln -sf /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers models/Wan2.1-T2V-1.3B-Diffusers
ln -sf /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct

# 评测模型 (CLIP / X-CLIP)
ln -sf /root/autodl-tmp/models/clip-vit-base-patch32 models/clip-vit-base-patch32
ln -sf /root/autodl-tmp/models/xclip-base-patch32 models/xclip-base-patch32
```

### 3. 准备数据

```bash
mkdir -p data/videos

# 将参考视频放入，命名为 {id}.mp4
ln -sf /root/autodl-tmp/data/video-200/water_mark_out/* data/videos/
```

如果 `data/captions_qwen/` 下还没有原始 caption，pipeline 会自动调用 VLM 生成（需要本地 VLM 模型或 DashScope API）。

### 4. 配置 API Key

```bash
export DASHSCOPE_API_KEY="your-key"   # Layer 1 改写需要
```

---

## 逐层验证（从 Layer 1 到 Layer 3）

以下命令逐层叠加，让你看到每一层的独立贡献。

### Step 0：Baseline（VLM 原始 caption 直出）

```bash
# 用 VLM 原始 caption + 纯随机噪声生成
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/step0_baseline \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 加载视频 → 读取 caption → 纯随机噪声 → Wan2.1 生成（30步）→ 输出
**预期**: CLIP ≈ 0.8703, XCLIP ≈ 0.7164

---

### Step 1：+Layer 1（v9 Subtract+Supplement）

```bash
# 1a. LLM 纯删减 + VLM 视觉补充（一次性预处理）
python scripts/rewrite_minimal.py \
    --input-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir data/captions_v9 \
    --video-dir data/videos \
    --backend dashscope --model qwen-plus \
    --vlm-provider dashscope --vlm-model qwen-vl-max \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --skip-existing

# 1b. 用改写后的 caption 生成视频
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_v9 \
    --output_dir outputs/step1_L1 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: Step1 LLM 纯删减（去 preamble/hedging/summary，不碰运动描述）→ Step2 VLM 看原始视频补充真实视觉细节 → 用改写后 caption + 纯随机噪声 → Wan2.1 生成 → 输出
**预期**: CLIP ≈ 0.8947, XCLIP ≈ 0.7973 (vs baseline: CLIP +2.2%, XCLIP +6.4%)

---

### Step 2：+Layer 2（SVD Noise Prior）

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/step2_L2 \
    --noise_prior --alpha 0.004 --svd_mode v1 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 读取 VLM 原始 caption → VAE 编码参考视频为 z₀ → Flow Inversion (50步逆向ODE) 得到 η_inv → SVD v1 两阶段滤波 (去内容保运动) 得到 η_temporal (std≈0.3) → η = √0.004·η_temporal + √0.996·η_random → 用混合噪声生成 → 输出
**预期**: CLIP ≈ 0.8964, XCLIP ≈ 0.7874 (CLIP +2.4%, XCLIP +5.1%)

> **重要**: L2 使用 `--svd_mode v1`（不做 renorm），配合裸 VLM caption 效果最佳。v1 模式保留 raw SVD 输出（std≈0.3），在 α=0.004 下形成最优的"低剂量方向偏置"。L1 改写与 L2 存在结构性矛盾（精确运动描述 + SVD 偏置互相拉扯），因此 L2 独立使用时不叠加 L1。

---

### Step 3：+Layer 2 + Layer 3（SVD + Feature Injection）

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/step3_L2L3_FI \
    --inversion --svd --blend --alpha 0.004 --svd_mode v1 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --verbose
```

**执行内容**: 读取 VLM 原始 caption → VAE 编码参考视频 → Flow Inversion (50步) → **反演时 inline hook 缓存 DiT 中间层特征** → SVD v1 两阶段滤波 → η = √0.004·η_temporal + √0.996·η_random → 混合噪声生成 → **每步 DiT forward 注入参考特征: h = (1-λ_eff)·h_gen + λ_eff·h_ref**（自适应门控 + 质量门控）→ 输出
**预期**: CLIP ≈ 0.9042, XCLIP ≈ 0.8138 (10样本, vs Baseline CLIP +3.9%, XCLIP +13.6%)

> 注：L2+L3(FI) 使用裸 VLM caption（`captions_qwen`）效果最佳。L1(v9) 可独立使用提升 XCLIP +1.3%，但与 L2 叠加时存在运动描述冲突，因此当前最强配置为 L2+L3 不叠加 L1 改写。

---

### 评估每一步

评测脚本自动识别 `outputs/` 下的子目录结构（`sample_{id}/{id}.mp4`），无需手动平铺文件。

```bash
# 评测 L2 (SVD Noise Prior)
python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/step2_L2 \
    --caption-dir data/captions_qwen \
    --output-dir outputs/step2_L2/eval_clip

# 评测 L2+L3(FI)
python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/step3_L2L3_FI \
    --caption-dir data/captions_qwen \
    --output-dir outputs/step3_L2L3_FI/eval_clip

# 批量评测所有 step
for step_dir in step0_baseline step1_L1 step2_L2 step3_L2L3_FI; do
    [ -d "outputs/$step_dir" ] || continue
    echo "====== $step_dir ======"
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/$step_dir \
        --caption-dir data/captions_qwen \
        --output-dir outputs/$step_dir/eval_clip
done
```

---

### 预期指标汇总

| Step | 配置 | CLIP (orig_gen) | XCLIP (orig_gen) | 相对 Baseline | 备注 |
|------|------|----------------|-----------------|--------------|------|
| 0a | Baseline (InternVL2 caption) | 0.8753 | 0.7491 | — | 6-11周会 #0 |
| 0b | Baseline (Qwen-VL caption) | 0.8703 | 0.7164 | — | FI 实验 baseline |
| 1 | +L1 (v9 Subtract+Supplement) | 0.8947 | 0.7973 | +2.2%, +6.4% | vs 0a |
| 2 | +L2 (SVD v1, α=0.004) | **0.8964** | **0.7874** | +2.4%, +5.1% | vs 0a |
| 3 | +L2+L3(FI) (当前最强) | **0.9042** | **0.8138** | +3.9%, +13.6% | vs 0b |

> ⚠️ **Baseline 说明**：Step 1/2 数据来自 InternVL2 caption 实验（baseline 0a）；Step 3 (FI) 使用 Qwen-VL caption（baseline 0b）。两组 baseline 不同（XCLIP 差 0.03），百分比不可直接跨行比较。待统一 caption 源后更新。Step 3 实测值 (2026-06-15, A800, seed=42, α=0.004, λ=0.05, layers=mid, 10 samples)。

---

## 一键全跑（L2+L3(FI) 当前最强配置）

如果不需要逐层验证，直接一步到位：

```bash
# L2+L3(FI) — 使用裸 VLM caption，无需 L1 改写
cd /root/xixihaha/P-Flow && python run.py \
    --data_dir data/videos \
    --caption_dir /root/xixihaha/test-v200/test-v200/captions \
    --output_dir outputs/FI_clean_code_3samples \
    --sample_ids 7 17 21 \
    --inversion --svd --blend --alpha 0.004 \
    --feature_inject --fi_layers mid --fi_lambda 0.05 \
    --fi_schedule middle_peak --fi_cache_mode attention \
    --seed 42 --verbose

python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/FI_clean_code_3samples \
    --caption-dir /root/xixihaha/test-v200/test-v200/captions \
    --output-dir outputs/FI_clean_code_3samples/eval_clip
```

或者用一键脚本（含逐层验证 + 自动评估）：

```bash
bash scripts/reproduce.sh
```

---

## Flag 体系

| Flag | 作用 | 快捷等价 |
|------|------|---------|
| `--inversion` | Flow Matching Inversion | — |
| `--svd` | SVD 两阶段滤波 | — |
| `--blend` | 噪声混合 (α 权重) | — |
| `--noise_prior` | 噪声先验组合 | = `--inversion --svd --blend` |
| `--feature_inject` | Feature Injection 特征注入 (L3) | — |
| `--full` | 全部启用 | = `--noise_prior` + `--iter 10 --composite` |

关键参数（最优值 ≠ 默认值，需显式指定）：

| 参数 | 默认值 | 最优值 | 说明 |
|------|--------|--------|------|
| `--alpha` | 0.001 | **0.004** | 噪声混合权重 |
| `--fi_lambda` | 0.1 | **0.05** | FI 注入强度 λ |
| `--fi_layers` | all | **mid** | 注入层 (mid=10~19层) |
| `--fi_schedule` | middle_peak | middle_peak | λ 调度策略 |
| `--fi_cache_mode` | attention | attention | 缓存特征类型 |

---

## 硬件要求

| 项目 | 规格 |
|------|------|
| GPU | A800 80GB (推荐) / 4090 24GB (可用) |
| 模型 | Wan2.1-T2V-1.3B (~2.6GB bfloat16) |
| 分辨率 | 480×832, 81 frames, 15fps |
| Baseline 单样本 | ~30s |
| +Noise Prior (L2) | ~80s (2.7×) |
| +L2+L3(FI) | ~230s (7.7×) |

---

## 文档

- **技术原理与架构**: `docs/P-Flow故事与技术演进.md`
- **逐层复现指南**: `docs/复现指南_逐层验证.md`
- **优化方向**: `docs/TODO_优化方向.md`
- **实验记录**: `docs/实验记录.md`
