# P-Flow: Pipeline as Flags — 渐进式视频复现框架

通过逐层叠加信息（文字 → 噪声 → 嵌入），让 T2V 模型仅凭文字和噪声重新生成与参考视频高度一致的版本。所有改动点以 CLI flag 形式组合，无需维护多个代码分支。

---

## 核心思想

| 层 | 技术 | 信息维度 | 实际执行的事 |
|----|------|---------|-------------|
| Layer 1 | V4 Hybrid Prompt Rewrite | 语义："什么在动" | LLM 将 VLM 描述改写为指导性 prompt（一次 API 调用，非迭代） |
| Layer 2 | SVD Noise Prior | 结构："从哪里开始动" | 参考视频 → VAE 编码 → Flow Inversion 反演噪声 → SVD 去内容保运动 → 与随机噪声按 α=0.004 混合 |
| Layer 3 | Velocity Field Matching | 轨迹："怎么动" | 30 步 Adam 优化 Δe，使模型速度场对齐参考视频理想轨迹，生成时注入 e₀+0.02·Δe |

三层叠加效果（10 样本验证）：CLIP +3.4%，XCLIP +8.0%（相对 baseline）。

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
│   └── captions_hybrid/             #    Layer 1 产出 (脚本自动生成)
├── outputs/                           #    实验输出 (自动创建)
├── src/
│   ├── pipeline.py                    #    统一管线 (PFlowConfig + PFlowPipeline)
│   ├── velocity_matching.py           #    Velocity Matcher (30步 Δe 优化)
│   ├── flow_matching.py               #    Flow Matching Inverter
│   ├── svd_filter.py                  #    SVD 两阶段滤波
│   ├── vlm_client.py                  #    VLM 客户端
│   ├── video_utils.py                 #    视频 I/O
│   └── distributed.py                 #    GPU 推理工具
├── scripts/
│   ├── rewrite_hybrid.py              #    Layer 1: LLM 话术改写
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
    --caption_dir data/captions_qwen \
    --output_dir outputs/step0_baseline \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 加载视频 → 读取 caption → 纯随机噪声 → Wan2.1 生成（30步）→ 输出
**预期**: CLIP ≈ 0.8703, XCLIP ≈ 0.7164

---

### Step 1：+Layer 1（Hybrid Prompt Rewrite）

```bash
# 1a. LLM 改写 caption（一次性预处理，10 个样本约 10 秒）
python scripts/rewrite_hybrid.py \
    --input-dir data/captions_qwen \
    --output-dir data/captions_hybrid \
    --backend dashscope --model qwen-plus \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --skip-existing

# 1b. 用改写后的 caption 生成视频
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_hybrid \
    --output_dir outputs/step1_L1 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 读取 hybrid caption → 纯随机噪声 → Wan2.1 生成 → 输出（和 baseline 唯一区别是换了更好的 prompt）
**预期**: CLIP ≈ 0.8842, XCLIP ≈ 0.7430 (CLIP +1.6%, XCLIP +3.7%)

---

### Step 2：+Layer 1 + Layer 2（Noise Prior）

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_hybrid \
    --output_dir outputs/step2_L1L2 \
    --noise_prior --alpha 0.004 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 读取 hybrid caption → VAE 编码参考视频为 z₀ → Flow Inversion (50步逆向ODE) 得到 η_inv → SVD 两阶段滤波 (去内容保运动) 得到 η_temporal → η = √0.004·η_temporal + √0.996·η_random → 用混合噪声生成 → 输出
**预期**: CLIP ≈ 0.8953, XCLIP ≈ 0.7667 (CLIP +2.9%, XCLIP +7.0%)

---

### Step 3：+Layer 1 + Layer 2 + Layer 3（Velocity Matching）

```bash
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_hybrid \
    --output_dir outputs/step3_L1L2L3 \
    --velocity_full --alpha 0.004 --embed_strength 0.02 \
    --velocity_K 4 --velocity_motion_weight 1.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

**执行内容**: 读取 hybrid caption → VAE 编码 z₀ → Flow Inversion 得 η_inv → SVD 得 η_temporal → **额外: 30步 Adam 优化 Δe（每步 K=4 分层时间点采样，运动加权 Loss），使 v_θ(x_t, t, e₀+Δe) ≈ v*（= z₀ - η_inv）** → 噪声混合 → **生成时通过 hook 注入 e_final = e₀ + 0.02·Δe** → 输出
**预期**: CLIP ≈ 0.8998, XCLIP ≈ 0.7736 (CLIP +3.4%, XCLIP +8.0%)

---

### 评估每一步

评测脚本自动识别 `outputs/` 下的子目录结构（`sample_{id}/{id}.mp4`），无需手动平铺文件。

```bash
# 评测单个 step
python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/step2_L1L2 \
    --caption-dir data/captions_hybrid \
    --output-dir outputs/step2_L1L2/eval_clip

# 批量评测所有 step
for step_dir in step0_baseline step1_L1 step2_L1L2 step3_L1L2L3; do
    [ -d "outputs/$step_dir" ] || continue
    echo "====== $step_dir ======"
    python evaluation/run_clip_xclip_eval.py \
        --orig-dir data/videos \
        --gen-dir outputs/$step_dir \
        --caption-dir data/captions_hybrid \
        --output-dir outputs/$step_dir/eval_clip
done
```

---

### 预期指标汇总

| Step | 配置 | CLIP (orig_gen) | XCLIP (orig_gen) | 相对 Baseline |
|------|------|----------------|-----------------|--------------|
| 0 | Baseline | 0.8703 | 0.7164 | — |
| 1 | +L1 | 0.8842 | 0.7430 | +1.6%, +3.7% |
| 2 | +L1+L2 | **0.8952** | **0.7747** | +2.9%, +8.1% |
| 3 | +L1+L2+L3 | 0.8998 | 0.7736 | +3.4%, +8.0% |

> Step 2 实测值 (2026-06-05, A800, seed=42, alpha=0.004, 10 samples)。

---

## 一键全跑（L1+L2+L3 最强配置）

如果不需要逐层验证，直接一步到位：

```bash
# 先生成 hybrid caption（如果没有的话）
python scripts/rewrite_hybrid.py \
    --input-dir data/captions_qwen \
    --output-dir data/captions_hybrid \
    --backend dashscope --model qwen-plus \
    --sample-ids 7 17 21 31 32 33 34 43 46 47 \
    --skip-existing

# 跑全量 pipeline
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_hybrid \
    --output_dir outputs/full_L1L2L3 \
    --velocity_full --alpha 0.004 --embed_strength 0.02 \
    --velocity_K 4 --velocity_motion_weight 1.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
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
| `--velocity` | Velocity Field Matching (Δe) | — |
| `--noise_prior` | 噪声先验组合 | = `--inversion --svd --blend` |
| `--velocity_full` | 全量组合 | = `--inversion --svd --blend --velocity` |
| `--full` | 全部启用 | = 上述 + `--iter 10 --composite` |

关键参数（最优值 ≠ 默认值，需显式指定）：

| 参数 | 默认值 | 最优值 | 说明 |
|------|--------|--------|------|
| `--alpha` | 0.001 | **0.004** | 噪声混合权重 |
| `--embed_strength` | 0.005 | **0.02** | Δe 注入强度 |
| `--velocity_K` | 4 | 4 | 分层采样数 |
| `--velocity_motion_weight` | 1.0 | 1.0 | 运动加权 |

---

## 硬件要求

| 项目 | 规格 |
|------|------|
| GPU | A800 80GB (推荐) / 4090 24GB (可用) |
| 模型 | Wan2.1-T2V-1.3B (~2.6GB bfloat16) |
| 分辨率 | 480×832, 81 frames, 15fps |
| Baseline 单样本 | ~30s |
| +Noise Prior | ~80s (2.7×) |
| +Velocity | ~170s (5.7×) |

---

## 文档

- **技术原理与架构**: `docs/P-Flow故事与技术演进.md`
- **逐层复现指南**: `docs/复现指南_逐层验证.md`
- **优化方向**: `docs/TODO_优化方向.md`
- **实验记录**: `docs/实验记录.md`
