# P-Flow: Pipeline as Flags — 渐进式视频复现框架

通过逐层叠加信息（文字 → 噪声 → 嵌入），让 T2V 模型仅凭文字和噪声重新生成与参考视频高度一致的版本。所有改动点以 CLI flag 形式组合，无需维护多个代码分支。

---

## 核心思想

| 层 | 技术 | 信息维度 | Flag |
|----|------|---------|------|
| Layer 1 | V4 Hybrid Prompt Rewrite | 语义："什么在动" | 外部预处理 |
| Layer 2 | SVD Noise Prior | 结构："从哪里开始动" | `--noise_prior` |
| Layer 3 | Velocity Field Matching | 轨迹："怎么动" | `--velocity_full` |

三层叠加效果（10 样本验证）：CLIP +3.4%，XCLIP +8.0%（相对 baseline）。

---

## 目录结构

```
P-Flow/
├── models/                            # ← 需要准备 (见下方说明)
│   ├── Wan2.1-T2V-1.3B-Diffusers/    #    T2V 生成模型
│   └── Qwen2.5-VL-7B-Instruct/      #    VLM 模型
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
ln -s /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers models/Wan2.1-T2V-1.3B-Diffusers
ln -s /root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct models/Qwen2.5-VL-7B-Instruct
```

### 3. 准备数据

```bash
mkdir -p data/videos

# 将参考视频放入，命名为 {id}.mp4
# 例如: 从已有数据集软链接
ln -sf /root/autodl-tmp/data/video-200/water_mark_out/* data/videos/
```

如果 `data/captions_qwen/` 下还没有原始 caption，pipeline 会自动调用 VLM 生成（需要本地 VLM 模型或 DashScope API）。

### 4. 配置 API Key

```bash
export DASHSCOPE_API_KEY="your-key"   # Layer 1 改写需要
```

### 5. 运行

```bash
# 最简单: 单视频 baseline
python run.py --video data/videos/31.mp4 --caption "a cat jumping"

# 最强配置: Layer 1 + 2 + 3
python run.py \
    --video data/videos/31.mp4 \
    --caption "$(cat data/captions_hybrid/31.txt)" \
    --velocity_full --alpha 0.004 --embed_strength 0.02

# 批量 10 样本 + 全部层
python run.py \
    --data_dir data/videos \
    --caption_dir data/captions_hybrid \
    --output_dir outputs/full_run \
    --velocity_full --alpha 0.004 --embed_strength 0.02 \
    --velocity_K 4 --velocity_motion_weight 1.0 \
    --sample_ids 7 17 21 31 32 33 34 43 46 47 \
    --seed 42 --resume
```

### 6. 一键逐层复现

```bash
bash scripts/reproduce.sh
```

此脚本会依次运行 Baseline → L1 → L1+L2 → L1+L2+L3v1 → L1+L2+L3v2，并自动评估每步指标。详见 `docs/复现指南_逐层验证.md`。

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

## 评估

```bash
python evaluation/run_clip_xclip_eval.py \
    --orig-dir data/videos \
    --gen-dir outputs/<experiment> \
    --caption-dir data/captions_hybrid \
    --output-dir outputs/<experiment>/eval_clip
```

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
