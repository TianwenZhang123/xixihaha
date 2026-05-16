# P-Flow + VISTA 消融实验操作手册

## 一、环境要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 | 备注 |
|------|---------|---------|------|
| GPU | NVIDIA A100 40GB | NVIDIA A100 80GB × 2 | Wan 2.1 14B 模型需约 28GB 显存 |
| 内存 | 64GB | 128GB | VAE 编解码和 SVD 计算需大量内存 |
| 磁盘 | 200GB SSD | 500GB NVMe SSD | 模型权重约 56GB + 中间视频输出 |
| CPU | 8 核 | 16 核以上 | SVD 分解有 CPU fallback |

> **注意**: Mock 模式（不加载真实模型）可在任何 CPU 机器上运行，适合调试流程。

### 1.2 软件环境

```
操作系统: Linux (推荐 Ubuntu 22.04) 或 macOS
Python:   3.10 - 3.12
CUDA:     12.1+ (配合 PyTorch 2.1+)
FFmpeg:   4.0+ (imageio-ffmpeg 会自动下载)
```

### 1.3 Python 依赖

核心依赖（完整列表见 `requirements.txt`）：

```
torch>=2.1.0              # 深度学习框架
diffusers>=0.31.0         # Wan 2.1 视频生成 pipeline
transformers>=4.40.0      # 文本编码器
openai>=1.12.0             # VLM API (OpenAI兼容格式，走中转站)
scipy>=1.11.0             # SVD 计算
imageio>=2.31.0           # 视频 I/O
eva-decord>=0.6.0         # 高性能视频解码（兼容 Python 3.12）
```

### 1.4 外部 API 依赖

| 服务 | 用途 | 获取方式 |
|------|------|---------|
| LinkAPI 中转站 | VLM prompt 优化 (gemini-2.0-flash) | https://api.linkapi.org |
| HuggingFace | 下载 Wan 2.1 模型权重 | https://huggingface.co/settings/tokens |

---

## 二、环境搭建

### 2.1 创建虚拟环境

```bash
# 方式一: conda（推荐）
conda create -n pflow python=3.11 -y
conda activate pflow

# 方式二: venv
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2.2 安装 PyTorch（需匹配 CUDA 版本）

```bash
# CUDA 12.1
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu124

# macOS (MPS 加速，仅测试用)
pip install torch==2.3.0 torchvision==0.18.0
```

### 2.3 安装项目依赖

```bash
cd /path/to/videofake
pip install -e .
# 或
pip install -r requirements.txt
```

### 2.4 配置 API Key

```bash
# 方式一: 环境变量（推荐）
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

# 方式二: 写入 .env 文件
echo 'OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>' > .env

# 运行时指定中转站地址
python scripts/run_pflow.py --vlm_base_url "https://api.linkapi.org/v1" ...
```

### 2.5 下载模型权重

```bash
# 需要先登录 HuggingFace
huggingface-cli login

# 预下载 Wan 2.1 14B（约 56GB，首次运行也会自动下载）
python -c "from diffusers import WanPipeline; WanPipeline.from_pretrained('Wan-AI/Wan2.1-T2V-14B', torch_dtype='bfloat16')"
```

### 2.6 验证安装

```bash
# 快速检查所有模块能否正常导入
python -c "
from pflow import PFlowPipeline, PromptOptimizer, VISTAOptimizer, NoisePriorEnhancement
print('All modules imported successfully')
"

# Mock 模式完整流程测试（不需要 GPU 和 API）
python scripts/run_ablation.py \
    --reference_video dummy.mp4 \
    --prompt "test particles effect" \
    --ablation vista_full \
    --output_dir outputs/verify_install \
    --mock
```

---

## 三、准备实验数据

### 3.1 参考视频要求

P-Flow 论文使用 480×832 分辨率、81 帧（约 5 秒 @16fps）的视频：

| 属性 | 推荐值 | 说明 |
|------|--------|------|
| 分辨率 | 480×832 | 宽高比约 16:9 |
| 帧数 | 81 | 配合 Wan 2.1 默认设置 |
| FPS | 16 | 模型训练时使用的帧率 |
| 格式 | MP4 (H.264) | 兼容性最佳 |
| 内容 | 包含明确视觉特效 | 粒子、光效、运动模糊等 |

### 3.2 数据集组织

```
data/
├── reference_videos/
│   ├── particles_glow.mp4       # 发光粒子特效
│   ├── fire_explosion.mp4       # 火焰爆炸
│   ├── water_ripple.mp4         # 水波纹
│   ├── smoke_trail.mp4          # 烟雾轨迹
│   └── light_streak.mp4         # 光线条纹
├── prompts.json                 # 每个视频对应的 prompt
└── metadata.json                # 视频元信息
```

`prompts.json` 格式：

```json
{
  "particles_glow": {
    "prompt": "Glowing particles floating upward with dynamic light trails",
    "description": "Golden particles rise from bottom, leave fading glow trails"
  },
  "fire_explosion": {
    "prompt": "Explosive fire burst with expanding shockwave and flying debris",
    "description": "Central explosion with radial shockwave ring and scattered sparks"
  }
}
```

### 3.3 视频预处理（如需调整分辨率）

```bash
# 使用 ffmpeg 将视频调整到标准分辨率
ffmpeg -i input.mp4 -vf "scale=832:480,fps=16" -frames:v 81 -c:v libx264 output.mp4
```

---

## 四、运行实验

### 4.1 单次 P-Flow 完整运行

```bash
python scripts/run_pflow.py \
    --reference_video data/reference_videos/particles_glow.mp4 \
    --prompt "Glowing particles floating upward with dynamic light trails" \
    --output_dir outputs/pflow_particles \
    --config config/default.yaml \
    --seed 42
```

### 4.2 单次 VISTA 模式运行

使用消融脚本直接选择 VISTA 配置：

```bash
python scripts/run_ablation.py \
    --reference_video data/reference_videos/particles_glow.mp4 \
    --prompt "Glowing particles floating upward with dynamic light trails" \
    --ablation vista_full \
    --output_dir outputs/vista_particles \
    --seed 42
```

### 4.3 完整消融实验（核心实验）

```bash
# 运行所有 8 种消融配置
python scripts/run_ablation.py \
    --reference_video data/reference_videos/particles_glow.mp4 \
    --prompt "Glowing particles floating upward with dynamic light trails" \
    --ablation all \
    --output_dir outputs/ablation_particles \
    --seed 42
```

8 种消融配置：

| 配置名 | 描述 | 启用的 VISTA 组件 |
|--------|------|-------------------|
| `pflow_original` | P-Flow 原始优化器（baseline） | 无 |
| `vista_full` | 完整 VISTA 框架 | SVPP + Tournament + MMAC + DTPA |
| `vista_no_svpp` | 去掉场景规划 | Tournament + MMAC + DTPA |
| `vista_no_tournament` | 去掉锦标赛选择 | SVPP + MMAC + DTPA |
| `vista_no_mmac` | 去掉多智能体评判 | SVPP + Tournament + DTPA |
| `vista_no_dtpa` | 去掉深度思考 | SVPP + Tournament + MMAC |
| `vista_mmac_only` | 仅 MMAC | MMAC |
| `vista_dtpa_only` | 仅 DTPA | DTPA |

### 4.4 批量实验（多个视频）

```bash
#!/bin/bash
# batch_ablation.sh - 对所有参考视频运行消融实验

VIDEOS_DIR="data/reference_videos"
PROMPTS_FILE="data/prompts.json"
OUTPUT_BASE="outputs/full_study"
SEED=42

for video in $VIDEOS_DIR/*.mp4; do
    name=$(basename "$video" .mp4)
    prompt=$(python -c "import json; print(json.load(open('$PROMPTS_FILE'))['$name']['prompt'])")
    
    echo "=========================================="
    echo "Running ablation for: $name"
    echo "=========================================="
    
    python scripts/run_ablation.py \
        --reference_video "$video" \
        --prompt "$prompt" \
        --ablation all \
        --output_dir "$OUTPUT_BASE/$name" \
        --seed $SEED
done

echo "All experiments complete!"
```

### 4.5 评估实验结果

```bash
# 单个视频对评估
python scripts/evaluate.py \
    --reference_video data/reference_videos/particles_glow.mp4 \
    --generated_video outputs/ablation_particles/vista_full/best_result.mp4 \
    --metrics dynamic_degree \
    --output outputs/eval_results.json

# 批量评估
python scripts/evaluate.py \
    --reference_dir data/reference_videos/ \
    --generated_dir outputs/ablation_particles/vista_full/ \
    --metrics fid_vid fvd dynamic_degree \
    --output outputs/eval_batch.json
```

---

## 五、超参数调优

### 5.1 P-Flow 核心超参数

```yaml
# config/default.yaml 中可调参数

noise_prior:
  alpha: 0.001      # 噪声混合权重，越小越依赖新随机噪声
                    # 建议范围: 0.0001 - 0.01
  rho_s: 0.1       # 空间 SVD 保留比例（去除前 10% 空间分量）
                    # 建议范围: 0.05 - 0.2
  rho_m: 0.9       # 时序 SVD 保留比例（保留前 90% 时序分量）
                    # 建议范围: 0.7 - 0.95

prompt_optimization:
  max_iterations: 10  # 优化迭代次数
                      # P-Flow 论文使用 10 次，VISTA 使用 4-5 次
```

### 5.2 VISTA 特有参数

在代码中通过 `VISTAOptimizer` 构造函数设置：

```python
optimizer = VISTAOptimizer(
    max_iterations=5,            # VISTA 论文建议 4-5 次
    candidates_per_iteration=3,  # 每轮生成候选视频数（用于 Tournament）
    video_duration=5.0,          # 视频时长（用于 SVPP 场景划分）
)
```

### 5.3 参数敏感性实验

```bash
# 测试不同 alpha 值
for alpha in 0.0001 0.0005 0.001 0.005 0.01; do
    python scripts/run_pflow.py \
        --reference_video data/reference_videos/particles_glow.mp4 \
        --prompt "Glowing particles floating upward" \
        --alpha $alpha \
        --output_dir outputs/alpha_sweep/alpha_${alpha} \
        --seed 42
done
```

---

## 六、输出文件结构

```
outputs/ablation_particles/
├── ablation_summary.json          # 所有配置的汇总对比
├── pflow_original/
│   ├── ablation_results.json      # 该配置的实验结果
│   ├── reference.mp4              # 参考视频副本
│   ├── best_result.mp4            # 最佳生成视频
│   ├── generated_iter_001.mp4     # 每轮生成的视频
│   ├── generated_iter_002.mp4
│   ├── prompts_history.json       # prompt 演化历史
│   ├── convergence.json           # 收敛信息
│   └── optimization_log/
│       ├── iter_001.json          # 每轮优化详情
│       └── iter_002.json
├── vista_full/
│   ├── ablation_results.json
│   ├── vista_optimization_log/
│   │   ├── vista_iter_001.json    # VISTA 特有：含 MMAC/DTPA 详情
│   │   └── vista_iter_002.json
│   └── vista_composites/          # 评判用的对比视频
├── vista_no_mmac/
│   └── ...
└── ...
```

---

## 七、常见问题与排错

### Q1: CUDA OOM (显存不足)

```bash
# 方案一: 启用 CPU offload（已默认开启）
# 方案二: 减小视频分辨率
python scripts/run_pflow.py --height 320 --width 576 --num_frames 49 ...

# 方案三: 使用 8-bit 量化 (需 bitsandbytes)
pip install bitsandbytes
```

### Q2: Gemini API 报错 429 (Rate Limit)

VISTA 模式每轮需要多次 VLM 调用（MMAC 9 次 + DTPA 1 次），容易触发限流：

```
解决方案:
1. 使用 Gemini 2.0 Flash（配额更高、速度更快）
2. 在 VLM client 中增加 retry with exponential backoff
3. 申请更高的 API 配额
4. 降低 max_iterations 到 3-4 次
```

### Q3: decord 安装失败

```bash
# 用 eva-decord 替代（社区维护版，兼容 Python 3.12）
pip install eva-decord

# 或者 fallback 到 imageio（代码已支持自动切换）
pip install imageio[ffmpeg]
```

### Q4: Mock 模式与真实模式的差异

Mock 模式仅验证代码流程逻辑，不生成真实视频，也不调用 Gemini API。真实实验必须配置 GPU 和 API Key。

---

## 八、实验时间估算

| 配置 | 单视频耗时 | VLM 调用次数/轮 | 总 VLM 调用 (10轮) |
|------|-----------|----------------|------------|
| pflow_original | ~30 min | 1 | ~10 |
| vista_full | ~60 min | ~12 | ~60 |
| vista_no_mmac | ~35 min | 2 | ~10 |
| vista_no_dtpa | ~50 min | 10 | ~50 |
| 全部 8 配置 | ~5 h | - | ~250 |

> 以上基于 A100 80GB + Gemini 1.5 Pro 估算。视频生成（Wan 2.1 14B, 81 帧, 50步）约 2-3 分钟/次。

---

## 九、结果分析

实验完成后，查看 `ablation_summary.json` 对比各配置：

```bash
python -c "
import json
with open('outputs/ablation_particles/ablation_summary.json') as f:
    data = json.load(f)
print(f\"{'Config':<25} {'Confidence':<12} {'Iters':<8} {'Time(s)':<10}\")
print('-'*55)
for name, info in data.items():
    print(f\"{name:<25} {info['best_confidence']:<12.3f} {info['num_iterations']:<8} {info['elapsed_time']:<10.1f}\")
"
```

期望结果趋势（基于论文分析）：

- `vista_full` ≥ `pflow_original`（VISTA 多智能体应优于单 VLM）
- `vista_no_mmac` 下降明显（MMAC 是核心评判机制）
- `vista_no_dtpa` 下降明显（DTPA 是核心推理机制）
- `vista_no_svpp` 轻微下降（SVPP 对短视频效果有限）
- `vista_no_tournament` 轻微下降（单候选时 Tournament 无用）
