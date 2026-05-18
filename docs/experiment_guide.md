# P-Flow 实验指南：Wan 2.1 本地 + Wan 2.7 API 双路径

## 实验设计概览

本项目支持两种视频生成后端，用于对比实验：

| 路径 | 模型 | 运行方式 | 优势 | 限制 |
|------|------|----------|------|------|
| 路径A | Wan 2.1-T2V-1.3B | 本地 GPU 推理 (4090) | 可注入 noise prior、完整控制 | 画质较低(1.3B小模型) |
| 路径B | Wan 2.7-T2V | DashScope API | 画质最高、无需本地 GPU | 无法注入 noise prior、有API费用 |

两条路径都使用 gemini-2.0-flash (LinkAPI中转) 做 VLM prompt optimization。

---

## 环境要求

### 硬件
- AutoDL RTX 4090 24GB（路径A需要有卡模式；路径B无卡模式即可跑）
- 镜像：PyTorch 2.3.0 / Python 3.12 / CUDA 12.1

### API Keys
| 服务 | Key | 用途 |
|------|-----|------|
| LinkAPI 中转 | `<YOUR_OPENAI_API_KEY>` | VLM (gemini-2.0-flash) |
| DashScope 百炼 | `<YOUR_DASHSCOPE_API_KEY>` | Wan 2.7 视频生成 API |

---

## AutoDL 完整操作步骤

### 阶段一：无卡模式（下载+安装）

```bash
# 1. 开启网络加速
source /etc/network_turbo

# 2. 克隆代码
cd /root
git clone https://ghproxy.com/https://github.com/TianwenZhang123/xixihaha.git videofake
cd videofake

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 配置数据盘
export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/reference_videos
mkdir -p /root/autodl-tmp/outputs

# 5. 下载 Wan 2.1-1.3B 模型（路径A需要，约5GB）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B

# 6. 生成测试视频
ffmpeg -f lavfi -i "testsrc2=duration=5:size=832x480:rate=16" \
    -frames:v 81 -c:v libx264 -y \
    /root/autodl-tmp/data/reference_videos/test_effect.mp4

# 7. 配置 API Keys
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
echo 'export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"' >> ~/.bashrc
echo 'export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"' >> ~/.bashrc

# 8. 验证安装（Mock模式，不需GPU不需API）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_test --mock --seed 42

python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_api --mock --seed 42
```

验证通过后关机。

---

### 阶段二：路径B — Wan 2.7 API（无卡模式即可！）

路径B不需要本地GPU，可以在无卡模式下运行（省钱）。

```bash
# 有卡或无卡模式均可，配置环境变量
source /etc/network_turbo
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
cd /root/videofake

# === 实验 B1：单次生成（最便宜，验证API能通） ===
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails, cinematic" \
    --output_dir /root/autodl-tmp/outputs/wan27_single \
    --single_shot \
    --seed 42

# === 实验 B2：完整 P-Flow 优化循环（5轮） ===
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan27_optimized \
    --max_iterations 5 \
    --seed 42

# === 实验 B3：1080P 高分辨率 ===
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan27_1080p \
    --video_size "1920*1080" \
    --video_duration 5 \
    --single_shot \
    --seed 42

# === 实验 B4：多镜头视频 ===
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "一位少女走在星空下，粒子从脚底升起。镜头切换到她伸手触碰光粒，光芒绽放。" \
    --output_dir /root/autodl-tmp/outputs/wan27_multishot \
    --video_duration 10 \
    --multi_shot \
    --single_shot \
    --seed 42
```

---

### 阶段三：路径A — Wan 2.1 本地推理（需有卡模式）

有卡模式开机后：

```bash
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/huggingface
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
cd /root/videofake

# 确认 GPU
nvidia-smi

# === 实验 A1：仅 Noise Prior（不调VLM，快速验证模型能跑） ===
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward" \
    --output_dir /root/autodl-tmp/outputs/wan21_noise_prior \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --noise_prior_only \
    --seed 42

# === 实验 A2：完整 P-Flow（3轮，约5-10分钟） ===
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan21_3iter \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --max_iterations 3 \
    --seed 42

# === 实验 A3：完整 P-Flow（10轮，约20分钟） ===
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan21_full \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --max_iterations 10 \
    --seed 42
```

---

## 实验对比方案

### 推荐实验顺序

1. **先跑路径B（API）**：无卡模式，验证整个 prompt optimization 流程能跑通
2. **再跑路径A（本地）**：有卡模式，验证 noise prior + 本地推理完整链路
3. **对比结果**：比较两条路径在相同 prompt 下的视频质量

### 对比维度

| 维度 | 路径A (Wan 2.1 本地) | 路径B (Wan 2.7 API) |
|------|---------------------|---------------------|
| 视频质量 | 基础（1.3B模型） | 高质量（完整大模型） |
| Noise Prior | ✓ 有 | ✗ 无（API不支持自定义latent） |
| 推理时间 | ~1-2分钟/视频 | ~1-5分钟/视频（看排队） |
| GPU需求 | 4090 有卡模式 | 不需要（无卡即可） |
| 费用 | GPU时间费 ~1-2元/小时 | 0.6元/秒(720P) API费 |
| 可控性 | 完全可控 | 有限（只能控制prompt） |

### 核心科研问题

- **Q1**：Noise Prior 在低画质模型上的增益有多大？（A1 vs A2）
- **Q2**：Prompt Optimization 对高画质模型还有必要吗？（B1 vs B2）
- **Q3**：低画质+Noise Prior vs 高画质+纯Prompt，哪个效果更好？（A3 vs B2）

---

## 费用估算

### 路径A（4090有卡）
- GPU费：~1.5元/小时
- 完整实验（3组）：~30分钟 ≈ 1元
- VLM API：~10次调用 ≈ 0.1元
- **合计：~1-2元**

### 路径B（Wan 2.7 API）
- 单次720P 5秒：0.6×5 = 3元
- 5轮优化：15元
- VLM API：~5次调用 ≈ 0.05元
- **合计：~3-15元**（新用户有免费额度可抵扣）

---

## 文件路径总览

```
/root/videofake/                        # 代码（系统盘）
├── scripts/
│   ├── run_pflow.py                    # 路径A：本地推理
│   ├── run_pflow_api.py                # 路径B：API调用
│   ├── run_ablation.py                 # VISTA消融实验
│   └── evaluate.py                     # 评估脚本

/root/autodl-tmp/                       # 数据盘
├── models/Wan2.1-T2V-1.3B/            # 本地模型权重
├── data/reference_videos/              # 参考视频
└── outputs/
    ├── wan21_noise_prior/              # A1: 仅noise prior
    ├── wan21_3iter/                    # A2: 本地3轮
    ├── wan21_full/                     # A3: 本地10轮
    ├── wan27_single/                   # B1: API单次
    ├── wan27_optimized/                # B2: API优化
    ├── wan27_1080p/                    # B3: API高清
    └── wan27_multishot/                # B4: API多镜头
```

---

## 常见问题

### Q: DashScope API 返回 403
免费额度用完了。去百炼控制台充值或开通按量计费。

### Q: API 调用超时
视频生成需要1-5分钟，默认超时600秒。如果持续超时，可能是服务端排队拥堵。

### Q: 路径B能否注入 Noise Prior？
不能。DashScope API 不暴露 latent 接口。这也是为什么我们保留路径A的原因——论文的核心贡献之一就是 noise prior，需要本地推理才能验证。

### Q: 两条路径能否用相同的 prompt？
可以且推荐如此。用同一个 prompt 对比两个模型的输出，才能看出 noise prior 的增益。
