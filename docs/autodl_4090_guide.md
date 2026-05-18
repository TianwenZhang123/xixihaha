# AutoDL 部署指南 —— 4090 + Wan 2.1 本地 + Wan 2.7 API

## 一、AutoDL 选机配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU | **RTX 4090 24GB** | 1.3B 模型显存绰绰有余 |
| 镜像 | **PyTorch 2.3.0 / Python 3.12 / CUDA 12.1** | AutoDL 官方镜像直接选 |
| 系统盘 | 默认 30GB | 放代码、conda 环境 |
| 数据盘 | **50GB 够用** | 模型约 5GB + 输出视频 |

### 费用估算

- 4090 有卡模式约 **1-2 元/小时**
- 无卡模式（仅下载/API调用）约 **0.1 元/小时**
- 总流程：无卡下载 10 分钟 + API 实验 30 分钟 + 有卡实验 30 分钟 ≈ **3-5 元搞定**

---

## 二、双后端方案

### 后端 A：Wan 2.1-T2V-1.3B 本地推理

完整 P-Flow 流程（含 Noise Prior Enhancement + Test-Time Prompt Optimization）。

- 需要有卡模式（GPU）
- 可注入自定义 noise prior latents
- 验证论文核心算法

### 后端 B：Wan 2.7-T2V API (DashScope 百炼)

通过阿里云 API 调用最新 Wan 2.7 模型生成视频。

- 无卡模式即可运行（不需要本地 GPU 做视频生成）
- 画质远超本地 1.3B 模型
- 异步 API：提交任务 → 等待 1-5 分钟 → 下载视频
- 仍然用 VLM (gemini-2.0-flash) 做 prompt optimization

### VLM：Gemini 2.0 Flash (LinkAPI 中转)

| 配置项 | 值 |
|--------|-----|
| API Key | `<YOUR_OPENAI_API_KEY>` |
| Base URL | `https://api.linkapi.org/v1` |
| 模型 | `gemini-2.0-flash` |

### DashScope 百炼（Wan 2.7 视频生成）

| 配置项 | 值 |
|--------|-----|
| API Key | `<YOUR_DASHSCOPE_API_KEY>` |
| 模型 | `wan2.7-t2v` |
| 价格 | 720P: 0.6元/秒, 1080P: 1元/秒 |
| 免费额度 | 有（用完即停） |

---

## 三、完整操作流程

### 阶段一：无卡模式（下载 + 安装 + API 验证）

#### Step 1: 开启网络

```bash
source /etc/network_turbo
```

#### Step 2: 克隆代码

```bash
cd /root
git clone https://ghproxy.com/https://github.com/TianwenZhang123/xixihaha.git videofake
cd videofake
```

#### Step 3: 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Step 4: 配置环境变量

```bash
# HuggingFace 缓存到数据盘
export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc

# VLM API Key (LinkAPI 中转 gemini-2.0-flash)
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
echo 'export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"' >> ~/.bashrc

# DashScope API Key (Wan 2.7)
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
echo 'export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"' >> ~/.bashrc
```

#### Step 5: 创建数据目录 + 下载模型

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/reference_videos
mkdir -p /root/autodl-tmp/outputs

# 下载 Wan 2.1 本地模型（约 5GB）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B
```

#### Step 6: 生成测试视频

```bash
ffmpeg -f lavfi -i "testsrc2=duration=5:size=832x480:rate=16" \
    -frames:v 81 -c:v libx264 -y \
    /root/autodl-tmp/data/reference_videos/test_effect.mp4
```

#### Step 7: Mock 验证（不用 GPU 不用 API）

```bash
cd /root/videofake

# 验证本地路径代码
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_local --mock --seed 42

# 验证 API 路径代码
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_api --mock --seed 42
```

#### Step 8: 【可选】验证 Wan 2.7 API（无卡模式就能跑！）

```bash
# 单次生成，验证 API 连通性（费用：5秒×0.6元=3元，有免费额度）
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails, cinematic quality" \
    --output_dir /root/autodl-tmp/outputs/wan27_test \
    --single_shot \
    --seed 42
```

如果成功生成了视频，说明 API 链路没问题。

---

### 阶段二：路径 B — Wan 2.7 API 实验（无卡模式）

不需要切有卡，直接在无卡模式下运行所有 API 实验：

```bash
source /etc/network_turbo
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
cd /root/videofake

# 完整 prompt optimization 循环（5轮）
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan27_optimized \
    --max_iterations 5 \
    --seed 42
```

---

### 阶段三：路径 A — Wan 2.1 本地推理实验（有卡模式）

切到有卡模式开机后：

```bash
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/huggingface
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
cd /root/videofake

# 确认 GPU
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 仅 Noise Prior（快速验证模型推理，不调 VLM）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward" \
    --output_dir /root/autodl-tmp/outputs/wan21_noise_prior \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --noise_prior_only \
    --seed 42

# 完整 P-Flow（3轮，约5-10分钟）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan21_3iter \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --max_iterations 3 \
    --seed 42
```

---

## 四、一键脚本

### setup_all.sh（无卡模式一键安装）

```bash
#!/bin/bash
set -e
echo "===== P-Flow 环境安装 ====="

source /etc/network_turbo 2>/dev/null || true

# 环境变量
export HF_HOME=/root/autodl-tmp/huggingface
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc
echo 'export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"' >> ~/.bashrc
echo 'export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"' >> ~/.bashrc

# 克隆代码
cd /root
if [ ! -d "videofake" ]; then
    git clone https://ghproxy.com/https://github.com/TianwenZhang123/xixihaha.git videofake
fi
cd videofake

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 数据目录
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/reference_videos
mkdir -p /root/autodl-tmp/outputs

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B

# 生成测试视频
ffmpeg -f lavfi -i "testsrc2=duration=5:size=832x480:rate=16" \
    -frames:v 81 -c:v libx264 -y \
    /root/autodl-tmp/data/reference_videos/test_effect.mp4 2>/dev/null

# 验证
python -c "from pflow import PFlowPipeline, WanAPIClient; print('All imports OK')"
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_local --mock --seed 42
python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_api --mock --seed 42

echo ""
echo "===== 安装完成！====="
echo "下一步："
echo "  路径B (API): python scripts/run_pflow_api.py --reference_video ... --prompt ... --single_shot"
echo "  路径A (本地): 关机→有卡模式→python scripts/run_pflow.py --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B ..."
```

### run_api_experiment.sh（无卡模式跑API实验）

```bash
#!/bin/bash
set -e
echo "===== 路径B: Wan 2.7 API 实验 ====="

source /etc/network_turbo 2>/dev/null || true
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
export DASHSCOPE_API_KEY="<YOUR_DASHSCOPE_API_KEY>"

cd /root/videofake

python scripts/run_pflow_api.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails, cinematic lighting" \
    --output_dir /root/autodl-tmp/outputs/wan27_optimized \
    --max_iterations 5 \
    --seed 42

echo "===== API 实验完成！====="
```

### run_local_experiment.sh（有卡模式跑本地实验）

```bash
#!/bin/bash
set -e
echo "===== 路径A: Wan 2.1 本地推理实验 ====="

source /etc/network_turbo 2>/dev/null || true
export HF_HOME=/root/autodl-tmp/huggingface
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

cd /root/videofake

python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/wan21_full \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --max_iterations 5 \
    --seed 42

echo "===== 本地实验完成！====="
```

---

## 五、性能预期

### 路径 A：Wan 2.1 本地（4090）

| 指标 | 预期值 |
|------|--------|
| 模型加载 | ~30 秒 |
| 单次生成 (480P, 81帧, 50步) | ~1-2 分钟 |
| 完整 5 轮优化 | ~10-15 分钟 |
| GPU 显存 | ~8-12 GB |

### 路径 B：Wan 2.7 API

| 指标 | 预期值 |
|------|--------|
| 单次生成 (720P, 5秒) | ~1-3 分钟 |
| 单次生成 (1080P, 5秒) | ~3-5 分钟 |
| 完整 5 轮优化 | ~10-20 分钟 |
| 本地 GPU | 不需要 |
| API 费用 | ~3-15 元/轮（有免费额度） |

---

## 六、升级路径

| 阶段 | 视频后端 | VLM | 目的 |
|------|----------|-----|------|
| 当前 | 4090 + 1.3B 本地 | gemini-2.0-flash | 验证 noise prior |
| 当前 | Wan 2.7 API | gemini-2.0-flash | 验证 prompt optimization |
| 升级一 | 两条路径 + 真实参考视频 | gemini-2.0-flash | 验证实际效果 |
| 升级二 | A100 + 14B 本地 | gemini-2.0-flash | 出正式结果 |
| 升级三 | A100 + 14B + VISTA 多智能体 | gemini-2.0-flash | 消融实验 |

---

## 七、常见问题

### Q: DashScope API 返回 403 错误
免费额度用完了。去 [百炼控制台](https://bailian.console.aliyun.com/) 充值或开通按量计费。

### Q: LinkAPI VLM 调用失败
```bash
curl -s https://api.linkapi.org/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY" | python -m json.tool | head -20
```

### Q: 关机后数据还在吗
- **数据盘** `/root/autodl-tmp/`：关机保留
- **系统盘** `/root/`：关机保留，释放实例清空
- 模型在数据盘，不用重复下载

### Q: API 路径能做 Noise Prior 吗
不能。DashScope API 不暴露 latent 注入接口。这是保留本地路径的核心原因。

### Q: 想后台跑实验
```bash
nohup bash run_api_experiment.sh > /root/autodl-tmp/outputs/api_log.txt 2>&1 &
tail -f /root/autodl-tmp/outputs/api_log.txt
```
