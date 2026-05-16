# AutoDL 部署指南 —— 4090 + Wan2.1-1.3B + Gemini 2.0 Flash 跑通 P-Flow

## 一、AutoDL 选机配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU | **RTX 4090 24GB** | 1.3B 模型显存绰绰有余 |
| 镜像 | **PyTorch 2.3.0 / Python 3.12 / CUDA 12.1** | AutoDL 官方镜像直接选 |
| 系统盘 | 默认 30GB | 放代码、conda 环境 |
| 数据盘 | **50GB 够用** | 模型约 5GB + 输出视频 |

### 费用估算

- 4090 有卡模式约 **1-2 元/小时**
- 无卡模式（仅下载）约 **0.1 元/小时**
- 总流程：无卡下载 10 分钟 + 有卡实验 30 分钟 ≈ **2-3 元搞定**

---

## 二、VLM 方案：LinkAPI 中转站 + Gemini 2.0 Flash

通过 LinkAPI 中转站调用 Gemini 2.0 Flash，**国内直连，无需代理**。

### 已配置信息

| 配置项 | 值 |
|--------|-----|
| API Key | `<YOUR_OPENAI_API_KEY>` |
| Base URL | `https://api.linkapi.org/v1` |
| 模型 | `gemini-2.0-flash` |

### 工作原理

中转站使用 OpenAI 兼容格式的 API。代码会自动从视频中抽取 8 帧关键帧，编码为 base64 图片发给 Gemini 2.0 Flash 分析。Gemini 2.0 Flash 视觉理解能力很强，通过关键帧足以分析运动趋势、光效变化等视觉特征。

---

## 三、完整操作流程

### 阶段一：无卡模式（下载模型，省钱）

在 AutoDL 控制台创建实例后，先用**无卡模式**开机（不消耗 GPU 费用），把模型下载好。

#### Step 1: 无卡模式开机，进入终端

```bash
# AutoDL 控制台 → 实例 → 无卡模式开机 → JupyterLab → Terminal
```

#### Step 2: 开启学术加速

```bash
source /etc/network_turbo
```

#### Step 3: 克隆代码

```bash
cd /root
git clone https://github.com/TianwenZhang123/xixihaha.git videofake
cd videofake
```

#### Step 4: 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### Step 5: 下载 Wan2.1-T2V-1.3B 到数据盘（约 5GB）

```bash
# 设置 HF 缓存到数据盘
export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc

# 创建目录
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/reference_videos
mkdir -p /root/autodl-tmp/outputs

# 下载模型（约 5GB，2-5 分钟）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B
```

#### Step 6: 准备测试视频

```bash
ffmpeg -f lavfi -i "testsrc2=duration=5:size=832x480:rate=16" \
    -frames:v 81 -c:v libx264 -y \
    /root/autodl-tmp/data/reference_videos/test_effect.mp4
```

#### Step 7: 验证代码（Mock 模式，无需 GPU 和 API）

```bash
cd /root/videofake
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with glowing trails" \
    --output_dir /root/autodl-tmp/outputs/mock_test \
    --mock \
    --seed 42
```

如果输出正常，说明代码和依赖都没问题。

#### Step 8: 关机

无卡模式下模型已下载完毕、代码已验证。现在**关机**，准备切换到有卡模式。

---

### 阶段二：有卡模式（跑实验）

#### Step 1: 有卡模式开机

AutoDL 控制台 → 实例 → 开机（正常模式，会分配 GPU）

#### Step 2: 配置 API Key

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
echo 'export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"' >> ~/.bashrc
```

#### Step 3: 确认 GPU 可用

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Step 4: 跑 P-Flow 完整流程

```bash
cd /root/videofake

python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/pflow_run1 \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --seed 42
```

#### 其他运行方式

```bash
# 减少迭代（3轮，约 5 分钟，快速验证）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/pflow_3iter \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --max_iterations 3 \
    --seed 42

# 只做 noise prior（快速验证模型推理，不调 VLM API）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward" \
    --output_dir /root/autodl-tmp/outputs/noise_prior_test \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --noise_prior_only \
    --seed 42

# 换模型（如果中转站支持其他模型）
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/pflow_gemini15 \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --vlm_model gemini-1.5-pro \
    --seed 42
```

---

## 四、文件路径总览

```
/root/                              # 系统盘（30GB）
├── videofake/                      # 项目代码（<1MB）
│   ├── pflow/
│   ├── scripts/
│   ├── config/
│   └── docs/

/root/autodl-tmp/                   # 数据盘（50GB）
├── huggingface/                    # HF 缓存
├── models/
│   └── Wan2.1-T2V-1.3B/           # 模型权重（~5GB）
├── data/
│   └── reference_videos/
│       └── test_effect.mp4
└── outputs/                        # 所有实验输出
    ├── mock_test/                  # Mock 验证
    ├── noise_prior_test/           # 噪声先验测试
    └── pflow_run1/                 # 完整实验结果
        ├── best_result.mp4
        ├── generated_iter_*.mp4
        ├── prompts_history.json
        └── final_results.json
```

---

## 五、一键脚本

### 无卡阶段脚本（setup_download.sh）

```bash
#!/bin/bash
set -e
echo "===== [无卡模式] 下载模型 + 安装依赖 ====="

source /etc/network_turbo 2>/dev/null || true

export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc

cd /root
if [ ! -d "videofake" ]; then
    git clone https://github.com/TianwenZhang123/xixihaha.git videofake
fi
cd videofake

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/data/reference_videos
mkdir -p /root/autodl-tmp/outputs

# 下载 1.3B 模型
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B

# 生成测试视频
ffmpeg -f lavfi -i "testsrc2=duration=5:size=832x480:rate=16" \
    -frames:v 81 -c:v libx264 -y \
    /root/autodl-tmp/data/reference_videos/test_effect.mp4 2>/dev/null

# 验证
python -c "from pflow import PFlowPipeline; print('All imports OK')"
python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "test" --output_dir /root/autodl-tmp/outputs/mock_test --mock --seed 42

echo ""
echo "===== 下载完成！关机后切换有卡模式运行实验 ====="
```

### 有卡阶段脚本（run_experiment.sh）

```bash
#!/bin/bash
set -e
echo "===== [有卡模式] 运行 P-Flow 实验 ====="

source /etc/network_turbo 2>/dev/null || true
export HF_HOME=/root/autodl-tmp/huggingface
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"

cd /root/videofake

python scripts/run_pflow.py \
    --reference_video /root/autodl-tmp/data/reference_videos/test_effect.mp4 \
    --prompt "Golden particles floating upward with dynamic light trails" \
    --output_dir /root/autodl-tmp/outputs/pflow_run1 \
    --model /root/autodl-tmp/models/Wan2.1-T2V-1.3B \
    --vlm_base_url "https://api.linkapi.org/v1" \
    --seed 42

echo "===== 实验完成！====="
```

---

## 六、4090 + 1.3B 性能预期

| 指标 | 预期值 |
|------|--------|
| 模型加载时间 | ~30 秒 |
| 单次视频生成（50步, 480x832, 81帧） | ~1-2 分钟 |
| 完整 10 轮迭代 | ~15-20 分钟 |
| GPU 显存占用 | ~8-12 GB（远低于 24GB 上限） |
| VLM API 调用 | 10 次 |
| API 费用 | 约 ¥0.1-0.3（中转站计费） |

---

## 七、跑通后的升级路径

| 阶段 | 改动 | 目的 |
|------|------|------|
| 当前 | 4090 + 1.3B + gemini-2.0-flash | 验证流程 |
| 升级一 | 4090 + 1.3B + 真实参考视频 | 验证效果 |
| 升级二 | A100 + 14B + gemini-2.0-flash | 出正式结果 |
| 升级三 | A100 + 14B + VISTA 多智能体 | 消融实验 |

每次升级只需改 `--model` 和机器配置，代码不用动。

---

## 八、常见问题

### Q: API 调用失败

```bash
# 测试中转站连通性
curl -s https://api.linkapi.org/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY" | python -m json.tool | head -20
```

中转站国内直连，一般不需要代理。如果 AutoDL 环境下连不上，开学术加速即可：`source /etc/network_turbo`

### Q: 关机后数据还在吗

- **数据盘** `/root/autodl-tmp/`：关机保留，释放实例也保留（跟账号绑定）
- **系统盘** `/root/`：关机保留，释放实例清空

模型在数据盘，不用重复下载。代码在系统盘但可以 git clone 恢复。

### Q: 想后台跑

```bash
nohup bash run_experiment.sh > /root/autodl-tmp/outputs/experiment.log 2>&1 &
tail -f /root/autodl-tmp/outputs/experiment.log
```

### Q: 想换其他 VLM 模型

中转站通常支持多种模型，只需改 `--vlm_model` 参数：

```bash
# gemini-1.5-pro（更强的视频理解）
--vlm_model gemini-1.5-pro

# gpt-4o（OpenAI 视觉模型）
--vlm_model gpt-4o

# claude-3.5-sonnet（Anthropic）
--vlm_model claude-3.5-sonnet
```

具体支持哪些模型取决于 LinkAPI 中转站的配置。
