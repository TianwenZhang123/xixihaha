# AutoDL 部署指南 —— A800 + Wan 2.1-14B + P-Flow 论文完整复现

## 一、AutoDL 选机配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU | **A800-80GB** | 14B 模型 + CPU offload，峰值约 40-50GB |
| 镜像 | **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4** | 需要 PyTorch 2.4+ 适配 diffusers |
| 系统盘 | 默认 30GB | 放代码、pip 环境 |
| 数据盘 | **100GB（推荐）** | 14B 模型约 55GB + 数据集约 15GB + 输出 |

### 费用估算

- A800 有卡模式约 **3-5 元/小时**
- 无卡模式（仅下载）约 **0.1 元/小时**
- 总流程：无卡下载约 60-90 分钟（模型 55GB）+ 有卡实验约 20 分钟/sample ≈ **按需计费**

---

## 二、方案说明

### 视频生成：Wan 2.1-T2V-14B 本地推理（单卡 A800）

> **重要**：必须使用 **Diffusers 格式** 的模型（`Wan-AI/Wan2.1-T2V-14B-Diffusers`），不能使用原始 checkpoint 格式。原始格式没有 `model_index.json`，无法被 Diffusers 库的 `from_pretrained()` 加载。

完整 P-Flow 论文流程（Algorithm 1）：
- Noise Prior Enhancement（Flow Inversion + SVD Filter + Blend）
- Test-Time Prompt Optimization（固定 10 轮迭代，论文完整复现）
- 单卡 A800 + `enable_model_cpu_offload()`
- 论文参数：α=0.001, ρ_s=0.1, ρ_m=0.9, i_max=10

### VLM：Qwen-VL-Max (DashScope 百炼)

| 配置项 | 值 |
|--------|-----|
| API Key | 通过 `DASHSCOPE_API_KEY` 环境变量设置 |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 模型 | `qwen-vl-max` |

### 数据集：Open-VFX

| 配置项 | 值 |
|--------|-----|
| 效果类别 | 15 种视觉特效 |
| 样本数量 | 1003 个 |
| 分辨率 | 480×832, 81 帧, 16fps (5s) |

---

## 三、完整操作流程

### 阶段一：无卡模式（下载 + 安装）

#### Step 1: 开启网络加速

```bash
source /etc/network_turbo
```

#### Step 2: 克隆代码

```bash
cd /root/autodl-tmp
git clone https://github.com/TianwenZhang123/xixihaha.git videofake
cd videofake/P-Flow
```

> 如果 GitHub 访问慢，用镜像：
> ```bash
> git clone https://ghproxy.com/https://github.com/TianwenZhang123/xixihaha.git videofake
> ```

#### Step 3: 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install flash-attn --no-build-isolation
```

#### Step 4: 配置环境变量

```bash
# HuggingFace 缓存到数据盘
export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc

# DashScope API Key (VLM: Qwen-VL-Max)
export DASHSCOPE_API_KEY="你的百炼API Key"
echo 'export DASHSCOPE_API_KEY="你的百炼API Key"' >> ~/.bashrc

source ~/.bashrc
```

#### Step 5: 创建目录 + 下载模型

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/outputs

# 设置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Wan 2.1-T2V-14B-Diffusers 模型（约 55GB）
# 注意：必须是 Diffusers 格式！
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-14B-Diffusers
```

> **注意事项**：
> - 14B 模型较大（~55GB），下载需要 60-90 分钟。
> - 如果 `huggingface-cli` 提示 deprecated，使用 `hf download` 代替。
> - 如果下载中断（SSL timeout），直接重新执行同一命令，支持断点续传。
> - 下载完成后验证：`ls /root/autodl-tmp/models/Wan2.1-T2V-14B-Diffusers/model_index.json` 必须存在。

#### Step 6: 下载 Open-VFX 数据集

```bash
mkdir -p /root/autodl-tmp/datasets/Open-VFX
cd /root/autodl-tmp/videofake/P-Flow

# 使用数据集准备脚本
python scripts/prepare_dataset.py \
    --output_dir /root/autodl-tmp/datasets/Open-VFX
```

> 如果 Open-VFX 数据集暂未公开发布，也可以用 MovieGenBench 数据集代替：
> ```bash
> mkdir -p /root/autodl-tmp/data
> cd /root/autodl-tmp/data
> wget -c https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz
> tar -xzf MovieGenVideoBench.tar.gz
> mv MovieGenVideoBench moviegen_bench
> cd moviegen_bench
> wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
> ls *.mp4 | wc -l  # 应该是 1003
> rm -f /root/autodl-tmp/data/MovieGenVideoBench.tar.gz
> ```

#### Step 7: 验证环境

```bash
cd /root/autodl-tmp/videofake/P-Flow
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from diffusers import WanPipeline; print('diffusers OK')"
python -c "from src.pipeline import PFlowPipeline; print('P-Flow pipeline OK')"
```

---

### 阶段二：有卡模式 — Wan 2.1-14B 本地推理实验

切到有卡模式开机后：

#### Step 1: 确认环境

```bash
source /etc/network_turbo
export DASHSCOPE_API_KEY="你的百炼API Key"

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应该看到：`True NVIDIA A800-SXM4-80GB`

#### Step 2: 快速验证（仅 Noise Prior，不调 VLM）

```bash
cd /root/autodl-tmp/videofake/P-Flow

python run.py \
    --video /root/autodl-tmp/data/moviegen_bench/23.mp4 \
    --prompt "Borneo wildlife on the Kinabatangan River" \
    --output /root/autodl-tmp/outputs/test_noise_prior \
    --mock_vlm \
    --seed 42
```

这个用 mock VLM 跑完整流程（不调真实 API），验证模型加载 + 推理正常。14B 模型加载约 1-2 分钟，单次生成约 90-120 秒。

#### Step 3: 完整 P-Flow（10 轮迭代）

```bash
python scripts/run_experiment.py \
    --video /root/autodl-tmp/data/moviegen_bench/23.mp4 \
    --prompt "Borneo wildlife on the Kinabatangan River" \
    --output_dir /root/autodl-tmp/outputs/test_023 \
    --seed 42
```

这会执行完整的 Algorithm 1：
1. 加载参考视频
2. 计算 Noise Prior Enhancement（Flow Inversion + SVD Filter）
3. 固定 10 轮 Test-Time Prompt Optimization
4. 每轮：生成视频 → VLM 分析 → 优化 prompt
5. 输出到 `/root/autodl-tmp/outputs/test_023/`

预计耗时：~17-22 分钟/sample

#### Step 4: 后台运行（防止 SSH 断连）

```bash
nohup python scripts/run_experiment.py \
    --video /root/autodl-tmp/data/moviegen_bench/23.mp4 \
    --prompt "Borneo wildlife on the Kinabatangan River" \
    --output_dir /root/autodl-tmp/outputs/test_023 \
    --seed 42 \
    > /root/autodl-tmp/outputs/run_log.txt 2>&1 &

tail -f /root/autodl-tmp/outputs/run_log.txt
```

---

## 四、命令速查

```bash
# 单个视频实验（完整 10 轮）
python scripts/run_experiment.py \
    --video /path/to/reference.mp4 \
    --prompt "effect description" \
    --output_dir /root/autodl-tmp/outputs/my_test \
    --seed 42

# 快速入口
python run.py \
    --video /path/to/reference.mp4 \
    --prompt "effect description" \
    --output /root/autodl-tmp/outputs/my_test \
    --seed 42

# Mock VLM 测试（不调 API，快速验证）
python run.py \
    --video /path/to/video.mp4 \
    --prompt "description" \
    --output /root/autodl-tmp/outputs/mock_test \
    --mock_vlm --seed 42

# 批量实验（顺序执行，一个一个跑）
python scripts/run_experiment.py \
    --dataset /root/autodl-tmp/datasets/Open-VFX \
    --split test \
    --output_dir /root/autodl-tmp/outputs/batch_test \
    --seed 42 \
    --start_index 0 --end_index 10

# 运行评测
python evaluation/run_evaluation.py \
    --experiment_dir /root/autodl-tmp/outputs/test_023 \
    --reference_video /root/autodl-tmp/data/moviegen_bench/23.mp4

# 选择最佳迭代
python evaluation/run_evaluation.py \
    --select_best \
    --experiment_dir /root/autodl-tmp/outputs/test_023 \
    --metric dynamic
```

---

## 五、性能预期

### Wan 2.1-14B 本地（单卡 A800-80GB）

| 指标 | 预期值 |
|------|--------|
| 模型加载 | ~1-2 分钟 |
| 单次生成 (480×832, 81帧, 50步) | ~90-120 秒 |
| VLM 调用 (Qwen-VL-Max) | ~5-10 秒/次 |
| 完整 10 轮优化 | ~17-22 分钟/sample |
| GPU 显存峰值 | ~40-50 GB |
| CPU RAM 需求 | ~32 GB+ |

---

## 六、输出目录结构

```
/root/autodl-tmp/outputs/test_023/
├── reference.mp4                    # 参考视频副本
├── generated_iter_001.mp4           # 第1轮生成
├── generated_iter_002.mp4           # 第2轮生成
├── ...
├── generated_iter_010.mp4           # 第10轮生成
├── full_trajectory.json             # 完整轨迹
├── experiment_metadata.json         # 实验元数据（含 timing）
├── composites/                      # VLM 输入的垂直 composite
│   ├── composite_iter_001.mp4
│   ├── ...
│   └── composite_iter_010.mp4
└── optimization_log/                # 每轮优化日志
    ├── iter_001.json
    ├── ...
    └── iter_010.json
```

---

## 七、磁盘空间说明

| 目录 | 大小 | 说明 |
|------|------|------|
| `/root/autodl-tmp/models/Wan2.1-T2V-14B-Diffusers` | ~55GB | 14B Diffusers 格式模型 |
| `/root/autodl-tmp/data/moviegen_bench` | ~15GB | 1003 个视频 + prompt 文件 |
| `/root/autodl-tmp/outputs` | ~几百MB | 实验输出（每轮视频约 50-100MB） |
| 总计 | ~70-75GB | 建议 100GB 数据盘 |

---

## 八、常见问题

### Q: `model_index.json` not found 错误
说明下载了原始格式模型（`Wan2.1-T2V-14B`）而非 Diffusers 格式。解决：
```bash
rm -rf /root/autodl-tmp/models/Wan2.1-T2V-14B
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-14B-Diffusers
```

### Q: OOM (Out of Memory) 显存不足
A800-80GB 正常情况下不会 OOM。如果遇到：
```bash
# 确认 VAE slicing + tiling 已启用（代码默认开启）
# 如果仍然 OOM，降低分辨率：
# 在 configs/paper_default.yaml 中改为 height: 320, width: 576
```

### Q: `DASHSCOPE_API_KEY` 环境变量未设置
```bash
export DASHSCOPE_API_KEY="你的key"
echo 'export DASHSCOPE_API_KEY="你的key"' >> ~/.bashrc
source ~/.bashrc
```

### Q: 下载中断 / SSL timeout
HuggingFace 镜像偶尔不稳定，直接重新运行同一命令即可，支持断点续传。

### Q: flash-attn 安装失败
```bash
# 确认 CUDA toolkit 版本
nvcc --version
# 如果报错，尝试不使用 flash-attn（会慢一些但能跑）
# 代码会自动 fallback 到标准 attention
```

### Q: 模型加载慢
14B 模型首次加载需要 1-2 分钟（从磁盘读 55GB），后续已缓存在内存中会快一些。这是正常的。

### Q: 关机后数据还在吗
- **数据盘** `/root/autodl-tmp/`：关机保留
- **系统盘** `/root/`：关机保留，释放实例清空
- 模型和数据集在数据盘，不用重复下载

### Q: 想换其他视频测试
```bash
# 查看所有可用 prompt（MovieGenBench）
head -50 /root/autodl-tmp/data/moviegen_bench/MovieGenVideoBench.txt

# 换视频即可
python scripts/run_experiment.py \
    --video /root/autodl-tmp/data/moviegen_bench/0.mp4 \
    --prompt "对应的 prompt" \
    --output_dir /root/autodl-tmp/outputs/test_000 \
    --seed 42
```

### Q: 代码更新后运行还报旧错误
```bash
cd /root/autodl-tmp/videofake
source /etc/network_turbo
git pull origin main
```

---

## 九、与 4090 (1.3B) 版本的对比

| 对比项 | 4090 + 1.3B | A800 + 14B |
|--------|-------------|------------|
| 显卡 | RTX 4090 (24GB) | A800 (80GB) |
| 模型 | Wan2.1-T2V-1.3B | Wan2.1-T2V-14B |
| 显存占用 | ~8-12GB | ~40-50GB |
| 生成速度 | ~1-2 分钟/视频 | ~90-120 秒/视频 |
| 迭代次数 | 3 轮 (缩减) | 10 轮 (论文原始) |
| 生成质量 | 良好 | 最佳（论文级别） |
| 单样本耗时 | ~5-8 分钟 | ~17-22 分钟 |
| 磁盘需求 | 50GB | 100GB |
| VLM | Qwen-VL-Max | Qwen-VL-Max |
