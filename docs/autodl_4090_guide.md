# AutoDL 部署指南 —— 4090 + Wan 2.1 本地 + P-Flow 论文复现

## 一、AutoDL 选机配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU | **RTX 4090 24GB** | 1.3B 模型显存绰绰有余 |
| 镜像 | **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4** | 需要 PyTorch 2.4+ 适配 diffusers 0.38 |
| 系统盘 | 默认 30GB | 放代码、conda 环境 |
| 数据盘 | **50GB 够用** | 模型约 5GB + 数据集 + 输出视频 |

### 费用估算

- 4090 有卡模式约 **1-2 元/小时**
- 无卡模式（仅下载/API调用）约 **0.1 元/小时**
- 总流程：无卡下载 10 分钟 + 有卡实验 30 分钟 ≈ **3-5 元搞定**

---

## 二、方案说明

### 视频生成：Wan 2.1-T2V-1.3B 本地推理

完整 P-Flow 论文流程（Algorithm 1）：
- Noise Prior Enhancement（Flow Inversion + SVD Filter + Blend）
- Test-Time Prompt Optimization（固定 3 轮迭代）
- 需要有卡模式（GPU）
- 可注入自定义 noise prior latents

### VLM：Qwen-VL-Max (DashScope 百炼)

| 配置项 | 值 |
|--------|-----|
| API Key | 通过 `DASHSCOPE_API_KEY` 环境变量设置 |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 模型 | `qwen-vl-max` (Qwen3-VL-Flash) |

### 数据集：MovieGenBench

| 配置项 | 值 |
|--------|-----|
| 视频来源 | CloudFront CDN (`MovieGenVideoBench.tar.gz`，约 15GB) |
| Prompt 文件 | GitHub `facebookresearch/MovieGenBench` 仓库 |
| 视频数量 | 1003 个 |
| 文件命名 | `{index}.mp4`（0-indexed，对应 txt 文件行号） |

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
cd videofake
```

> 如果 GitHub 访问慢，用镜像：
> ```bash
> git clone https://ghproxy.com/https://github.com/TianwenZhang123/xixihaha.git videofake
> ```

#### Step 3: 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
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

# 下载 Wan 2.1-T2V-1.3B 模型（约 5GB）
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B
```

> 注意：新版 huggingface_hub 使用 `hf` 命令而非 `huggingface-cli`。
> 如果 `hf` 不可用，尝试 `pip install -U huggingface_hub` 后重试。

#### Step 6: 下载 MovieGenBench 数据集

```bash
# 6a. 下载视频压缩包（约 15GB，CloudFront CDN）
mkdir -p /root/autodl-tmp/data
cd /root/autodl-tmp/data
wget -c https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz

# 6b. 解压（解压后约 16GB）
tar -xzf MovieGenVideoBench.tar.gz
mv MovieGenVideoBench moviegen_bench

# 6c. 下载 Prompt 文件
cd /root/autodl-tmp/data/moviegen_bench
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBenchWithTag.csv

# 6d. 验证
ls water_mark_out/ | wc -l  # 应该是 1003
head -3 MovieGenVideoBench.txt
```

> 如果 GitHub raw 文件下载慢，可加代理：
> ```bash
> wget https://ghproxy.com/https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
> wget https://ghproxy.com/https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBenchWithTag.csv
> ```

#### Step 7: 验证环境

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from diffusers import WanPipeline; print('diffusers OK')"
python -c "from pflow import PFlowPipeline; print('pflow OK')"
```

#### Step 8: Mock 验证（不用 GPU 不用 API）

```bash
cd /root/autodl-tmp/videofake

python scripts/run_pflow_paper.py --video_index 0 --mock
```

如果输出完整运行过程（3 轮迭代 + 保存视频），说明代码逻辑正确。

---

### 阶段二：有卡模式 — Wan 2.1 本地推理实验

切到有卡模式开机后：

#### Step 1: 确认环境

```bash
source /etc/network_turbo
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Step 2: 仅 Noise Prior（快速验证模型加载 + 推理）

```bash
cd /root/autodl-tmp/videofake

python scripts/run_pflow_paper.py \
    --video_index 23 \
    --noise_prior_only \
    --seed 42
```

这个只做一次视频生成（不调 VLM），约 1-2 分钟，验证模型能正常推理。

#### Step 3: 完整 P-Flow（3 轮迭代）

```bash
export DASHSCOPE_API_KEY="你的百炼API Key"

python scripts/run_pflow_paper.py \
    --video_index 23 \
    --seed 42
```

这会执行完整的 Algorithm 1：
1. 加载参考视频 `23.mp4`（猫叫主人起床）
2. 计算 Noise Prior Enhancement
3. 固定 3 轮 Test-Time Prompt Optimization
4. 每轮生成视频 + VLM 分析 + 优化 prompt
5. 输出到 `/root/autodl-tmp/outputs/test_023/`

---

## 四、一键脚本

### setup_all.sh（无卡模式一键安装）

```bash
#!/bin/bash
set -e
echo "===== P-Flow 环境安装 ====="

source /etc/network_turbo 2>/dev/null || true

# 克隆代码
cd /root/autodl-tmp
if [ ! -d "videofake" ]; then
    git clone https://github.com/TianwenZhang123/xixihaha.git videofake
fi
cd videofake

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 环境变量
export HF_HOME=/root/autodl-tmp/huggingface
echo 'export HF_HOME=/root/autodl-tmp/huggingface' >> ~/.bashrc

# 数据目录
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/outputs

# 下载模型
export HF_ENDPOINT=https://hf-mirror.com
hf download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B

# 下载数据集（视频约 15GB）
mkdir -p /root/autodl-tmp/data
cd /root/autodl-tmp/data
wget -c https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz
tar -xzf MovieGenVideoBench.tar.gz
mv MovieGenVideoBench moviegen_bench
cd moviegen_bench
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBenchWithTag.csv

# Mock 验证
python scripts/run_pflow_paper.py --video_index 0 --mock

echo ""
echo "===== 安装完成！====="
echo "下一步："
echo "  1. 设置 API Key: export DASHSCOPE_API_KEY=\"你的key\""
echo "  2. 切有卡模式"
echo "  3. 运行: python scripts/run_pflow_paper.py --video_index 23 --seed 42"
```

### run_experiment.sh（有卡模式跑实验）

```bash
#!/bin/bash
set -e
echo "===== P-Flow 完整实验 ====="

source /etc/network_turbo 2>/dev/null || true

cd /root/autodl-tmp/videofake

# 确认 GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 完整 P-Flow（3 轮优化，猫视频）
python scripts/run_pflow_paper.py \
    --video_index 23 \
    --seed 42

echo "===== 实验完成！====="
echo "结果在: /root/autodl-tmp/outputs/test_023/"
```

---

## 五、命令速查

```bash
# 按视频 index 运行（自动读取 prompt + 视频路径）
python scripts/run_pflow_paper.py --video_index 23 --seed 42

# 指定视频和 prompt 运行
python scripts/run_pflow_paper.py \
    --reference_video /root/autodl-tmp/data/moviegen_bench/water_mark_out/23.mp4 \
    --prompt "A cat waking up its sleeping owner demanding breakfast..." \
    --output_dir /root/autodl-tmp/outputs/test_023 \
    --seed 42

# 只跑 Noise Prior（不调 VLM，快速验证）
python scripts/run_pflow_paper.py --video_index 23 --noise_prior_only --seed 42

# Mock 模式（不需要 GPU 和 API）
python scripts/run_pflow_paper.py --video_index 0 --mock

# 修改迭代次数
python scripts/run_pflow_paper.py --video_index 23 --i_max 5 --seed 42

# 后台运行
nohup python scripts/run_pflow_paper.py --video_index 23 --seed 42 \
    > /root/autodl-tmp/outputs/run_log.txt 2>&1 &
tail -f /root/autodl-tmp/outputs/run_log.txt
```

---

## 六、性能预期

### Wan 2.1 本地（4090）

| 指标 | 预期值 |
|------|--------|
| 模型加载 | ~30 秒 |
| 单次生成 (480P, 81帧, 50步) | ~1-2 分钟 |
| VLM 调用 (Qwen-VL-Max) | ~5-10 秒/次 |
| 完整 3 轮优化 | ~5-8 分钟 |
| GPU 显存 | ~8-12 GB |

---

## 七、输出目录结构

```
/root/autodl-tmp/outputs/test_023/
├── reference.mp4                    # 参考视频副本
├── generated_iter_001.mp4           # 第1轮生成
├── generated_iter_002.mp4           # 第2轮生成
├── generated_iter_003.mp4           # 第3轮生成
├── full_trajectory.json             # 完整轨迹
├── prompts_history.json             # Prompt 演化历史
├── composites/                      # VLM 输入的垂直 composite
│   ├── composite_iter_001.mp4
│   ├── composite_iter_002.mp4
│   └── composite_iter_003.mp4
├── optimization_log/                # 每轮优化日志
│   ├── iter_001.json
│   ├── iter_002.json
│   └── iter_003.json
└── trajectory/
    └── trajectory.json
```

---

## 八、常见问题

### Q: torch.xpu 报错 / AttributeError
PyTorch 版本太低（<2.4），需要用 PyTorch 2.5.1+ 镜像。

### Q: NumPy 兼容性错误
```bash
pip install "numpy<2"
```

### Q: DashScope API 返回 403 错误
免费额度用完。去 [百炼控制台](https://bailian.console.aliyun.com/) 充值或开通按量计费。

### Q: HuggingFace 下载慢
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: `huggingface-cli` 提示 deprecated
新版 huggingface_hub 改用 `hf` 命令：
```bash
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B
```

### Q: 关机后数据还在吗
- **数据盘** `/root/autodl-tmp/`：关机保留
- **系统盘** `/root/`：关机保留，释放实例清空
- 模型和数据集在数据盘，不用重复下载

### Q: 想换其他视频测试
```bash
# 查看所有可用 prompt
cat /root/autodl-tmp/data/moviegen_bench/MovieGenVideoBench.txt | head -50

# 换 index 即可（0-1002）
python scripts/run_pflow_paper.py --video_index 0 --seed 42
```
