# AutoDL 部署指南 —— 4090 + Wan 2.1 本地 + P-Flow 论文复现

## 一、AutoDL 选机配置

| 配置项 | 选择 | 说明 |
|--------|------|------|
| GPU | **RTX 4090 24GB** | 1.3B 模型显存绰绰有余 |
| 镜像 | **PyTorch 2.5.1 / Python 3.12 / CUDA 12.4** | 需要 PyTorch 2.4+ 适配 diffusers 0.38 |
| 系统盘 | 默认 30GB | 放代码、conda 环境 |
| 数据盘 | **50GB（最低要求）** | 模型约 27GB + 数据集约 15GB + 输出 |

### 费用估算

- 4090 有卡模式约 **1-2 元/小时**
- 无卡模式（仅下载/API调用）约 **0.1 元/小时**
- 总流程：无卡下载约 30-60 分钟（模型 27GB）+ 有卡实验约 10 分钟 ≈ **5-8 元搞定**

---

## 二、方案说明

### 视频生成：Wan 2.1-T2V-1.3B-Diffusers 本地推理

> **重要**：必须使用 **Diffusers 格式** 的模型（`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`），不能使用原始 checkpoint 格式（`Wan-AI/Wan2.1-T2V-1.3B`）。原始格式没有 `model_index.json`，无法被 Diffusers 库的 `from_pretrained()` 加载。

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
| 模型 | `qwen-vl-max` |

### 数据集：MovieGenBench

| 配置项 | 值 |
|--------|-----|
| 视频来源 | CloudFront CDN (`MovieGenVideoBench.tar.gz`，约 15GB) |
| Prompt 文件 | GitHub `facebookresearch/MovieGenBench` 仓库 |
| 视频数量 | 1003 个 |
| 文件命名 | `{index}.mp4`（0-indexed，直接在 `moviegen_bench/` 目录下） |

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

> **注意**：`echo` 写入 `~/.bashrc` 后，当前终端需要 `source ~/.bashrc` 才能生效。或者直接在当前终端 `export DASHSCOPE_API_KEY="你的key"` 即可立即使用。

#### Step 5: 创建目录 + 下载模型

```bash
mkdir -p /root/autodl-tmp/models
mkdir -p /root/autodl-tmp/outputs

# 设置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 下载 Wan 2.1-T2V-1.3B-Diffusers 模型（约 27GB）
# 注意：必须是 Diffusers 格式！
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers
```

> **注意事项**：
> - 如果 `huggingface-cli` 提示 deprecated，使用 `hf download` 代替。
> - 如果下载中断（SSL timeout），直接重新执行同一命令，支持断点续传。
> - 下载完成后验证：`ls /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers/model_index.json` 必须存在。

#### Step 6: 下载 MovieGenBench 数据集

```bash
# 6a. 下载视频压缩包（约 15GB，CloudFront CDN）
mkdir -p /root/autodl-tmp/data
cd /root/autodl-tmp/data
wget -c https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz

# 6b. 解压
tar -xzf MovieGenVideoBench.tar.gz
mv MovieGenVideoBench moviegen_bench

# 6c. 下载 Prompt 文件
cd /root/autodl-tmp/data/moviegen_bench
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
wget https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBenchWithTag.csv

# 6d. 验证（视频文件直接在 moviegen_bench/ 目录下）
ls *.mp4 | wc -l  # 应该是 1003
head -3 MovieGenVideoBench.txt

# 6e. 清理压缩包释放空间（可选，省约 15GB）
rm -f /root/autodl-tmp/data/MovieGenVideoBench.tar.gz
```

> 如果 GitHub raw 文件下载慢，可加代理：
> ```bash
> wget https://ghproxy.com/https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBench.txt
> wget https://ghproxy.com/https://raw.githubusercontent.com/facebookresearch/MovieGenBench/main/benchmark/MovieGenVideoBenchWithTag.csv
> ```

#### Step 7: 验证环境

```bash
cd /root/autodl-tmp/videofake
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from diffusers import WanPipeline; print('diffusers OK')"
python -c "from pflow import PFlowPipeline; print('pflow OK')"
```

---

### 阶段二：有卡模式 — Wan 2.1 本地推理实验

切到有卡模式开机后：

#### Step 1: 确认环境

```bash
source /etc/network_turbo
export DASHSCOPE_API_KEY="你的百炼API Key"

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
python scripts/run_pflow_paper.py \
    --video_index 23 \
    --seed 42
```

这会执行完整的 Algorithm 1：
1. 加载参考视频 `23.mp4`（Borneo wildlife on the Kinabatangan River）
2. 计算 Noise Prior Enhancement
3. 固定 3 轮 Test-Time Prompt Optimization
4. 每轮生成视频 + VLM 分析 + 优化 prompt
5. 输出到 `/root/autodl-tmp/outputs/test_023/`

---

## 四、命令速查

```bash
# 按视频 index 运行（自动读取 prompt + 视频路径）
python scripts/run_pflow_paper.py --video_index 23 --seed 42

# 指定视频和 prompt 运行
python scripts/run_pflow_paper.py \
    --reference_video /root/autodl-tmp/data/moviegen_bench/23.mp4 \
    --prompt "Borneo wildlife on the Kinabatangan River" \
    --output_dir /root/autodl-tmp/outputs/test_023 \
    --seed 42

# 只跑 Noise Prior（不调 VLM，快速验证）
python scripts/run_pflow_paper.py --video_index 23 --noise_prior_only --seed 42

# 修改迭代次数
python scripts/run_pflow_paper.py --video_index 23 --i_max 5 --seed 42

# 后台运行（防止 SSH 断连中断）
nohup python scripts/run_pflow_paper.py --video_index 23 --seed 42 \
    > /root/autodl-tmp/outputs/run_log.txt 2>&1 &
tail -f /root/autodl-tmp/outputs/run_log.txt
```

---

## 五、性能预期

### Wan 2.1 本地（4090）

| 指标 | 预期值 |
|------|--------|
| 模型加载 | ~30 秒 |
| 单次生成 (480P, 81帧, 50步) | ~1-2 分钟 |
| VLM 调用 (Qwen-VL-Max) | ~5-10 秒/次 |
| 完整 3 轮优化 | ~5-8 分钟 |
| GPU 显存 | ~8-12 GB |

---

## 六、输出目录结构

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

## 七、磁盘空间说明

| 目录 | 大小 | 说明 |
|------|------|------|
| `/root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers` | ~27GB | Diffusers 格式模型 |
| `/root/autodl-tmp/data/moviegen_bench` | ~15GB | 1003 个视频 + prompt 文件 |
| `/root/autodl-tmp/outputs` | ~几百MB | 实验输出（每轮视频约 50-100MB） |
| 总计 | ~42-45GB | 50GB 数据盘够用但较紧凑 |

> 如果空间紧张，可删除下载的 tar.gz 压缩包：
> ```bash
> rm -f /root/autodl-tmp/data/MovieGenVideoBench.tar.gz
> ```

---

## 八、常见问题

### Q: `model_index.json` not found 错误
说明下载了原始格式模型（`Wan2.1-T2V-1.3B`）而非 Diffusers 格式。解决：
```bash
# 删除旧模型（如果有）
rm -rf /root/autodl-tmp/models/Wan2.1-T2V-1.3B

# 下载 Diffusers 格式
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --local-dir /root/autodl-tmp/models/Wan2.1-T2V-1.3B-Diffusers
```

### Q: `DASHSCOPE_API_KEY` 环境变量未设置
```bash
# 方法 1：直接 export（当前终端立即生效）
export DASHSCOPE_API_KEY="你的key"

# 方法 2：写入 bashrc 后 source
echo 'export DASHSCOPE_API_KEY="你的key"' >> ~/.bashrc
source ~/.bashrc
```

### Q: 下载中断 / SSL timeout
HuggingFace 镜像偶尔不稳定，直接重新运行同一命令即可，支持断点续传。

### Q: `OMP_NUM_THREADS` 警告
```
libgomp: Invalid value for environment variable OMP_NUM_THREADS
```
这只是警告，不影响运行。如想消除：
```bash
export OMP_NUM_THREADS=1
```

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

### Q: 关机后数据还在吗
- **数据盘** `/root/autodl-tmp/`：关机保留
- **系统盘** `/root/`：关机保留，释放实例清空
- 模型和数据集在数据盘，不用重复下载

### Q: 想换其他视频测试
```bash
# 查看所有可用 prompt
head -50 /root/autodl-tmp/data/moviegen_bench/MovieGenVideoBench.txt

# 换 index 即可（0-1002）
python scripts/run_pflow_paper.py --video_index 0 --seed 42
```

### Q: 代码更新后运行还报旧错误
确保拉取了最新代码：
```bash
cd /root/autodl-tmp/videofake
source /etc/network_turbo
git pull origin main
```
