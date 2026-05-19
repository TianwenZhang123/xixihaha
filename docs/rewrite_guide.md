# P-Flow 代码重写指南

## 概述

本次重写的目标是将代码实现严格对齐论文 **P-Flow (arXiv:2603.22091)** 的 Algorithm 1 和 Listing 1，消除之前实现中与论文描述不一致的部分。

---

## 重写前后对比

| 维度 | 重写前 | 重写后（对齐论文） |
|------|--------|-------------------|
| 停止策略 | 基于 confidence 的 early stopping | **固定 i_max 次迭代，不提前停止** |
| VLM 输出 | 包含 confidence 字段 | **只有 analysis + refined_prompt** |
| Composite 布局 | 水平拼接 (side-by-side) | **垂直拼接 (top/middle/bottom)** |
| 最佳选择 | 在线置信度选择 | **离线 VBench/FVD 评估** |
| VLM 指令 | 自定义 prompt | **论文 Listing 1 结构化指令** |
| 视频模型 | API 调用 | **本地 Wan 2.1-T2V-1.3B** |
| VLM 接口 | 多种 API | **DashScope (Qwen3-VL-Flash)** |

---

## 文件对应关系

### 核心文件（已重写）

| 文件 | 对应论文章节 | 功能描述 |
|------|-------------|----------|
| `pflow/pipeline.py` | Algorithm 1 (Appendix A) | 主管线：编码参考视频 → Noise Prior → 固定迭代优化 → 输出全部视频 |
| `pflow/vlm_client.py` | Section 3.4, Listing 1 | VLM 客户端：DashScope API，结构化指令，JSON 输出格式 |
| `pflow/prompt_optimizer.py` | Section 3.4-3.5 | 提示词优化：垂直 composite 创建 + VLM 调用 |
| `pflow/trajectory.py` | Section 3.5 | 历史轨迹管理：只保存 V_{i-1} 在内存，全部文本历史作为上下文 |
| `pflow/video_utils.py` | Section 3.5 | 视频工具：新增 `create_vertical_composite()`，保留旧 `create_composite_video()` |
| `pflow/__init__.py` | — | 模块导出：移除 VISTAOptimizer 默认导入 |
| `config/default.yaml` | Paper parameters | AutoDL 路径 + 论文参数 (α=0.001, ρ_s=0.1, ρ_m=0.9, i_max=3) |

### 未修改文件（已经匹配论文）

| 文件 | 功能 |
|------|------|
| `pflow/noise_prior.py` | Noise Prior Enhancement: Flow Inversion + SVD Filter + Blend |
| `pflow/svd_filter.py` | SVD 空间去除 + 时间保留 |
| `pflow/flow_matching.py` | Flow Matching Inversion (Eq. 3-5) |

### 保留文件（API 模式兼容）

| 文件 | 说明 |
|------|------|
| `pflow/wan_api_client.py` | Wan API 客户端（非本地模式时使用） |
| `pflow/vista_optimizer.py` | VISTA 优化器（消融实验用） |
| `scripts/run_pflow_api.py` | API 模式入口（非论文模式） |

### 新增文件

| 文件 | 功能 |
|------|------|
| `scripts/run_pflow_paper.py` | AutoDL 入口脚本，跑论文 Algorithm 1 |
| `setup_autodl.sh` | AutoDL 一键环境安装脚本 |

---

## 论文 Algorithm 1 对应代码路径

```
Algorithm 1: Test-Time Prompt Optimization
─────────────────────────────────────────

Input: V_ref (参考视频), P_user (用户 prompt), G (视频模型)
Output: {V_1...V_{i_max}} (全部生成视频，离线评估选最佳)

1. x_1 = Encode(V_ref)              → pipeline.py: encode_video_to_latents()
2. η_inv = FlowInversion(x_1)      → noise_prior.py → flow_matching.py
3. η_temporal = SVDFilter(η_inv)    → noise_prior.py → svd_filter.py
4. η = Blend(η_temporal, η_new)     → noise_prior.py: enhance()
5. P_0 = P_user
6. For i = 1 to i_max:             → pipeline.py: run() 主循环
   a. V_i = G(P_i, η)              → pipeline.py: _generate_video()
   b. Composite = [V_ref|V_{i-1}|V_i]  → prompt_optimizer.py: _create_vertical_composite()
   c. Send to VLM                   → vlm_client.py: analyze_and_refine()
   d. Get A_i, P_{i+1}             → VLM JSON 输出
   e. Update trajectory             → trajectory.py: add_entry()
7. Return all {V_i, P_i, A_i}      → pipeline.py: 保存 full_trajectory.json
```

---

## 关键实现细节

### 1. 固定迭代（NO Early Stopping）

论文明确指出 "We acknowledge that P-Flow relies on a fixed number of iterations"，不存在基于 confidence 的提前终止：

```python
# pipeline.py
for iteration in range(1, i_max + 1):
    # 无条件执行所有迭代
    ...
```

### 2. VLM 输出格式（NO Confidence）

论文 Listing 1 定义的 VLM 输出仅包含：

```json
{
    "analysis": {
        "reference_description": "...",
        "last_generated_description": "...",
        "new_generated_description": "...",
        "comparison": "..."
    },
    "refined_prompt": "..."
}
```

### 3. 垂直 Composite 布局

论文 Section 3.5 描述为垂直排列（不是水平）：
- Panel A (top): V_ref
- Panel B (middle): V_{i-1}
- Panel C (bottom): V_i

### 4. Noise Prior 公式 (Eq. 10)

```
η = √α · η_temporal + √(1-α) · η_new
```
其中 α = 0.001，即最终噪声以随机噪声为主，但包含微弱的时序运动信息。

---

## AutoDL 运行说明

### 环境准备

```bash
# 1. 拉取代码
cd /root/autodl-tmp
git clone https://github.com/TianwenZhang123/xixihaha.git videofake
cd videofake

# 2. 安装依赖
chmod +x setup_autodl.sh
./setup_autodl.sh

# 3. 设置 API Key（重要！不要写入代码）
export DASHSCOPE_API_KEY="你的百炼API Key"
```

### 运行测试用例 #24

```bash
python scripts/run_pflow_paper.py \
    --reference_video /root/autodl-tmp/data/moviegen_bench/water_mark_out/024.mp4 \
    --prompt "A cat wakes up its owner, the owner ignores it, the cat changes strategy, the owner pulls out snacks" \
    --output_dir /root/autodl-tmp/outputs/test_024 \
    --seed 42
```

### Mock 模式测试（不需要 GPU/API）

```bash
python scripts/run_pflow_paper.py \
    --reference_video /root/autodl-tmp/data/moviegen_bench/water_mark_out/024.mp4 \
    --prompt "test" \
    --mock
```

---

## API Key 配置

**重要**: API Key 不要写入代码或配置文件。

设置方式：

```bash
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

代码中读取逻辑（`pflow/vlm_client.py`）：

```python
api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
```

---

## 输出目录结构

运行后 `output_dir` 下的文件结构：

```
/root/autodl-tmp/outputs/test_024/
├── reference.mp4                    # 参考视频副本
├── generated_iter_001.mp4           # 第1次迭代生成
├── generated_iter_002.mp4           # 第2次迭代生成
├── generated_iter_003.mp4           # 第3次迭代生成
├── full_trajectory.json             # 完整轨迹数据
├── prompts_history.json             # Prompt 演化历史
├── composites/                      # VLM 输入的垂直 composite
│   ├── composite_iter_001.mp4
│   ├── composite_iter_002.mp4
│   └── composite_iter_003.mp4
├── optimization_log/                # 每轮优化日志
│   ├── iter_001.json
│   ├── iter_002.json
│   └── iter_003.json
└── trajectory/                      # 实时轨迹备份
    └── trajectory.json
```

---

## 后续：离线最佳视频选择

论文使用 VBench 和 FVD 指标在生成的所有视频中离线选择最佳结果，这不在本仓库实现范围内。生成完成后，需要另外运行评估工具。
