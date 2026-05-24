# P-Flow 实验运行说明

## 运行命令

```bash
python scripts/run_pflow_paper.py --video_index 23 --seed 42
```

---

## 实验目标

给定一段参考视频（MovieGenBench 第 23 号视频），通过迭代优化 prompt 的方式，让 T2V 模型生成出视觉效果尽可能接近参考视频的新视频。

---

## `--video_index 23` 做的事

从 `/root/autodl-tmp/data/moviegen_bench/` 中读取 `23.mp4` 作为参考视频，从 `MovieGenVideoBench.txt` 第 24 行（0-indexed）读取对应的文本 prompt 作为初始 prompt P₀。

---

## 完整运行流程（Algorithm 1）

### 阶段 1：Noise Prior Enhancement（噪声先验增强）

这一步的目的是从参考视频中"偷"一点运动节奏信息，注入到生成过程的初始噪声中。

1. 加载参考视频 → VAE 编码到 latent 空间得到 x₁
2. **Flow Matching Inversion**：沿 ODE 从 t=1 反向积分到 t=0，得到参考视频对应的噪声 η_inv
3. **SVD 空间滤波**：把 η_inv reshape 成 (C×F, H×W)，做 SVD，减去 top 10% 的空间分量 → 去掉静态场景内容
4. **SVD 时序保留**：把结果 reshape 成 (C×H×W, F)，做 SVD，只保留 top 90% 时序分量 → 保留运动模式
5. **噪声混合**：η = √0.001 × η_temporal + √0.999 × η_new（99.9% 是随机噪声，只有 0.1% 的参考视频运动引导）

这个增强噪声 η 会被用于后续所有迭代的视频生成。

### 阶段 2：固定 3 轮迭代优化

每一轮做的事情：

```
第 i 轮（i = 1, 2, 3）:

  (a) 用当前 prompt Pᵢ + 增强噪声 η → Wan 2.1 生成视频 Vᵢ

  (b) 创建垂直对比视频（上/中/下三个面板）:
      - 顶部 Panel A: 参考视频
      - 中部 Panel B: 上一轮生成的视频 Vᵢ₋₁（第1轮没有）
      - 底部 Panel C: 本轮生成的视频 Vᵢ

  (c) 把对比视频的关键帧 + 历史记录发给 VLM (Qwen-VL-Max)
      VLM 的任务："看看参考视频和生成视频有什么区别？"

  (d) VLM 返回:
      - analysis: 参考视频描述、生成视频描述、差异对比
      - refined_prompt: 改进后的 prompt Pᵢ₊₁

  (e) 把 {Vᵢ, Pᵢ, 分析结果} 存入轨迹历史
```

### 阶段 3：保存所有结果

输出到 `/root/autodl-tmp/outputs/test_023/`：

- `generated_iter_001.mp4`、`002.mp4`、`003.mp4` — 三轮生成的视频
- `reference.mp4` — 参考视频副本
- `prompts_history.json` — prompt 演化轨迹
- `full_trajectory.json` — 完整实验记录
- `composites/` — 每轮发给 VLM 的对比视频

---

## 核心思路

论文的核心洞察是——VLM 能"看"，T2V 模型能"画"，但用户很难用文字精确描述视觉特效。所以让 VLM 反复对比"参考视频 vs 生成视频"的差异，自动帮你改 prompt，每轮生成的视频就会更接近参考。Noise Prior 则是从噪声空间给一个额外的运动节奏暗示，和 prompt 优化互补。

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video_index` | 无 | MovieGenBench 视频编号 (0~1002) |
| `--seed` | 42 | 随机种子，保证可复现 |
| `--i_max` | 3 | 迭代轮数（论文用 10，1.3B 小模型用 3） |
| `--mock` | 关闭 | 开启后不用 GPU/API，仅测试逻辑 |
| `--noise_prior_only` | 关闭 | 只跑 noise prior，跳过迭代优化 |
| `--alpha` | 0.001 | 噪声混合权重 |
| `--rho_s` | 0.1 | 空间 SVD 去除比例 |
| `--rho_m` | 0.9 | 时序 SVD 保留比例 |

---

## 对应论文章节

| 流程步骤 | 论文章节 | 代码文件 |
|----------|----------|----------|
| Flow Matching Inversion | Section 3.2-3.3 | `pflow/flow_matching.py` |
| SVD 滤波 | Section 3.3, Eq. 6-8 | `pflow/svd_filter.py` |
| 噪声混合 | Section 3.3, Eq. 10 | `pflow/noise_prior.py` |
| 迭代优化主循环 | Algorithm 1 (Appendix A) | `pflow/pipeline.py` |
| VLM 结构化指令 | Section 3.4, Listing 1 | `pflow/vlm_client.py` |
| 垂直 Composite | Section 3.5 | `pflow/prompt_optimizer.py` |
| 轨迹管理 | Section 3.5 | `pflow/trajectory.py` |
