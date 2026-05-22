# 视频 Prompt 反演与优化 Pipeline

## 一、数据集与模型选型

### 数据集

| 数据集 | 规模 | 用途 |
|--------|------|------|
| Open-VFX | 15 类视觉特效 | 主评估集 |
| MovieGenBench | 1003 样本 | 通用视频质量评估 |
| VidProM | 180K video-prompt 对 | Prompt 反演评估 + RAG 检索库 |

数据规范：480×832，81 帧 @16fps，mp4 H.264，归一化到 [-1, 1]。

### VLM 选型

| 模型 | 角色 | 理由 |
|------|------|------|
| **Qwen2.5-VL-72B** | 主力 | 视频理解≈GPT-4o，绝对时间编码，国内稳定 |
| GPT-4o | 交叉验证 | 排除系统性偏差 |
| InternVL3-78B | 本地批量 | 开源可部署 |

### T2V 选型

| 模型 | 角色 | 理由 |
|------|------|------|
| **Wan 2.1-14B** | 主力 | P-Flow 原始实现，Flow Matching 架构适配 Inversion |
| Wan 2.1-1.3B | 快速验证 | 单 4090 可跑 |
| CogVideoX-5B / HunyuanVideo | 对比 | 验证模型无关性 |

### 评估指标

CLIP-Score（语义一致性）、FVD（分布距离）、光流一致性（运动保真度）、帧间 SSIM 方差（时序连贯性）、VBench（综合）、VLM-as-Judge（5 维度 1-5 分）。

---

## 二、Pipeline 流程与优化点

下图展示完整的 VLM → Prompt → T2V 流程。每个环节标注了可插入的优化方法。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     参考视频 V_ref 输入                              │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│                    ┌────────────┼────────────┐                              │
│                    ▼                         ▼                              │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐            │
│  │  VLM 视频理解             │  │  VAE Encode + Inversion       │            │
│  │  (生成初始 Prompt P_0)    │  │  (提取运动先验 η_temporal)    │            │
│  │                          │  │                              │            │
│  │  ⚡优化点:                │  │  ⚡优化点:                    │            │
│  │  • 多粒度描述(4层)       │  │  • RF-Solver 高精度反演       │            │
│  │  • 多VLM集成             │  │  • Predictor-Corrector       │            │
│  │  • 自适应关键帧选择       │  │  • Midpoint 二阶积分         │            │
│  │  • 内容/运动解耦描述      │  │  • 自适应SVD阈值(ρ_s/ρ_m)   │            │
│  │                          │  │  • 多尺度SVD滤波             │            │
│  │                          │  │  • 频域FFT替代SVD            │            │
│  │                          │  │  • DisMo运动编码器替代        │            │
│  └────────────┬─────────────┘  └──────────────┬───────────────┘            │
│               │                                │                            │
│               ▼                                ▼                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                      │  │
│  │                    迭代优化循环 (i = 1..10)                           │  │
│  │                                                                      │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  ① 噪声混合: η = √α·η_temporal + √(1-α)·η_new                │  │  │
│  │  │                                                                │  │  │
│  │  │  ⚡优化点:                                                      │  │  │
│  │  │  • 自适应α(前期大/后期小)                                       │  │  │
│  │  │  • 分步注入(早期强先验→后期无先验)                               │  │  │
│  │  │  • Video-MSG结构化噪声(视频草稿→inversion)                      │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  ② T2V 生成: V_i = T2V(P_i, η)                                │  │  │
│  │  │                                                                │  │  │
│  │  │  ⚡优化点:                                                      │  │  │
│  │  │  • Prompt压缩重排(核心语义前置,≤77token)                        │  │  │
│  │  │  • Negative Prompt工程                                          │  │  │
│  │  │  • INT8量化 + Flash Attention 2加速                             │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  ③ VLM 评审: 比较 V_ref vs V_i，输出分析 + 候选 Prompt         │  │  │
│  │  │                                                                │  │  │
│  │  │  ⚡优化点:                                                      │  │  │
│  │  │  • 结构化5维度分析(运动/外观/空间/时序/交互)                     │  │  │
│  │  │  • 历史感知增量反馈(LLM摘要避免重复/遗忘)                       │  │  │
│  │  │  • 置信度控制修改幅度                                           │  │  │
│  │  │  • 光流分析结果注入VLM上下文                                    │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  ④ Prompt 优化: VLM输出 → 检索增强 → 排序 → P_{i+1}           │  │  │
│  │  │                                                                │  │  │
│  │  │  ⚡优化点:                                                      │  │  │
│  │  │  • RAG检索增强(RAPO++ Stage1, 对齐训练数据分布)                 │  │  │
│  │  │  • 3R: Retrieval→Refinement→Ranking                            │  │  │
│  │  │  • 并行多候选(不同温度/不同检索) + 排序选优                     │  │  │
│  │  │  • 结构化模板([Scene/Subject/Motion/Effect/Style/Camera])       │  │  │
│  │  │  • 分层优化(前3轮宏观场景,后7轮微观运动)                        │  │  │
│  │  │  • 光流引导embedding优化(MotionPrompt)                          │  │  │
│  │  │  • 树搜索(Beam/MCTS)全局最优                                    │  │  │
│  │  │  • RL/DPO微调VLM的prompt生成策略                                │  │  │
│  │  │  • 遗传算法(LLM交叉变异 + CLIP适应度)                           │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  │                              ↓                                       │  │
│  │                     下一轮迭代 / 结束                                │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                 │                                           │
│                                 ▼                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  ⑤ 评估选优: argmax(CLIP + FVD + 光流一致性 + VLM-Judge)            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 各环节优化点汇总

| 环节 | 优化点 | 方法来源 |
|------|--------|---------|
| **VLM 初始描述** | 多粒度 4 层描述（场景/运动/效果/时序）→ LLM 融合 | 自研 |
| | 多 VLM 集成（Qwen + GPT-4o + Gemini 去重融合） | 自研 |
| | 光流自适应关键帧（运动剧烈处密采样） | 自研 |
| | 内容/运动解耦分支描述 | DisMo (NeurIPS 2025) |
| **Inversion** | RF-Solver 高精度 ODE 求解（误差降 40-60%） | ICML 2025 |
| | Predictor-Corrector 近乎无损反演 | UniEdit-Flow |
| | Midpoint 二阶积分 | P-Flow 已实现 |
| **SVD 滤波** | 基于光流幅度自适应 ρ_s/ρ_m | 自研 |
| | 多尺度（帧级/片段级/全局级）分别滤波 | 自研 |
| | FFT 频域替代（物理意义更明确） | 自研 |
| | DisMo 运动编码器替代 SVD | NeurIPS 2025 |
| | 比特率控制解耦 | ICCV 2025 |
| **噪声混合** | 自适应 α（前期 0.01 → 后期 0.0001） | 自研 |
| | 分步注入（不同 timestep 不同强度） | 自研 |
| | Video-MSG 结构化噪声（视频草稿 → inversion） | arXiv:2504.08641 |
| | FlowEdit 无反演路径 | FlowEdit |
| **T2V 生成** | Prompt 压缩重排（核心前置 ≤77 token） | 自研 |
| | RAPO++ 训练数据对齐改写 | CVPR 2025 |
| | Negative Prompt 工程 | 工程经验 |
| **VLM 评审** | 结构化 5 维度量化分析 | P-Flow Listing 1 |
| | 历史摘要避免重复/遗忘 | 自研 |
| | 置信度控制修改幅度 | 自研 |
| | 光流分析结果注入 VLM 上下文 | 自研 |
| **Prompt 优化** | RAG 检索增强对齐训练分布 | RAPO++ (CVPR 2025) |
| | 3R: Retrieval → Refinement → Ranking | arXiv:2603.01509 |
| | 并行多候选 + 排序选优 | 自研 |
| | 结构化模板（固定字段，按需修改） | 自研 |
| | 分层优化（宏观→微观） | 自研 |
| | MotionPrompt 光流引导 embedding | CVPR 2025 |
| | 树搜索 Beam/MCTS | 自研 |
| | RL/DPO 微调 VLM prompt 策略 | VPO (ICCV 2025) |
| | 遗传算法搜索 | 自研 |

---

## 三、已验证内容

### 3.1 已跑通的 Baseline 流程

基于 P-Flow 代码 + Wan 2.1-1.3B 模型，已完整跑通以下流程：

```
V_ref → VAE Encode → Flow Matching Inversion (Euler, 50步)
     → SVD 空间滤波 (ρ_s=0.1) → SVD 时序保留 (ρ_m=0.9) → η_temporal
     → 10轮迭代: 噪声混合(α=0.001) → T2V生成 → VLM评审(Qwen-VL-Max) → Prompt更新
     → 离线评估选优
```

### 3.2 已验证的参数配置

```yaml
generation: {height: 480, width: 832, frames: 81, fps: 16, cfg: 5.0, steps: 50}
inversion: {steps: 50, cfg: 1.0, solver: euler}
svd: {rho_s: 0.1, rho_m: 0.9}
noise_blend: {alpha: 0.001}
iteration: {i_max: 10, no_early_stop: true}
hardware: {gpu: A800-80GB (14B) / RTX4090 (1.3B), vlm: DashScope API}
```

### 3.3 已验证的实验结论

**1.3B 模型能力边界**：能完成基本场景渲染和单主体动态，无法处理细粒度物种区分和多步因果行为链。3 轮迭代即趋于饱和。

**VLM 行为观察**：
- 参考视频理解随迭代逐步加深（递进式分析）
- 能一致识别核心缺陷并持续追踪（跨迭代一致性）
- comparison 字段展示递进式思维（增量反馈）

**Noise Prior 效果**：α=0.001 下时序先验仅作为"微弱运动暗示"，生成模型保持充分创造自由度。每轮重新采样 η_new 提供探索多样性。

**性能数据**：单样本 17-22 min（10 轮，14B），显存 40-50 GB，prompt 从 7-50 词扩展到 150-250 词。

### 3.4 已确认的方法论互补关系

| | P-Flow (Baseline) | Reverse Prompt Engineering |
|---|---|---|
| 目标 | 视觉效果迁移 | Prompt 精确恢复 |
| 训练 | 否 (Training-Free) | 是 (RL Fine-tune) |
| 噪声先验 | 有 | 无 |
| 迭代 | 10 轮 VLM-guided | 单次推理 |
| 融合方式 | RPE Phase 1 作为 P_0 生成器 → P-Flow 迭代优化 |

---

## 参考文献

1. P-Flow: Prompting Visual Effects Generation. arXiv:2603.22091
2. Reverse Prompt Engineering. github.com/cyprivlab/reverse-prompt-engineering
3. RAPO++: Cross-Stage Prompt Optimization for T2V. CVPR 2025
4. MotionPrompt: Optical-Flow Guided Prompt Optimization. CVPR 2025
5. 3R: RAG-based Prompt Optimization. arXiv:2603.01509
6. VPO: Video Prompt Optimization. ICCV 2025
7. Video-MSG: Training-free Guidance via Multimodal Planning. arXiv:2504.08641
8. RF-Solver: Taming Rectified Flow for Inversion and Editing. ICML 2025
9. UniEdit-Flow. arXiv:2504.13109
10. DisMo: Disentangled Motion Representation. NeurIPS 2025
11. Bitrate-Controlled Diffusion. ICCV 2025
12. Motion-Textual Inversion. Disney Research
13. Qwen2.5-VL. arXiv:2502.13923
14. Wan 2.1. Alibaba
15. VBench
