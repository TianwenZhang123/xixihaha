# 高质量 Prompt-视频配对数据集调研报告

## 1. MovieGenBench 仓库分析

**仓库地址**: [facebookresearch/MovieGenBench](https://github.com/facebookresearch/MovieGenBench)  
**发布方**: Meta (Facebook Research)  
**许可协议**: CC-BY-NC

MovieGenBench 是 Meta 发布的视频/音频生成评测基准，配套其 Movie Gen 系列基础模型（可生成 1080p HD 多宽高比视频及同步音频）。仓库包含两个核心部分：

### 1.1 Movie Gen Video Bench

包含 **1003 条文本 prompt**，覆盖五大测试维度：

- 人体活动（肢体运动、口部动作、表情等）
- 动物
- 自然风光
- 物理现象（流体动力学、重力、加速、碰撞、爆炸等）
- 非常规主题与活动

同时覆盖高/中/低三种运动幅度级别。仓库提供 `benchmark/MovieGenVideoBench.txt`（纯 prompt 列表）和 `benchmark/MovieGenVideoBenchWithTag.csv`（含概念分类和运动级别标注）。Movie Gen 生成的对应视频可通过 [CloudFront 链接](https://d14whct5a0wtwm.cloudfront.net/moviegen/MovieGenVideoBench.tar.gz) 或 [HuggingFace](https://huggingface.co/datasets/meta-ai-for-media-research/movie_gen_video_bench) 下载。

### 1.2 Movie Gen Audio Bench

包含 **527 条生成视频**及对应的音效和音乐 prompt，覆盖室内、城市、自然、交通等多种声学环境和音效类别（人声、动物、物体等）。支持评估音效生成、视频转音频以及文本+视频到音频生成等任务。

### 1.3 局限性

MovieGenBench 本质上是一个**评测基准**（benchmark），而非训练数据集。prompt 数量有限（约 1000 条），视频均为 AI 生成（非真实拍摄），且以单镜头为主，不涉及多镜头叙事结构。

---

## 2. 类似高质量 Prompt-视频配对数据集

### 第一梯队：专为多镜头场景设计

#### 2.1 MuSS（Multi-Shot Subject-to-Video）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2604.23789](https://arxiv.org/abs/2604.23789) (2025) |
| 代码 | [GitHub](https://github.com/zhang-haojie/MuSS) |
| 来源 | 3000+ 部电影 |
| 定位 | 多镜头视频 + Subject-to-Video 生成 |

MuSS 是目前最契合"多镜头高质量视频"需求的数据集。它提供双轨数据：复杂蒙太奇转场和以主体为中心的叙事。采用两阶段 VLM 重新标注管线——先为每个镜头生成精细描述，再串联为连贯叙事。重点解决三个在单镜头中难以暴露的问题：跨镜头主体一致性、叙事连贯性和转场合理性。同时配套了一个 cinematic narrative benchmark 用于评测。

#### 2.2 MovieBench（CVPR 2025）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2411.15262](https://arxiv.org/abs/2411.15262) |
| 代码 | [GitHub](https://github.com/showlab/MovieBench) |
| 数据 | [HuggingFace](https://huggingface.co/datasets/weijiawu/MovieBench)（视频需申请） |
| 来源 | 91 部电影，平均时长 45.6 分钟 |
| 机构 | 新加坡国立大学 Show Lab + 浙江大学 |

MovieBench 提供**三层级标注**结构：

- **电影级**：整体剧情概要
- **场景级**：场景描述和角色关系
- **镜头级**：每个 shot 的视觉描述

特别强调跨场景角色 ID 一致性和连贯叙事线，是 script-to-movie 生成范式的代表性数据集。

#### 2.3 Cine250K + CineTrans（ICLR 2026）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2508.11484](https://arxiv.org/abs/2508.11484) |
| 代码 | [GitHub](https://github.com/Vchitect/CineTrans) |
| 规模 | 从 63.3 万条 Vimeo 视频中筛选出 25 万条 |
| 定位 | 电影级转场效果（叠化、淡入淡出等） |

Cine250K 包含高美学质量的多镜头视频-文本对，每条数据都有详细的 shot 标注（镜头边界、转场类型等）。CineTrans 是首个时间级可控的自动化转场模型，观测到扩散模型在处理多镜头时 Attention Map 呈现块对角结构（镜头内关联强、镜头间关联弱）。

#### 2.4 ConStoryBoard + STAGE（CVPR 2026）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2512.12372](https://arxiv.org/abs/2512.12372) |
| 代码 | [GitHub](https://github.com/escapistmost/Storyboard-Anchored-Generation) |
| 定位 | 分镜脚本式多镜头视频生成 |

ConStoryBoard 是大规模结构化标注的电影镜头数据集，为每个 shot 标注了起始帧-结束帧对。STAGE 框架用 storyboard 作为视觉锚点来保证长程镜头一致性，重点关注镜头内故事连贯性和电影语言表达（如推拉镜头）。

#### 2.5 HoloCine（CVPR 2026 Highlight）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2510.20822](https://arxiv.org/abs/2510.20822) |
| 代码 | [GitHub](https://github.com/yihao-meng/HoloCine) |
| 机构 | 蚂蚁集团 + 香港科技大学 |
| 定位 | 端到端电影级多镜头长视频叙事 |

HoloCine 附带一个大规模、分层标注的多镜头场景数据集，配套 100 条多样化层级 prompt 的评测基准。通过 Window Cross-Attention（镜头级文本控制）和 Sparse Inter-Shot Self-Attention（跨镜头一致性）实现分钟级视频生成，在长期一致性、叙事忠实度和镜头转场控制方面建立了新的 SOTA。

#### 2.6 ShotAdapter Multi-Shot Video Dataset（CVPR 2025）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2505.07652](https://arxiv.org/abs/2505.07652) |
| 主页 | [shotadapter.github.io](https://shotadapter.github.io/) |
| 机构 | UIUC |
| 定位 | 单镜头模型微调为多镜头模型 |

ShotAdapter 提供了一套从已有单镜头数据集构建多镜头视频数据集的管线，支持控制镜头数量、持续时间和内容，每个镜头都有独立的 text prompt。这是一个轻量级框架，通过最小微调即可将预训练的文本到视频模型升级为多镜头生成模型。

---

### 第二梯队：大规模高质量文本-视频对（单镜头为主，可作为多镜头素材源）

#### 2.7 Koala-36M（CVPR 2025，快手）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2410.08260](https://arxiv.org/abs/2410.08260) |
| 代码 | [GitHub](https://github.com/KlingAIResearch/Koala-36M) |
| 规模 | 3600 万条视频 |
| 字幕 | 平均 200+ 词的精细描述 |

目前唯一同时具备**千万级规模**和**高质量精细文本描述**的数据集。提出了 VTSS（Video Training Suitability Score）进行质量筛选，时间切分精准，条件与视频内容一致性显著优于前作。

#### 2.8 Panda-70M（CVPR 2024，Snap Research）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2402.19479](https://arxiv.org/abs/2402.19479) |
| 代码 | [GitHub](https://github.com/snap-research/Panda-70M) |
| 规模 | 7080 万条视频-字幕对 |
| 来源 | 从 HD-VILA-100M 筛选 380 万高分辨率视频并切分 |

规模巨大的视频-字幕数据集，使用多个跨模态教师模型生成字幕，再通过微调检索模型自动选择最佳字幕。适合大规模预训练。

#### 2.9 OpenVid-1M / OpenVidHD-0.4M（ICLR 2025）

| 属性 | 详情 |
|------|------|
| 论文 | [HuggingFace](https://huggingface.co/papers/2407.02371) |
| 规模 | 100 万+ 文本-视频对，其中 43.3 万条为 1080p |
| 定位 | 精确高质量、开放场景 |

包含富有表现力的字幕描述，覆盖面广。OpenVidHD-0.4M 子集全部为 1080p 高清视频，推进高清视频生成研究。

#### 2.10 MiraData（NeurIPS 2024，腾讯）

| 属性 | 详情 |
|------|------|
| 论文 | [arXiv: 2407.06358](https://arxiv.org/abs/2407.06358) |
| 平均视频时长 | 72.1 秒 |
| 平均字幕长度 | 318 字（结构化） |
| 标注方式 | GPT-4V |

长视频数据集，具有高运动强度和详细的结构化字幕。同时引入 MiraBench 用于评估视频生成中的时间一致性和运动强度。特别适合需要高运动幅度和长时间一致性的场景。

#### 2.11 ShareGPT4Video（NeurIPS 2024）

| 属性 | 详情 |
|------|------|
| 论文 | [GitHub](https://github.com/ShareGPT4Omni/ShareGPT4Video) |
| 规模 | 4.8M 条视频描述 |
| 其中 | 4 万条 GPT-4V 直接标注 + 40 万条隐式分割字幕 |

密集且精确的视频描述数据集，旨在同时提升大型视频语言模型的理解能力和文本到视频模型的生成质量。

---

### 第三梯队：评测基准（可作为 Prompt 参考）

#### 2.12 VBench

综合视频生成评测基准，提供多维度分层评估体系和精心设计的 prompt 套件。将"视频生成质量"分解为多个定义清晰的维度，便于细粒度客观评估。

主页: [vchitect.github.io/VBench-project](https://vchitect.github.io/VBench-project/)

#### 2.13 T2V-CompBench（CVPR 2025）

首个组合式文本到视频生成评测基准，关注多物体、多属性、空间关系、动作组合等复合描述的生成能力评估。

---

## 3. 数据集对比总结

| 数据集 | 规模 | 多镜头 | 视频来源 | 字幕质量 | 分辨率 | 主要用途 |
|--------|------|--------|----------|----------|--------|----------|
| MovieGenBench | 1003 prompt | 否 | AI 生成 | 中等 | 1080p | 评测 |
| **MuSS** | 3000+ 电影 | **是** | 真实电影 | 高（两阶段 VLM） | 高 | 训练+评测 |
| **MovieBench** | 91 部电影 | **是** | 真实电影 | 高（三层级标注） | 高 | 训练+评测 |
| **Cine250K** | 25 万条 | **是** | Vimeo | 高（shot 标注） | 高 | 训练 |
| **ConStoryBoard** | 大规模 | **是** | 真实电影 | 高（帧对标注） | 高 | 训练 |
| **HoloCine** | 大规模 | **是** | 真实视频 | 高（分层标注） | 高 | 训练+评测 |
| **ShotAdapter** | 管线生成 | **是** | 合成 | 中高 | 可变 | 训练 |
| Koala-36M | 3600 万 | 否 | 真实视频 | 高（200+ 词） | 高 | 预训练 |
| Panda-70M | 7080 万 | 否 | YouTube | 中高 | 可变 | 预训练 |
| OpenVid-1M | 100 万 | 否 | 真实视频 | 高 | 1080p | 预训练 |
| MiraData | 中等 | 否 | 真实视频 | 高（318 词） | 高 | 训练 |
| ShareGPT4Video | 4.8M 描述 | 否 | 真实视频 | 极高（GPT-4V） | 可变 | 训练 |

---

## 4. 推荐策略

### 场景一：多镜头视频生成研究或训练

首选 **MuSS**（电影级多镜头数据，规模大、标注完善）和 **MovieBench**（层级结构清晰，角色一致性好），配合 **Cine250K** 获取转场相关数据。

### 场景二：大规模预训练 + 多镜头微调（两阶段方案）

用 **Koala-36M** 或 **Panda-70M** 做预训练底座，再用 MuSS / MovieBench / Cine250K 等多镜头数据集做微调。

### 场景三：高质量 Prompt 模板参考

**MovieGenBench** 的 1003 条 prompt（带标签和运动级别）是很好的起点，再结合 **MuSS** 和 **MovieBench** 的分层 prompt 结构来设计多镜头 prompt 体系。

---

*调研日期：2025 年 7 月*  
*主要参考来源：arXiv, GitHub, HuggingFace, CVPR/ICLR/NeurIPS 论文*
