"""
P-Flow: Video Reproduction via Orthogonal Three-Layer Decomposition.

通过命令行 flag 控制各改动点，一个管线搞定所有配置。

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Entry Point                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  run.py              — CLI 入口 (--svd --feature-inject ...)        │
    ├─────────────────────────────────────────────────────────────────────┤
    │                         Core Modules                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  pipeline.py         — 统一管线 (PFlowConfig + PFlowPipeline)       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                    Shared Infrastructure                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  flow_matching.py    — FlowMatchingInverter (Euler ODE反演)         │
    │  svd_filter.py       — SVDFilter (空间去内容 + 时间保运动 + 渐进多尺度)│
    │  vlm_client.py       — Local/DashScope/Mock VLM 客户端              │
    │  video_utils.py      — 视频 I/O 和处理工具                          │
    │  shot_detect.py      — 镜头边界检测 (TransNetV2 / PySceneDetect)    │
    │  distributed.py      — 单 GPU 推理工具                              │
    └─────────────────────────────────────────────────────────────────────┘

三层架构:
    L1 (Prompt): 外部预处理 (首尾替换 + 三版本择优), 可选 --iter/--composite
    L2 (Noise Prior): --svd 一键开启 (反演 + 两阶段SVD + sigmoid自适应α混合 + 渐进多尺度)
    L3 (Feature Injection): --feature-inject 开启 (三层自适应门控: 中峰调度 + 余弦门控 + QS)

组合示例:
    baseline:     无 flag → caption + 一次生成
    +L2:          --svd → 噪声先验引导
    +L2+L3:       --svd --feature-inject → 完整 P-Flow

用法:
    python run.py --data_dir data/videos --caption_dir data/captions
"""

__version__ = "8.1.0"
__task__ = "video_reproduction"

from .pipeline import PFlowConfig, PFlowPipeline
from .svd_filter import SVDFilter, SVDFilterConfig
from .shot_detect import ShotDetector, detect_shots, split_video_to_shots

__all__ = [
    "PFlowConfig", "PFlowPipeline",
    "SVDFilter", "SVDFilterConfig",
    "ShotDetector", "detect_shots", "split_video_to_shots",
]
