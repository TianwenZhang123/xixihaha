"""
P-Flow: Video Reproduction via Prompt Optimization + Noise Prior.

通过命令行 flag 控制各改动点，一个管线搞定所有配置。

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Entry Point                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  run.py              — CLI 入口 (--svd --inversion --blend ...)     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                         Core Modules                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  pipeline.py         — 统一管线 (PFlowConfig + PFlowPipeline)       │
    ├─────────────────────────────────────────────────────────────────────┤
    │                    Shared Infrastructure                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  flow_matching.py    — FlowMatchingInverter (Euler + Midpoint)       │
    │  svd_filter.py       — SVDFilter V2 (空间去内容 + 时间保运动 + 频段分离)│
    │  vlm_client.py       — Local/DashScope/Mock VLM 客户端              │
    │  video_utils.py      — 视频 I/O 和处理工具                          │
    │  shot_detect.py      — 镜头边界检测 (TransNetV2 / PySceneDetect)   │
    │  distributed.py      — 单 GPU 推理工具                              │
    └─────────────────────────────────────────────────────────────────────┘

改动点 (L1 + L2):
    L1 (Prompt Optimization):
        --iter N       迭代 VLM 优化 (N轮反馈循环)
        --composite    三面板垂直拼接 (ref|prev|current 送 VLM 对比)

    L2 (Noise Prior):
        --inversion    Flow Matching Inversion (从参考视频反演噪声)
        --svd          SVD V2 两阶段滤波 (空间去内容 + 时间保运动 + 频段分离)
        --blend        噪声混合 (η = √α·η_temporal + √(1-α)·η_random)
        --midpoint     二阶中点法 ODE 求解器 (替代 Euler)

组合示例:
    baseline:     无任何 flag → caption + 一次生成
    +noise_prior: --inversion --svd --blend → 噪声先验引导
    +iteration:   --iter 10 → 迭代优化
    full:         --inversion --svd --blend --iter 10 --composite

用法:
    python run.py --video ref.mp4 --caption "..." --inversion --svd --blend
"""

__version__ = "8.0.0"
__task__ = "video_reproduction"

from .pipeline import PFlowConfig, PFlowPipeline
from .svd_filter import SVDFilter, SVDFilterConfig
from .shot_detect import ShotDetector, detect_shots, split_video_to_shots

__all__ = [
    "PFlowConfig", "PFlowPipeline",
    "SVDFilter", "SVDFilterConfig",
    "ShotDetector", "detect_shots", "split_video_to_shots",
]
