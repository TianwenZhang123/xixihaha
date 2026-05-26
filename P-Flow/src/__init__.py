"""
P-Flow: Video Reproduction via Iterative Prompt Optimization + Noise Prior.

通过命令行 flag 控制各改动点，一个管线搞定所有配置。

    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Entry Point                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  run.py              — CLI 入口 (--svd --inversion --blend ...)     │
    ├─────────────────────────────────────────────────────────────────────┤
    │                         Core Modules                                 │
    ├─────────────────────────────────────────────────────────────────────┤
    │  pipeline.py         — 统一管线 (PFlowConfig + PFlowPipeline)       │
    │  baseline.py         — 纯 baseline (caption → 一次生成)             │
    ├─────────────────────────────────────────────────────────────────────┤
    │                    Shared Infrastructure                              │
    ├─────────────────────────────────────────────────────────────────────┤
    │  flow_matching.py    — FlowMatchingInverter (Euler + Midpoint)       │
    │  svd_filter.py       — SVDFilter (空间去内容 + 时间保运动)           │
    │  vlm_client.py       — Local/DashScope/Mock VLM 客户端              │
    │  video_utils.py      — 视频 I/O 和处理工具                          │
    │  distributed.py      — 单 GPU 推理工具                              │
    └─────────────────────────────────────────────────────────────────────┘

改动点 (通过 flag 启用):
    --inversion    Flow Matching Inversion (从参考视频反演噪声)
    --svd          SVD 两阶段滤波 (空间去内容 + 时间保运动)
    --blend        噪声混合 (η = √α·η_temporal + √(1-α)·η_random)
    --iter N       迭代 VLM 优化 (N轮反馈循环)
    --midpoint     二阶中点法 ODE 求解器 (替代 Euler)
    --composite    三面板垂直拼接 (ref|prev|current 送 VLM 对比)

用法:
    python run.py --video ref.mp4 --caption "..." --inversion --svd --blend --iter 10
"""

__version__ = "5.0.0"
__task__ = "video_reproduction"

from .pipeline import PFlowConfig, PFlowPipeline

__all__ = ["PFlowConfig", "PFlowPipeline"]
