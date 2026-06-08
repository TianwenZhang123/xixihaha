"""
P-Flow: Video Reproduction via Iterative Prompt Optimization + Noise Prior + Velocity Matching.

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
    │  svd_filter.py       — SVDFilter (空间去内容 + 时间保运动)           │
    │  velocity_matching.py — VelocityMatcher (轻量版Δe优化, 30步)        │
    │  attn_inject.py      — AttnInjector (Self-Attention K/V Injection)   │
    │  vlm_client.py       — Local/DashScope/Mock VLM 客户端              │
    │  video_utils.py      — 视频 I/O 和处理工具                          │
    │  shot_detect.py      — 镜头边界检测 (TransNetV2 / PySceneDetect)   │
    │  distributed.py      — 单 GPU 推理工具                              │
    └─────────────────────────────────────────────────────────────────────┘

改动点 (通过 flag 启用):
    --inversion    Flow Matching Inversion (从参考视频反演噪声)
    --svd          SVD 两阶段滤波 (空间去内容 + 时间保运动)
    --blend        噪声混合 (η = √α·η_temporal + √(1-α)·η_random)
    --velocity     Velocity Field Matching (Δe embedding 注入, 轻量版)
    --attn_inject  Self-Attention K/V Injection (参考视频注意力注入)
    --iter N       迭代 VLM 优化 (N轮反馈循环)
    --midpoint     二阶中点法 ODE 求解器 (替代 Euler)
    --composite    三面板垂直拼接 (ref|prev|current 送 VLM 对比)

组合示例:
    baseline:     无任何 flag → caption + 一次生成
    +noise_prior: --inversion --svd --blend → 噪声先验引导
    +velocity:    --inversion --velocity → Δe embedding 注入 (需要 inversion)
    +full:        --inversion --svd --blend --velocity --iter 10 → 全功能

用法:
    python run.py --video ref.mp4 --caption "..." --inversion --svd --blend --velocity
"""

__version__ = "7.0.0"
__task__ = "video_reproduction"

from .pipeline import PFlowConfig, PFlowPipeline
from .velocity_matching import VelocityMatcher
from .attn_inject import AttnInjector, AttnInjectConfig, AttentionKVCache
from .shot_detect import ShotDetector, detect_shots, split_video_to_shots

__all__ = [
    "PFlowConfig", "PFlowPipeline", "VelocityMatcher",
    "AttnInjector", "AttnInjectConfig", "AttentionKVCache",
    "ShotDetector", "detect_shots", "split_video_to_shots",
]
