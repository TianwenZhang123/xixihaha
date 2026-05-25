"""
Video Reproduction via Iterative Prompt Optimization + Noise Prior.

Adapted from P-Flow (arXiv:2603.22091) for faithful video reproduction.

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    run_batch.py                          │
    │         --method baseline  |  --method pflow            │
    ├─────────────────────────────┬───────────────────────────┤
    │     BaselinePipeline        │      PFlowEnhancer        │
    │  (src/baseline.py)          │   (src/enhancement.py)    │
    │                             │                           │
    │  VLM Caption → Wan Generate │  + Noise Prior            │
    │  (one-shot, no iteration)   │  + Iterative Optimization │
    │                             │  + Best Selection         │
    └─────────────────────────────┴───────────────────────────┘

Two methods:
    1. Baseline (Direct Caption): VLM看视频 → 一次性caption → Wan生成
    2. P-Flow (Ours): VLM caption → [Noise Prior + 迭代VLM优化] → 最优视频

Core components:
- baseline: Direct caption → generation pipeline (no optimization)
- enhancement: P-Flow iterative optimization module (pluggable)
- pipeline: Legacy full pipeline (kept for backward compatibility)
- noise_prior: Flow Matching Inversion + SVD two-stage filtering
- svd_filter: Spatial removal + Temporal retention
- flow_matching: Euler integration for inversion (t=1->0)
- vlm_client: Local Qwen2.5-VL-7B / DashScope API
- prompt_optimizer: Vertical composite + VLM refinement
- video_utils: Video I/O and processing
- trajectory: History management
- distributed: Single-GPU inference utilities
"""

__version__ = "3.0.0"
__task__ = "video_reproduction"
