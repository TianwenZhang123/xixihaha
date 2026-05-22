"""
Video Reproduction via Iterative Prompt Optimization + Noise Prior.

Adapted from P-Flow (arXiv:2603.22091) for faithful video reproduction.

Given a reference video, iteratively optimizes a T2V prompt so that the
generated video reproduces the reference as closely as possible.

Two mechanisms work together:
1. Noise Prior: encodes motion dynamics from reference (flow inversion + SVD)
2. Prompt Optimization: VLM compares videos, refines prompt each iteration

Core components:
- pipeline: Main orchestration (iterative loop)
- noise_prior: Flow Matching Inversion + SVD two-stage filtering
- svd_filter: Spatial removal + Temporal retention
- flow_matching: Euler integration for inversion (t=1->0)
- vlm_client: DashScope Qwen-VL integration
- prompt_optimizer: Vertical composite + VLM refinement
- video_utils: Video I/O and processing
- trajectory: History management
- distributed: Single-GPU inference utilities
"""

__version__ = "2.0.0"
__task__ = "video_reproduction"
