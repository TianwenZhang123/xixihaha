"""
P-Flow: Training-Free Visual Effects Customization via Test-Time Prompt Optimization.

Full reproduction targeting single A800 (80GB) with CPU offload.
Paper: arXiv:2603.22091

Core components:
- pipeline: Main Algorithm 1 orchestration
- noise_prior: Flow Matching Inversion + SVD two-stage filtering
- svd_filter: Spatial removal + Temporal retention
- flow_matching: Euler integration for inversion (t=1->0)
- vlm_client: DashScope Qwen-VL integration
- prompt_optimizer: Vertical composite + VLM refinement
- video_utils: Video I/O and processing
- trajectory: History management
- distributed: Single-GPU inference utilities
"""

__version__ = "1.0.0"
__paper__ = "arXiv:2603.22091"
