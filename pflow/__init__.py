"""
P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects
via Test-Time Prompt Optimization.

Paper: arXiv:2603.22091

Paper-faithful implementation:
- Fixed i_max iterations (NO early stopping)
- NO confidence score
- Vertical composite layout (top/middle/bottom)
- VLM structured instruction (Listing 1 format)
- Noise Prior Enhancement (Flow Inversion + SVD Filter + Blend)
"""

from .pipeline import PFlowPipeline
from .noise_prior import NoisePriorEnhancement
from .prompt_optimizer import PromptOptimizer
from .vlm_client import VLMClient, MockVLMClient
from .trajectory import TrajectoryManager

__version__ = "1.0.0"
__all__ = [
    "PFlowPipeline",
    "NoisePriorEnhancement",
    "PromptOptimizer",
    "VLMClient",
    "MockVLMClient",
    "TrajectoryManager",
]
