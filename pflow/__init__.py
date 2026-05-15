"""
P-Flow: A Training-Free Framework for Customizing Dynamic Visual Effects
via Test-Time Prompt Optimization.

Paper: arXiv:2603.22091

Includes VISTA-style optimizer for ablation experiments.
VISTA Paper: arXiv:2510.15831
"""

from .pipeline import PFlowPipeline
from .noise_prior import NoisePriorEnhancement
from .prompt_optimizer import PromptOptimizer
from .vista_optimizer import VISTAOptimizer
from .trajectory import TrajectoryManager

__version__ = "0.1.0"
__all__ = [
    "PFlowPipeline",
    "NoisePriorEnhancement",
    "PromptOptimizer",
    "VISTAOptimizer",
    "TrajectoryManager",
]
