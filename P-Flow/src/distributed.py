"""
Single-GPU Inference Utilities for P-Flow on A800 (1x 80GB).

Supports:
1. Single A800 with enable_model_cpu_offload (14B fits ~40GB VRAM)
2. VAE slicing + tiling for memory efficiency
3. Sequential video-by-video processing

Hardware: 1x A800-80GB
- Wan 2.1-14B: ~28GB in bfloat16
- With cpu_offload: components loaded on-demand, peak ~40GB
- Video generation: ~90-120s per video (single card)
"""

import os
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_single_gpu():
    """
    Setup single-GPU environment for A800 inference.
    Call this before loading any models.

    Returns:
        Device string ("cuda" or "cpu").
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name} ({props.total_mem / 1024**3:.1f} GB)")
    if num_gpus > 1:
        logger.info(f"  {num_gpus} GPUs detected, but using only GPU 0 (single-card mode)")

    # Set memory allocation strategy
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Use only GPU 0
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    return "cuda"


def load_model_single_gpu(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    model_type: str = "t2v",
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = True,
) -> Any:
    """
    Load Wan 2.1-14B model on single A800 with CPU offload.

    The 14B model (~28GB bfloat16) fits on a single A800-80GB with CPU offload:
    - Model components are loaded to GPU on-demand during inference
    - Peak VRAM usage: ~40-50GB
    - Requires sufficient system RAM (~32GB+)

    Args:
        model_path: Path to model weights.
        dtype: Model dtype (bfloat16 recommended for A800).
        model_type: "t2v" for Text-to-Video, "i2v" for Image-to-Video.
        enable_vae_slicing: Enable VAE slicing for memory efficiency.
        enable_vae_tiling: Enable VAE tiling for large resolutions.

    Returns:
        Loaded pipeline ready for inference.
    """
    logger.info(f"Loading Wan 2.1-14B ({model_type}) from: {model_path}")
    logger.info(f"  Mode: single GPU + CPU offload, dtype={dtype}")

    if model_type == "i2v":
        try:
            from diffusers import WanImageToVideoPipeline
            pipe = WanImageToVideoPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        except (ImportError, ValueError):
            from diffusers import WanPipeline
            pipe = WanPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
    else:
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
        )

    # Use CPU offload: components are moved to GPU on-demand
    # This allows 14B model to fit on single 80GB card
    pipe.enable_model_cpu_offload()
    logger.info("  CPU offload enabled (components loaded to GPU on-demand)")

    # Memory optimizations
    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        logger.info("  VAE slicing enabled")

    if enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        logger.info("  VAE tiling enabled")

    # Report memory after loading
    _log_gpu_memory()

    return pipe


def _log_gpu_memory():
    """Log GPU memory usage after model loading."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"  GPU 0: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def cleanup_gpu_memory():
    """Force cleanup of GPU memory between experiments."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# Keep old names as aliases for compatibility
setup_distributed_env = setup_single_gpu
load_model_distributed = load_model_single_gpu
