"""
Single-GPU Inference Utilities for Wan2.1-1.3B on 4090 (24GB).

Supports:
1. Single 4090 — model fits fully in VRAM (~2.6GB bfloat16)
2. VAE slicing + tiling for memory efficiency during video decode
3. Sequential video-by-video processing

Hardware: 1x RTX 4090 (24GB)
- Wan 2.1-1.3B: ~2.6GB in bfloat16
- No CPU offload needed — full model on GPU
- Peak VRAM during generation: ~12-16GB (with 81 frames 480p)
- Video generation: ~30-50s per video
"""

import os
import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def setup_single_gpu():
    """
    Setup single-GPU environment for 4090 inference.
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
    Load Wan 2.1-1.3B model on single 4090.

    The 1.3B model (~2.6GB bfloat16) fits comfortably on a 4090 (24GB).
    No CPU offload needed. Peak VRAM during generation: ~12-16GB.

    Args:
        model_path: Path to model weights.
        dtype: Model dtype (bfloat16 recommended).
        model_type: "t2v" for Text-to-Video, "i2v" for Image-to-Video.
        enable_vae_slicing: Enable VAE slicing for memory efficiency.
        enable_vae_tiling: Enable VAE tiling for large resolutions.

    Returns:
        Loaded pipeline ready for inference.
    """
    logger.info(f"Loading Wan 2.1-1.3B ({model_type}) from: {model_path}")
    logger.info(f"  Mode: single GPU (full model on VRAM), dtype={dtype}")

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

    # Move entire model to GPU — 1.3B fits comfortably
    pipe = pipe.to("cuda")
    logger.info("  Model loaded to GPU (no CPU offload needed for 1.3B)")

    # Memory optimizations for video decoding
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
