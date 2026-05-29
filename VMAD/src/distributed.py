"""
Single-GPU Inference Utilities for Wan2.1-1.3B.

硬件: 1x RTX 4090 (24GB) 或 A800
- Wan 2.1-1.3B: ~2.6GB in bfloat16, 全模型放 GPU
- Peak VRAM: ~12-16GB (81 frames 480p)
- 无需 CPU offload
"""

import os
import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def setup_single_gpu() -> str:
    """
    配置单卡推理环境。

    Returns:
        Device string ("cuda" or "cpu")
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return "cpu"

    props = torch.cuda.get_device_properties(0)
    logger.info(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    加载 Wan 2.1-1.3B 到单卡。

    Args:
        model_path: 模型权重路径
        dtype: 模型精度
        model_type: "t2v" 或 "i2v"
        enable_vae_slicing: VAE 切片 (省显存)
        enable_vae_tiling: VAE 分块 (大分辨率)

    Returns:
        加载好的 pipeline
    """
    logger.info(f"Loading Wan 2.1-1.3B ({model_type}) from: {model_path}")

    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to("cuda")
    logger.info("  Model loaded to GPU")

    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()

    _log_gpu_memory()
    return pipe


def _log_gpu_memory():
    """打印 GPU 显存使用。"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"  GPU 0: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def cleanup_gpu_memory():
    """强制清理 GPU 显存。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
