"""
Multi-GPU & Diffusers Format Inference Utilities for Wan2.1.

Supports:
1. Single GPU — full model on VRAM
2. Multi-GPU — pinned to specific GPU via --gpu
3. Full Diffusers format (model_index.json required)

Hardware:
  - Wan 2.1-1.3B: ~2.6GB bf16, fits single 24GB
  - Wan 2.1-14B:  ~28GB bf16, needs 48GB+ GPU
"""

import os
import torch
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_multi_gpu(gpu_id: int = 0):
    """Setup GPU environment, optionally pin to specific GPU."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"  Pinned to GPU {gpu_id}")
    else:
        logger.info(f"  Using all {num_gpus} GPUs")
    return "cuda"


def load_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    model_type: str = "t2v",
    enable_vae_slicing: bool = True,
    enable_vae_tiling: bool = True,
) -> Any:
    """
    Load Wan 2.1 model (Diffusers format required).

    Args:
        model_path: Path to Diffusers-format model directory (must contain model_index.json).
        dtype: Model dtype (bfloat16 recommended).
        model_type: "t2v" for Text-to-Video.
        enable_vae_slicing: Enable VAE slicing for memory efficiency.
        enable_vae_tiling: Enable VAE tiling for large resolutions.

    Returns:
        Loaded WanPipeline ready for inference.
    """
    model_dir = Path(model_path)
    if not (model_dir / "model_index.json").exists():
        raise RuntimeError(
            f"模型路径 {model_path} 缺少 model_index.json，需要 Diffusers 格式模型。\n"
            f"下载: huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir {model_path}"
        )

    logger.info(f"Loading Wan ({model_type}) from: {model_path}")
    from diffusers import WanPipeline, AutoencoderKLWan
    # 官方建议: VAE 用 float32 保证解码质量, 其余 bfloat16
    vae = AutoencoderKLWan.from_pretrained(model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_path, vae=vae, torch_dtype=dtype)

    # GPU loading (L40 44GB 不够时自动退到 CPU offload)
    try:
        pipe = pipe.to("cuda")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        logger.info("  GPU OOM, enabling model CPU offload...")
        pipe.enable_model_cpu_offload()
    mem_free, mem_total = torch.cuda.mem_get_info()
    logger.info(f"  Model loaded: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
    _log_gpu_memory()

    # VAE optimizations
    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        logger.info("  VAE slicing enabled")
    if enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        logger.info("  VAE tiling enabled")

    return pipe


def _log_gpu_memory():
    """Log GPU memory usage for all devices."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"  GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def cleanup_gpu_memory():
    """Force cleanup of GPU memory between experiments."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
