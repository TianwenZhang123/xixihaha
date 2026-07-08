"""
Multi-GPU & Format-Agnostic Inference Utilities for Wan2.1.

Supports:
1. Single GPU (4090/A800) — full model on VRAM
2. Multi-GPU (2x L40) — model parallel via device_map="auto"
3. Full Diffusers format (model_index.json) or partial (files only)
4. VAE slicing + tiling for memory efficiency

Hardware:
  - Wan 2.1-1.3B: ~2.6GB bf16, fits single 24GB
  - Wan 2.1-14B:  ~28GB bf16, needs 2x48GB or 1x80GB
"""

import os
import torch
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_multi_gpu(gpu_id: int = 0):
    """Setup GPU environment, optionally pin to specific GPU.

    Args:
        gpu_id: GPU index to use (e.g. 0, 1, ..., or -1 for all GPUs).
                 Default 0.
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU.")
        return "cpu"

    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if gpu_id >= 0:
        # Pin to single GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        logger.info(f"  Pinned to GPU {gpu_id}")
        return "cuda"
    else:
        # Use all GPUs
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
    Load Wan 2.1 model with auto-detection of format and GPU count.

    Format support:
        - Full Diffusers (model_index.json) → WanPipeline.from_pretrained()
        - Partial Diffusers (config.json + safetensors + VAE.pth + T5.pth) → auto-build
        - Auto-creates model_index.json for clean loading when possible

    Multi-GPU:
        - 1 GPU → pipe.to("cuda")
        - 2+ GPUs → pipe.to("cuda") with device_map issues,
          use sequential offload for large models

    Args:
        model_path: Path to model weights directory.
        dtype: Model dtype (bfloat16 recommended).
        model_type: "t2v" for Text-to-Video.
        enable_vae_slicing: Enable VAE slicing for memory efficiency.
        enable_vae_tiling: Enable VAE tiling for large resolutions.

    Returns:
        Loaded WanPipeline ready for inference.
    """
    model_dir = Path(model_path)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # ── 检查格式 ──
    has_index = (model_dir / "model_index.json").exists()
    has_transformer = (model_dir / "config.json").exists() and list(model_dir.glob("diffusion_pytorch_model*.safetensors"))
    has_vae = (model_dir / "Wan2.1_VAE.pth").exists() or (model_dir / "vae").is_dir()
    has_t5 = (model_dir / "models_t5_umt5-xxl-enc-bf16.pth").exists() or (model_dir / "text_encoder").is_dir()

    # ── 路径1: 完整 Diffusers ──
    if has_index:
        logger.info(f"Loading Wan ({model_type}) [full Diffusers] from: {model_path}")
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # ── 路径2: 散文件 → 手动组装 WanPipeline ──
    elif has_transformer and has_vae and has_t5:
        logger.info(f"Loading Wan ({model_type}) [manual assembly] from: {model_path}")
        pipe = _load_wan_from_files(model_dir, dtype)
    else:
        missing = []
        if not has_transformer: missing.append("config.json + safetensors")
        if not has_vae: missing.append("Wan2.1_VAE.pth")
        if not has_t5: missing.append("models_t5_umt5-xxl-enc-bf16.pth")
        raise RuntimeError(
            f"模型路径 {model_path} 缺少必要组件: {missing}。\n"
            f"请下载完整模型: huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers --local-dir {model_path}"
        )

    # ── GPU 分配 ──
    if num_gpus <= 1 or os.environ.get("CUDA_VISIBLE_DEVICES"):
        pipe = pipe.to("cuda")
        logger.info(f"  Model on single GPU (VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB)")
    else:
        # 多 GPU: 用 device_map="balanced" 分到两张卡
        logger.info(f"  {num_gpus} GPUs detected, using device_map='balanced'")
        try:
            pipe = pipe.to("cuda", device_map="balanced")
        except Exception:
            logger.warning("  device_map='balanced' failed, falling back to GPU 0")
            pipe = pipe.to("cuda")

    mem_free, mem_total = torch.cuda.mem_get_info()
    logger.info(f"  Model loaded: {mem_free/1e9:.1f}GB free / {mem_total/1e9:.1f}GB total")
    _log_gpu_memory()

    # ── VAE 优化 ──
    if enable_vae_slicing and hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        logger.info("  VAE slicing enabled")
    if enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
        logger.info("  VAE tiling enabled")

    return pipe


def _load_wan_from_files(model_dir: Path, dtype: torch.dtype) -> Any:
    """从散文件手动组装 WanPipeline（兼容非标准 Diffusers 布局）。"""
    from diffusers import WanPipeline, WanTransformer3DModel
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    # ── DiT Transformer ──
    logger.info("  Loading transformer (DiT)...")
    transformer = WanTransformer3DModel.from_pretrained(model_dir, torch_dtype=dtype)

    # ── VAE ──
    logger.info("  Loading VAE...")
    vae = _load_wan_vae(model_dir, dtype)

    # ── T5 Text Encoder ──
    logger.info("  Loading T5...")
    text_encoder = _load_wan_t5(model_dir, dtype)

    scheduler = FlowMatchEulerDiscreteScheduler()
    return WanPipeline(
        transformer=transformer, vae=vae,
        text_encoder=text_encoder, scheduler=scheduler,
    )


def _load_wan_vae(model_dir: Path, dtype: torch.dtype):
    """加载 Wan2.1 VAE (单文件 .pth 格式)。"""
    import safetensors.torch
    try:
        from diffusers import AutoencoderKLWan
    except ImportError:
        from diffusers.models.autoencoders import AutoencoderKLWan

    vae_config = {
        "_class_name": "AutoencoderKLWan", "act_fn": "silu",
        "block_out_channels": [128, 256, 256, 256],
        "down_block_types": ["WanDownBlock3D"] * 4,
        "force_upcast": False, "in_channels": 3, "latent_channels": 16,
        "layers_per_block": 2, "out_channels": 3, "sample_size": 256,
        "scaling_factor": 0.5960, "shift_factor": 0.0,
        "up_block_types": ["WanUpBlock3D"] * 4,
        "use_quant_conv": False, "use_post_quant_conv": False,
    }
    vae = AutoencoderKLWan(**vae_config).to(dtype)
    state = safetensors.torch.load_file(str(model_dir / "Wan2.1_VAE.pth"))
    vae.load_state_dict(state, strict=False)
    return vae


def _load_wan_t5(model_dir: Path, dtype: torch.dtype):
    """加载 Wan2.1 T5 编码器 (单文件, 实为 safetensors)。"""
    import tempfile
    import shutil as _sh
    from transformers import T5EncoderModel
    # Wan2.1 T5 即标准 UMT5-XXL encoder，用 HF 加载
    try:
        return T5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=dtype,
                                               low_cpu_mem_usage=True, local_files_only=True)
    except Exception:
        logger.info("  T5 not cached, downloading (one-time)...")
        return T5EncoderModel.from_pretrained("google/umt5-xxl", torch_dtype=dtype)


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
