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
import json
import shutil
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

    # ── 路径2: 散文件 → 创建 model_index.json 后标准加载 ──
    elif has_transformer and has_vae and has_t5:
        logger.info(f"Loading Wan ({model_type}) [partial Diffusers, creating model_index.json] from: {model_path}")
        _create_model_index(model_dir)
        from diffusers import WanPipeline
        pipe = WanPipeline.from_pretrained(model_path, torch_dtype=dtype)
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


def _create_model_index(model_dir: Path):
    """为散文件模型创建 model_index.json，使 diffusers 能标准加载。
    
    模型目录结构:
        Wan2.1-T2V-14B/
        ├── config.json                          ← DiT config
        ├── diffusion_pytorch_model*.safetensors ← DiT weights
        ├── Wan2.1_VAE.pth                      ← VAE (单文件)
        └── models_t5_umt5-xxl-enc-bf16.pth     ← T5 (单文件)
    """
    index_path = model_dir / "model_index.json"
    if index_path.exists():
        return

    # 创建 vae/ 子目录 (软链接，不占磁盘)
    vae_dir = model_dir / "vae"
    if not (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        vae_dir.mkdir(exist_ok=True)
        os.symlink(model_dir / "Wan2.1_VAE.pth", vae_dir / "diffusion_pytorch_model.safetensors")
        vae_config = {
            "_class_name": "AutoencoderKLWan",
            "act_fn": "silu", "block_out_channels": [128, 256, 256, 256],
            "down_block_types": ["WanDownBlock3D"] * 4,
            "force_upcast": False, "in_channels": 3, "latent_channels": 16,
            "layers_per_block": 2, "out_channels": 3, "sample_size": 256,
            "scaling_factor": 0.5960, "shift_factor": 0.0,
            "up_block_types": ["WanUpBlock3D"] * 4,
            "use_quant_conv": False, "use_post_quant_conv": False,
        }
        json.dump(vae_config, (vae_dir / "config.json").open("w"), indent=2)

    # 创建 text_encoder/ 子目录 (软链接)
    te_dir = model_dir / "text_encoder"
    if not (te_dir / "diffusion_pytorch_model.safetensors").exists():
        te_dir.mkdir(exist_ok=True)
        os.symlink(model_dir / "models_t5_umt5-xxl-enc-bf16.pth", te_dir / "diffusion_pytorch_model.safetensors")
        t5_config = {
            "_class_name": "T5EncoderModel",
            "d_model": 4096, "d_kv": 64, "d_ff": 10240, "num_layers": 24,
            "num_heads": 64, "dropout_rate": 0.1, "dense_act_fn": "gelu_pytorch_tanh",
            "is_gated_act": True, "feed_forward_proj": "gated-gelu",
            "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
            "tie_word_embeddings": False, "vocab_size": 32128,
        }
        json.dump(t5_config, (te_dir / "config.json").open("w"), indent=2)

    # 创建 transformer/ 子目录 (软链接)
    tr_dir = model_dir / "transformer"
    tr_dir.mkdir(exist_ok=True)
    if not (tr_dir / "config.json").exists():
        os.symlink(model_dir / "config.json", tr_dir / "config.json")
    for f in sorted(model_dir.glob("diffusion_pytorch_model*.safetensors")):
        link = tr_dir / f.name
        if not link.exists():
            os.symlink(f, link)

    # 写 model_index.json
    index = {
        "_class_name": "WanPipeline",
        "_diffusers_version": "0.31.0",
        "transformer": ["transformer"],
        "vae": ["vae"],
        "text_encoder": ["text_encoder"],
    }
    json.dump(index, index_path.open("w"), indent=2)
    logger.info("  model_index.json created (VAE/T5/Transformer subdirs)")


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
